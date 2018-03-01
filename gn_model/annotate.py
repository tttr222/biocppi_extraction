#!/usr/bin/env python
import sys, os, random, pickle, json, codecs, re, fileinput
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import argparse
from multiprocessing import Pool
from gene_norm_model import GeneNormalizer

parser = argparse.ArgumentParser(description='Train and evaluate BiLSTM on a given dataset')
parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
parser.add_argument('--datapath', dest='datapath', type=str,
                    default='../corpus_train', 
                    help='path to the datasets')
parser.add_argument('--cachepath', dest='cachepath', type=str,
                    default='.', help='path to the cachefiles')
parser.add_argument('--seed', dest='seed', type=int, 
                    default=1, help='seed for training')

num_ensembles = 3
num_epoch = 40
batch_size = 500 # 8 without aux

def main(args):
    print >> sys.stderr, args
    random.seed(args.seed)

    lines = []
    for line in fileinput.input(args.files):
        lines.append(line)
    
    key = None
    tokens = {}
    curlist = []
    pmids = []
    for l in lines:
        if l.startswith('###'):
            key = l[3:].rstrip()
            pmids.append(key)
            continue
        
        if key not in tokens:
            tokens[key] = []
        
        if l.strip() == '':
            if len(curlist) > 0:
                tokens[key].append(curlist)
            
            curlist = []
        else:
            curlist.append(tuple(l.strip().split(' ')))
    
    gene_cache_file = os.path.join(args.cachepath,'gene_map.cache')
    pmid_cache_file = os.path.join(args.cachepath,'pmid_map.cache')
    pubtator_cache_file = os.path.join(args.cachepath,'pubtator_pmid_map.cache')
    
    gn_model = GeneNormalizer(gene_cache_file, pmid_cache_file, pubtator_cache_file)
    #gn_model.fit(X_train, y_train, X_aux, y_aux)
    
    for ij, pmid in enumerate(pmids):
        print >> sys.stderr, "Processing {}/{} {}".format(ij+1,len(pmids),pmid)
        print '###' + pmid
        print ''
        
        for line in tokens[pmid]:
            candidates = []
            span = []
            for i, (tok, tag) in enumerate(line):
                if tag == 'I-MISC':
                    span.append((i,(tok,tag)))
                elif tag == 'B-MISC':
                    if len(span) > 0:
                        candidates.append(span)
                        span = []
                    
                    span.append((i,(tok,tag)))
                elif tag == 'O':
                    if len(span) > 0:
                        candidates.append(span)
                        span = []
            
            gene_pos = {}
            for span in candidates:
                positions, toktags = zip(*span)
                x = ' '.join(tok for tok, tag in toktags)
                gene_id = gn_model.predict([(pmid,x)])[0]
                if gene_id is None:
                    gene_id = 'N'
                
                for p in positions:
                    gene_pos[p] = gene_id
                    
            for i, (tok, tag) in enumerate(line):
                if i in gene_pos:
                    print tok, tag, gene_pos[i]
                else:
                    print tok, tag, 'X'
            
            print ''
        
        print ''

def load_file(fname, load_labels = False):
    if load_labels:
        columns = ('pmid', 'text','label')
    else:
        columns = ('pmid', 'text')
    
    frame = pd.read_csv(fname,header=None,names=columns, dtype={'text': str, 'label': str},
                        usecols=columns,delimiter='\t')
                        
    return frame

def load_lexicon(fname):
    training_data = []
    with open(fname,'r') as f:
        for l in f:
            items = l.strip().split('\t')
            for span in items[1:]:
                training_data.append((span,items[0]))
    frame = pd.DataFrame(training_data)
    frame.columns = ('text','label')
    return frame

if __name__ == '__main__':
    main(parser.parse_args())
