#!/usr/bin/env python
import sys, os, random, pickle, json, codecs, re, fileinput
import numpy as np
import pandas as pd
import argparse
from nltk.tokenize import wordpunct_tokenize

parser = argparse.ArgumentParser(description='Train and evaluate BiLSTM on a given dataset')
parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
parser.add_argument('--datapath', dest='datapath', type=str,
                    default='../fold_train', 
                    help='path to the datasets')
parser.add_argument('--seed', dest='seed', type=int, 
                    default=1, help='seed for training')


def find_x_in_y(x,y, offset = 0):
    for i in range(offset,len(y)-len(x)):
        if x == y[i:i+len(x)]:
            return i, i + len(x)
    
    return None

def quick_search(sentence,spans):
    found = []
    for x in spans:
        offset = 0
        new_offset = True
        while new_offset == True:
            out = find_x_in_y(x,sentence,offset)
            if out is not None:
                found.append(out)
                new_offset = True
                offset = out[1]
            else:
                new_offset = False
    
    if len(found) == 0:
        return []
    
    longest_first = sorted(found, key=lambda x: (x[1] - x[0],-x[0]), reverse=True)
    
    final = [ longest_first.pop(0) ]
    while len(longest_first) > 0:
        x0, y0 = longest_first.pop(0)
        overlap = False
        for x1, y1 in final:
            if len(set(range(x0,y0)).intersection(range(x1,y1))) > 0:
                overlap = True
                break
        
        if not overlap:
            final.append((x0,y0))

    return final

def main(args):
    print >> sys.stderr, args
    random.seed(args.seed)
    assert(os.path.isdir(args.datapath))
    
    lexicon = load_lexicon(os.path.join(args.datapath,'entrezGeneLexicon.list'))
    
    spans = [ wordpunct_tokenize(x) for x in set(lexicon['text'].tolist()) ]
    sorted_spans = sorted(spans, key=lambda x: len(x),reverse=True)
    
    freq = []
    for z in sorted_spans:
        freq.append(len(z))
    
    cutoff = np.percentile(freq,90)
    
    spans = []
    for s in sorted_spans:
        if len(s) < cutoff:
            if len(s) == 1:
                if len(s[0]) > 1:
                    spans.append(s)
            else:
                spans.append(s)
    
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

    corrections = 0
    for ij, pmid in enumerate(pmids):
        print >> sys.stderr, "Processing {}/{} {}".format(ij+1,len(pmids),pmid)
        print '###' + pmid
        print ''
        for line in tokens[pmid]:
            sent = [ w for w, t in line ]
            tags = [ t for w, t in line ]
            found = quick_search(sent,spans)
            
            for (start,end) in found:
                tagged_seq = ['I-MISC'] * (end-start)
                lexicon = ['I-MISC'] * (end-start)
                if tags[start:end] != tagged_seq:
                    corrections += 1
                    tags[start:end] = lexicon
                
                if tags[end] == 'I-MISC':
                    tags[end] = 'B-MISC'
                
                if start > 0 and tags[start-1] == 'I-MISC':
                    tags[start-1] = 'B-MISC'
            
            for i, (tok, tag) in enumerate(zip(sent,tags)):
                print tok, tag
            
            print ''
        
        print ''
    
    print >> sys.stderr, corrections

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
