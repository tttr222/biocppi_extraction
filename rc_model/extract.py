#!/usr/bin/env python
import sys, os, random, pickle, re, fileinput
import argparse
import pandas as pd
import numpy as np
from model import WordCNN

parser = argparse.ArgumentParser(description='Train and evaluate BiLSTM on a given dataset')
parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
parser.add_argument('--datapath', dest='datapath', type=str,
                    default='../fold_train', 
                    help='path to the datasets')
parser.add_argument('--seed', dest='seed', type=int, 
                    default=1, help='seed for training')

num_ensembles = 10

#protein I-MISC N
#Atg17 I-MISC N

def main(args):
    print >> sys.stderr, args
    random.seed(args.seed)
    assert(os.path.isdir(args.datapath))
    fold_dir = args.datapath
    
    lines = []
    for line in fileinput.input(args.files):
        lines.append(line)
    
    tokens, pmids = load_annotated(lines)
    X_test = extract_candidates(tokens, pmids)
    #print len(pmids), 'articles'
    #print len(X_test), 'examples'
    
    vocab_cache = os.path.join(args.datapath,'word_vocab.mppi.txt')
    
    with open(vocab_cache,'r') as f:
        word_vocab = pickle.load(f)
        #print "Loaded vocab from", vocab_cache
        
    #print "Vocab", len(word_vocab),  word_vocab[:10]
    model_name = 'saved_model_ppi'
    labels = ['Negative','Positive']
    
    proba_cumulative = np.zeros((len(X_test),len(labels)))
    
    for j in range(num_ensembles):
        m = WordCNN(labels,word_vocab,
                    word_embeddings=None,
                    max_sequence_length=1000)
        
        save_path = '{}/{}/model_{}'.format(args.datapath,model_name,j)
        m.restore(save_path)
        print >> sys.stderr, "Restoring model {}/{}".format(j+1,num_ensembles)        

        proba_cumulative += m.predict_proba(X_test)
    
    proba_cumulative /= num_ensembles
    y_pred = np.argmax(proba_cumulative,axis=1)
    positive_pct = [ y[labels.index('Positive')] for y in proba_cumulative ]
    
    pair_scores = {}
    for inst, pred in zip(X_test, positive_pct):
        pmid, a, b, text = inst
        key = ','.join([pmid] + sorted([a,b]))
        if key not in pair_scores:
            pair_scores[key] = []
        
        pair_scores[key].append(pred)
    
    output_pairs = {}
    
    for k in sorted(pair_scores.keys()):
        #print k, pair_scores[k], np.mean(pair_scores[k])
        meanscore = np.mean(pair_scores[k])
        pmid, a, b = k.split(',')
        
        if pmid not in output_pairs:
            output_pairs[pmid] = []
        
        if len(pair_scores[k]) == 1:
            pair_scores[k].append(pair_scores[k][0])
        
        print '\t'.join([pmid, a, b, str(pair_scores[k][0]), str(pair_scores[k][1])])
        

def extract_candidates(tokens, pmids):
    examples = []
    for pmid in pmids:
        genes = []
        for sentence in tokens[pmid]:
            genes += [ gid for _, _, gid in sentence if gid not in ['X','N'] ]
        
        for a in set(genes):
            for b in set(genes):
                inst = (pmid, a, b, generate_input(tokens[pmid], a, b))
                if inst not in examples:
                    examples.append(inst)
    
    return examples
            
def generate_input(tokens, candidate_a, candidate_b):
    out_context = []
    for sentence in tokens:
        out_sentence = []
        for tok in sentence:
            w, _, gid = tok
            
            if candidate_a == candidate_b and gid == candidate_a:
                if len(out_sentence) > 0 and out_sentence[-1] != 'GENE_S':
                    out_sentence.append('GENE_S')
            elif gid == candidate_a:
                if len(out_sentence) > 0 and out_sentence[-1] != 'GENE_A':
                    out_sentence.append('GENE_A')
            elif gid == candidate_b:
                if len(out_sentence) > 0 and out_sentence[-1] != 'GENE_B':
                    out_sentence.append('GENE_B')
            elif gid != 'X' and gid != 'N':
                if len(out_sentence) > 0 and out_sentence[-1] != 'GENE_N':
                    out_sentence.append('GENE_N')
            else:
                out_sentence.append(w)
        
        out_context.append(out_sentence)

    return out_context

def load_annotated(lines):
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
    
    return tokens, pmids
    

if __name__ == '__main__':
    main(parser.parse_args())
