#!/usr/bin/env python
import sys, os, random, pickle, re, fileinput, argparse, codecs, time
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from model import WordCNN

parser = argparse.ArgumentParser(description='Train and evaluate BiLSTM on a given dataset')
parser.add_argument('--datapath', dest='datapath', type=str,
                    default='../fold_train', 
                    help='path to the datasets')
parser.add_argument('--embeddings', dest='embeddings_path', type=str,
                    default='../embeddings/PubMed-w2v.txt', 
                    help='path to the testing dataset')
parser.add_argument('--optimizer', dest='optimizer', type=str,
                    default='default', 
                    help='choose the optimizer: default, rmsprop, adagrad, adam.')
parser.add_argument('--batch-size', dest='batch_size', type=int, 
                    default=24, help='number of instances in a minibatch')
parser.add_argument('--num-epoch', dest='num_epoch', type=int, 
                    default=30, help='number of passes over the training set')
parser.add_argument('--seed', dest='seed', type=int, 
                    default=1, help='seed for training')

num_ensembles = 10
load_fuzzy = False

def main(args):
    print >> sys.stderr, args
    random.seed(args.seed)
    assert(os.path.isdir(args.datapath))
    fold_dir = args.datapath
    
    X_train, y_train = load_dataset(fold_dir,'train.ppi.txt',fuzzy=load_fuzzy)
    
    print "Positives", y_train.count('Positive'), "Negatives", y_train.count('Negative')
    
    word_vocab = [ '<ZERO>', 'UNK' ]
    for xx in X_train:
        _, _, _, text = xx
        for s in text:
            for w in s:
                if w.lower() not in word_vocab:
                    word_vocab.append(w.lower())
    
    print "Vocab", len(word_vocab),  word_vocab[:10]
    vocab_cache = os.path.join(args.datapath,'word_vocab.mppi.txt')
    with open(vocab_cache,'w') as f:
        print "Saved vocab to", vocab_cache
        pickle.dump(word_vocab,f)
    
    embeddings = load_embeddings(args.embeddings_path, word_vocab, 200)
    
    labels = ['Negative','Positive']
    
    model_name = 'saved_model_ppi'
    if not os.path.exists('{}/scratch'.format(args.datapath)):
        os.mkdir('{}/scratch'.format(args.datapath))
            
    if os.path.exists('{}/{}'.format(args.datapath,model_name)):
        os.rename('{}/{}'.format(args.datapath,model_name),
            '{}/{}_{}'.format(args.datapath,model_name,int(time.time())))
        
    os.mkdir('{}/{}'.format(args.datapath,model_name))
    
    for j in range(num_ensembles):
        m = WordCNN(labels,word_vocab,
                    word_embeddings=embeddings,
                    max_sequence_length=1000)
    
        m.fit(X_train,y_train, num_epoch=args.num_epoch, batch_size=args.batch_size, seed=j)
        
        save_path = '{}/{}/model_{}'.format(args.datapath,model_name,j)
        m.save(save_path)
        print "Saved model {} to {}".format(j,save_path)

def load_dataset(fold_dir,fname,fuzzy=False):
    lines = []
    with open(os.path.join(fold_dir,fname),'r') as f:
        for line in f:
            lines.append(line)
    
    tokens, relations, pmids = load_annotated(lines)
    examples = extract_candidates(tokens, relations, pmids)
    print "Loading dataset from {}: {} articles, {} examples".format(fname, len(pmids), len(examples))
    
    '''
    pos = 0
    neg = 0
    pos_all = 0
    neg_all = 0
    for (pid, a, b, x), label in examples:
        print pid, a, b, label
        if a != b:
            if label == 'Positive':
                pos += 1
            else:
                neg += 1
        
        if label == 'Positive':
            pos_all += 1
        else:
            neg_all += 1
        
    print pos_all, neg_all
    print pos, neg
    '''
    
    if fuzzy:
        lines = []
        fuzzy_file = fname[:-4] + '.fuzzy.txt'
        with open(os.path.join(fold_dir,fuzzy_file),'r') as f:
            for line in f:
                lines.append(line)
        
        tokens2, _, pmids2 = load_annotated(lines)
        examples2 = extract_candidates(tokens2, relations, pmids)
        examples += examples2
        print "Loading dataset from {}: {} articles, {} examples".format(fuzzy_file, len(pmids2), len(examples2))
    
    return zip(*examples)

def extract_candidates(tokens, relations, pmids):
    examples = []
    for pmid in pmids:
        genes = []
        for sentence in tokens[pmid]:
            genes += [ gid for _, _, gid in sentence if gid not in ['X','N'] ]
        
        for a in set(genes):
            for b in set(genes):
                if (a,b) in relations[pmid] or (b,a) in relations[pmid]:
                    label = 'Positive'
                else:
                    label = 'Negative'
                                        
                inst = (pmid, a, b, generate_input(tokens[pmid], a, b))
                if (inst,label) not in examples:
                    examples.append((inst,label))
    
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
    relations = {}
    curlist = []
    pmids = []
    
    for l in lines:
        if l.startswith('###'):
            items = l.rstrip().split('\t')
            key = items[0][3:].rstrip()
            pmids.append(key)
            relations[key] = [ tuple(x.split(',')) for x in items[1:] ]
            continue
        
        if key not in tokens:
            tokens[key] = []
        
        if l.strip() == '':
            if len(curlist) > 0:
                tokens[key].append(curlist)
            
            curlist = []
        else:
            curlist.append(tuple(l.strip().split(' ')))
    
    return tokens, relations, pmids

def load_embeddings(fname, vocab, dim=200):
    cached = 'scratch/embeddings_{}.npy'.format(abs(hash(' '.join(vocab))))
    
    if not os.path.exists(cached):
        weight_matrix = np.random.uniform(-0.05, 0.05, (len(vocab),dim)).astype(np.float32)
        ct = 0
        
        ctime = time.time()
        print 'Loading embeddings..',
        with codecs.open(fname, encoding='utf-8') as f:
            data = f.read()
        print '{}s'.format(int(time.time()-ctime))
        
        ctime = time.time()
        print 'Organizing embeddings..',
        lookup = {}
        for line in data.split('\n'):
            if line.strip() == '':
                continue
            
            word, vec = line.split(u' ', 1)
            lookup[word] = vec
        print '{}s'.format(int(time.time()-ctime))
            
        for word in vocab:
            if word not in lookup:
                continue
            
            vec = lookup[word]
            idx = vocab.index(word)
            vec = np.array(vec.split(), dtype=np.float32)
            weight_matrix[idx,:dim] = vec[:dim]
            ct += 1
            if ct % 33 == 0:
                sys.stdout.write('Vectorizing embeddings {}/{}   \r'.format(ct, len(vocab)))
        
        print "Loaded {}/{} embedding vectors".format(ct, len(vocab))
        np.save(cached,weight_matrix)
    else:
        weight_matrix = np.load(cached)
    
    print "Loaded weight matrix {}..".format(weight_matrix.shape)
    
    return weight_matrix

if __name__ == '__main__':
    main(parser.parse_args())
