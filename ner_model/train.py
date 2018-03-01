#!/usr/bin/env python
import sys, os, random, pickle, json, codecs, time
import numpy as np
from model import BiLSTM
import sklearn.metrics as skm
import argparse
import evaluation

parser = argparse.ArgumentParser(description='Train BiLSTM on a given dataset')
parser.add_argument('--datapath', dest='datapath', type=str,
                    default='../corpus_train', 
                    help='path to the datasets')
parser.add_argument('--embeddings', dest='embeddings_path', type=str,
                    default='../embeddings/PubMed-w2v.txt', 
                    help='path to the testing dataset')
parser.add_argument('--optimizer', dest='optimizer', type=str,
                    default='default', 
                    help='choose the optimizer: default, rmsprop, adagrad, adam.')
parser.add_argument('--batch-size', dest='batch_size', type=int, 
                    default=-1, help='number of instances in a minibatch')
parser.add_argument('--num-iterations', dest='num_iterations', type=int, 
                    default=10000, help='number of iterations')
parser.add_argument('--num-it-per-ckpt', dest='num_it_per_ckpt', type=int, 
                    default=100, help='number of iterations per checkpoint')
parser.add_argument('--learning-rate', dest='learning_rate', type=str,
                    default='default', help='learning rate')
parser.add_argument('--embedding-factor', dest='embedding_factor', type=float,
                    default=1.0, help='learning rate multiplier for embeddings')
parser.add_argument('--decay', dest='decay_rate', type=float,
                    default=0.95, help='exponential decay for learning rate')
parser.add_argument('--keep-prob', dest='keep_prob', type=float,
                    default=0.7, help='dropout keep rate')
parser.add_argument('--num-cores', dest='num_cores', type=int, 
                    default=20, help='seed for training')
parser.add_argument('--seed', dest='seed', type=int, 
                    default=2, help='seed for training')

num_ensembles = 10

def main(args):
    print "Running BiLSTM model"
    print args
    random.seed(args.seed)
    
    trainset = []
    devset = []

    print >> sys.stderr, "Loading dataset.."
    assert(os.path.isdir(args.datapath))
    
    word_vocab = []
    for fname in sorted(os.listdir(args.datapath)):
        if os.path.isdir(fname): 
            continue
        
        #if fname.endswith('train.ner.txt'):
        if fname.endswith('.ppi.txt'):
            print fname
            dataset, vocab = load_dataset(os.path.join(args.datapath,fname))
            word_vocab += vocab
            trainset += dataset
        
            print >> sys.stderr, "Loaded {} instances with a vocab size of {} from {}".format(len(dataset),len(vocab),fname)
    
    print "Loaded {} instances from data set".format(len(trainset))
    
    word_vocab = sorted(set(word_vocab))
    vocab_cache = os.path.join(args.datapath,'word_vocab.ner.txt')
    with open(vocab_cache,'w') as f:
        print "Saved vocab to", vocab_cache
        pickle.dump(word_vocab,f)
    
    embeddings = load_embeddings(args.embeddings_path, word_vocab, 200)
    
    labels = ['B-MISC','I-MISC','O']
    
    model_name = 'saved_model_autumn'
    if not os.path.exists('{}/scratch'.format(args.datapath)):
        os.mkdir('{}/scratch'.format(args.datapath))
            
    if os.path.exists('{}/{}'.format(args.datapath,model_name)):
        os.rename('{}/{}'.format(args.datapath,model_name),
            '{}/{}_{}'.format(args.datapath,model_name,int(time.time())))
        
    os.mkdir('{}/{}'.format(args.datapath,model_name))
    
    for j in range(num_ensembles):
        m = BiLSTM(labels=labels,
                    word_vocab=word_vocab,
                    word_embeddings=embeddings,
                        optimizer=args.optimizer,
                        embedding_size=200, 
                        char_embedding_size=32,
                        lstm_dim=200,
                        num_cores=args.num_cores,
                        embedding_factor=args.embedding_factor,
                        learning_rate=args.learning_rate,
                        decay_rate=args.decay_rate,
                        dropout_keep=args.keep_prob)
        
        training_samples = random.sample(trainset,len(trainset)/2)
        
        cut = int(0.8 * len(training_samples))
        X_train, y_train = zip(*training_samples[:cut]) 
        X_dev, y_dev = zip(*training_samples[cut:]) 
        
        print "Training on {}, tuning on {}".format(len(X_train),len(X_dev))
        
        m.fit(X_train, y_train, X_dev, y_dev,
                num_iterations=args.num_iterations,
                num_it_per_ckpt=args.num_it_per_ckpt,
                batch_size=args.batch_size,
                seed=j, fb2=True)
        
        save_path = '{}/{}/model_{}'.format(args.datapath,model_name,j)
        m.save(save_path)
        print "Saved model {} to {}".format(j,save_path)

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

def load_dataset(fname, shuffle=False):
    dataset = []
    with open(fname,'r') as f:
        dataset = [ x.split('\n') for x in f.read().split('\n\n') if x and not x.startswith('#') ]
    
    vocab = []
    output = []
    for x in dataset:
        tokens, labels = zip(*[ z.split(' ')[:2] for z in x if z ])

        for t in tokens:
            t = t.lower()
            if t not in vocab:
                vocab.append(t)
        
        output.append((tokens, labels))
    
    return output, vocab

if __name__ == '__main__':
    main(parser.parse_args())
