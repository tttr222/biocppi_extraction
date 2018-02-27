#!/usr/bin/env python
import sys, os, random, pickle, json
import argparse
import string

parser = argparse.ArgumentParser(description='Train and evaluate BiLSTM on a given dataset')
parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')

def main(args):
    docs = {}
    with open(args.files[0],'r') as f:
        obj = json.load(f)
        for d in obj['documents']:
            docs[d['id']] = d
    
    with open('pipeline_feed.txt','w') as f:
        for pmid, d in docs.iteritems():
            textconcat = []
        
            for p in d['passages']:
                textconcat.append(p['text'])
                
            textconcat = ' '.join(textconcat)
            print >> f, "{}\t{}".format(pmid,''.join([ c for c in textconcat if c in string.printable ]))
        
if __name__ == '__main__':
    main(parser.parse_args())
