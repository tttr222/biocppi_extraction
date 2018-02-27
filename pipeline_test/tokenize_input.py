#!/usr/bin/env python
import sys, os, random, pickle, fileinput
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import sent_tokenize

def main():
    lines = []
    for line in fileinput.input():
        lines.append(line.strip().split('\t'))
    
    for pmid, document in lines:
        spans = [ ts for ts in token_spans(document,wordpunct_tokenize) ]
        sent_offsets = [ (s[1],s[2]) for s in token_spans(document,sent_tokenize) ]
        
        aligned_sentences = []
        for sent0, sent1 in sent_offsets:
            sentence = []
            for span in spans:
                (text, start0, end1) = span
                if start0 >= sent0 and end1 <= sent1:
                    sentence.append(text)
            aligned_sentences.append(sentence)
        
        print '###' + pmid
        print ''
        for sentence in aligned_sentences:
            for word in sentence:
                print word
            
            print ''

def token_spans(txt,tokenizer):
    tokens = tokenizer(txt)
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        yield token, offset, offset+len(token)
        offset += len(token)

if __name__ == '__main__':
    main()
