#!/usr/bin/env python
import sys, os, random, pickle, json
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Train and evaluate BiLSTM on a given dataset')
parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')

def main(args):
    df = pd.read_csv('pipeline_output.txt',delimiter='\t',
        names=('pmid','gene_a','gene_b','score_1','score_2'),
        dtype={ 'pmid': object, 'gene_a': object, 'gene_b': object })
    
    with open(args.files[0],'r') as f:
        obj = json.load(f)
    
    for i in range(len(obj['documents'])):
        obj['documents'][i]['relations'] = []
        
        df_doc = df.loc[df['pmid'] == obj['documents'][i]['id']]
        print df_doc
        relid = 0
        
        best_rel = []
        best_score = 0
        for j, row in df_doc.iterrows():
            score = max(row['score_1'],row['score_2'])
            
            infons = { 'Gene1': row['gene_a'], 'Gene2': row['gene_b'], 'relation': 'PPIm', 'confidence': round(score,2)}
            relobj = { 'nodes': [], 'infons': infons, 'id': 'R{}'.format(relid) }
                
            if score > 0.5:
                obj['documents'][i]['relations'].append(relobj)
                relid += 1
            
            if score > best_score:
                best_rel = relobj
                best_score = score
        
        if len(obj['documents'][i]['relations']) == 0:
            obj['documents'][i]['relations'].append(best_rel)
        
        print obj['documents'][i]['id'], obj['documents'][i]['relations']
    
    with open('PMtask_results.json','w') as f:
        json.dump(obj,f)
	
if __name__ == '__main__':
    main(parser.parse_args())
