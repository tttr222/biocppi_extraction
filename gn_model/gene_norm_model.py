import os, random, pickle, re
#from fuzzywuzzy import process as fuzzproc
from fuzzyset import FuzzySet
from ncbi_lookup import gene_normalization

class GeneNormalizer(object):
    def __init__(self, gene_cache_file, pmid_cache_file, pubtator_cache_file):
        self.memory = []
        self.mapping = {}
        self.gene_cache_file = gene_cache_file
        self.pmid_cache_file = pmid_cache_file
        self.pubtator_cache_file = pubtator_cache_file
        
    def fit_old(self, X, y, X_aux, y_aux):
        self.memory = FuzzySet(X + X_aux)
        self.mapping = dict([ (k,v) for k, v in zip(X + X_aux,y + y_aux) ])
        
    def fit(self, X_pair, y, X_aux, y_aux):
        pmids, X = zip(*X_pair)
        self.mapping = {}
        
        candidates_map = {}
        frequency_map = {}
        for k, v in zip(X_aux,y_aux):
            if k not in candidates_map:
                candidates_map[k] = []
            
            candidates_map[k].append(v)
            
            if v not in frequency_map:
                frequency_map[v] = 1
            else:
                frequency_map[v] += 1
        
        for k, v in candidates_map.iteritems():
            if len(v) == 1:
                self.mapping[k] = v[0]
        
        for k, v in zip(X,y):
            self.mapping[k] = v
        
        self.memory = FuzzySet(self.mapping.keys())

    def predict(self,X_pair, y_test = None):
        pmids, X = zip(*X_pair)
        
        y_pred = []
        for i, x in enumerate(X):
            candidate = None
            
            if candidate == None:
                candidate = gene_normalization(x,pmids[i], 
                    self.gene_cache_file, 
                    self.pmid_cache_file,
                    self.pubtator_cache_file)
                
            #if candidate == None:
            #    response = self.memory.get(x)
            #    if response is not None:
            #        score, match_string = response[0]
            #        
            #        if score > 0.99:
            #            candidate = self.mapping[match_string]
            
            y_pred.append(candidate)
        
        return y_pred
