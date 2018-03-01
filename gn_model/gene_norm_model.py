import os, random, pickle, re
from ncbi_lookup import gene_normalization

class GeneNormalizer(object):
    def __init__(self, gene_cache_file, pmid_cache_file, pubtator_cache_file):
        self.gene_cache_file = gene_cache_file
        self.pmid_cache_file = pmid_cache_file
        self.pubtator_cache_file = pubtator_cache_file
        
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
            
            y_pred.append(candidate)
        
        return y_pred
