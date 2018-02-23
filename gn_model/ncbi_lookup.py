#!/usr/bin/env python
import sys, os, random, pickle, re, time
import urllib2, urllib
import pandas as pd

api_endpoint = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'


pubtator_api_endpoint = 'https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/Gene/{}/PubTator/'

def gene_normalization(span, pmid, gene_cache_file, pmid_cache_file, pubtator_cache_file):
    if len(span) <= 1:
        return None
    
    span_candidates = gene_name_lookup(span, gene_cache_file)
    pmid_candidates = gene_pmid_lookup(pmid, pmid_cache_file)
    pubtator_candidates = pubtator_pmid_lookup(pmid, pubtator_cache_file)
    
    for z in span_candidates:
        if z in pmid_candidates:
            return z
    
    for z in span_candidates:
        if z in pubtator_candidates:
            return z
    
    return None

cache_gene_map = None
cache_gene_count = 0
def gene_name_lookup(gene_name, cache_file = 'gene_map.cache'):
    global cache_gene_map
    global cache_gene_count
    if cache_gene_map is None:
        if os.path.exists(cache_file):
            with open(cache_file,'r') as f:
                cache_gene_map = pickle.load(f)
        else:
            cache_gene_map = {}
    
    if gene_name in cache_gene_map:
        return cache_gene_map[gene_name]
    
    api_query_gene = { 'db': 'gene', 
                'term': '{}'.format(gene_name),
                'sort': 'relevance',
                 'retmax': 3000 }
    query = urllib.urlencode(api_query_gene)
    xml = urllib2.urlopen('{}?{}'.format(api_endpoint,query)).read()
    match = re.findall(r'<Id>([0-9]+)</Id>',xml)
    #print 'Querying NCBI.. GENE={}: {} results.'.format(gene_name,len(match))
    
    cache_gene_map[gene_name] = match
    time.sleep(1)
    
    cache_gene_count += 1
    
    if cache_gene_count % 200 == 0:
        with open(cache_file,'w') as f:
            pickle.dump(cache_gene_map,f)
    
    return match

cache_pmid_map = None
cache_pmid_count = 0
def gene_pmid_lookup(pmid, cache_file = 'pmid_map.cache'):
    global cache_pmid_map
    global cache_pmid_count
    if cache_pmid_map is None:
        if os.path.exists(cache_file):
            with open(cache_file,'r') as f:
                cache_pmid_map = pickle.load(f)
        else:
            cache_pmid_map = {}
    
    if pmid in cache_pmid_map:
        return cache_pmid_map[pmid]
            
    api_query_pmid = { 'db': 'gene', 
                        'term': '{}[PMID]'.format(pmid),
                        'sort': 'relevance',
                        'retmax': 100
                     }
    query = urllib.urlencode(api_query_pmid)
    xml = urllib2.urlopen('{}?{}'.format(api_endpoint,query)).read()
    match = re.findall(r'<Id>([0-9]+)</Id>',xml)
    #print 'Querying NCBI.. PMID={}: {} results.'.format(pmid,len(match))
    
    cache_pmid_map[pmid] = match
    time.sleep(1)
    cache_pmid_count += 1
    
    if cache_pmid_count % 200 == 0:
        with open(cache_file,'w') as f:
            pickle.dump(cache_pmid_map,f)
        
    return match

cache_pubtator_map = None
cache_pubtator_count = 0
def pubtator_pmid_lookup(pmid, cache_file = 'pubtator_pmid_map.cache'):
    global cache_pubtator_map
    global cache_pubtator_count
    if cache_pubtator_map is None:
        if os.path.exists(cache_file):
            with open(cache_file,'r') as f:
                cache_pubtator_map = pickle.load(f)
        else:
            cache_pubtator_map = {}
    
    if pmid in cache_pubtator_map:
        return cache_pubtator_map[pmid]
            
    response = urllib2.urlopen('{}?{}'.format(pubtator_api_endpoint.format(pmid),{})).read()
    match = set(re.findall(r'Gene\t([0-9]+)\n',response))

    cache_pubtator_map[pmid] = list(match)
    time.sleep(0.1)
    cache_pubtator_count += 1
    
    if cache_pubtator_count % 200 == 0:
        with open(cache_file,'w') as f:
            pickle.dump(cache_pubtator_map,f)
        
    return list(match)

if __name__ == '__main__':
    main()
