#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import hashlib
import os
import re
import subprocess
import sys
from ftfy import fix_text
import pandas as pd
pd.set_option('display.max_colwidth', -1)
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import time
import pathlib
from matplotlib import pyplot as plt

import numpy as np
from scipy.sparse import csr_matrix, rand

import nmslib
from pathlib import Path

'''
	Used for matrix conversion, ngrams are the units on which we will build matrices using tf-idf and nmslib
		TODO: may need to increase stripped sequences if unimportant terms are biased
'''
'''
def ngrams(string, n=3):
	string = str(string)
	string = fix_text(string) # fix text
	string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
	string = string.lower()
	chars_to_remove = [")","(",".","|","[","]","{","}","'"]
	rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
	string = re.sub(rx, '', string)
	#string = string.replace('&', 'and')
	string = string.replace(',', ' ')
	string = string.replace('-', ' ')
	string = string.title() # normalise case - capital at start of each word
	string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
	string = ' '+ string +' ' # pad names for ngrams...
	string = re.sub(r'[,-./]|\sBD',r'', string)
	ngrams = zip(*[string[i:] for i in range(n)])
	return [''.join(ngram) for ngram in ngrams]
'''

#transforms company names with assumptions taken from: http://www.legislation.gov.uk/uksi/2015/17/regulation/2/made
def ngrams(string, n=3):
    """Takes an input string, cleans it and converts to ngrams. 
    This script is focussed on cleaning UK company names but can be made generic by removing lines below"""
    string = str(string)
    string = string.lower() # lower case
    string = fix_text(string) # fix text
    string = string.split('t/a')[0] # split on 'trading as' and return first name only
    string = string.split('trading as')[0] # split on 'trading as' and return first name only
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    chars_to_remove = [")","(",".","|","[","]","{","}","'","-"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']' #remove punc, brackets etc...
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace('limited', 'ltd')
    string = string.replace('public limited company', 'plc')
    string = string.replace('united kingdom', 'uk')
    string = string.replace('community interest company', 'cic')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
    string = ' '+ string +' ' # pad names for ngrams...
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


'''
  global config here
'''
pickle_directory = 'pickled_files'
script_path = Path(__file__).parent

def vectorize_and_query(_library_str : str, _ingested_str : str):
	words =  pd.read_csv('%s/test_docs/Gov Orgs ONS copy.csv' %(script_path)) # for testing
	input1_column = 'Institutions'
	doc_words = list(words[input1_column].unique())
	print(type(doc_words))
	_combined = [''.join(doc_words)]

	print('starting tf-idf vectorization')
	vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams) # TODO: review analyzer param
	tf_idf_matrix = vectorizer.fit_transform(_combined) # fit_transform on training data, or set

	words_2 =  pd.read_csv('%s/test_docs/messy org names copy.csv' %(script_path)) # for testing
	input2_column = 'buyer'
	doc_words_2 = list(words_2[input2_column].unique())
	_combined_2 = [''.join(doc_words_2)]
	tf_idf_matrix_2 = vectorizer.transform(_combined_2) # transform for test data, or new binary
	print(len(doc_words_2))
	print('just finished data vectorization')
	print(tf_idf_matrix.shape)
	'''
	Using NMSLIB for vector matching: https://benfred.github.io/nmslib/api.html#
	'''
	data_matrix = tf_idf_matrix

	""" # Set index parameters for HNSW
		M: -> the parameter M defines the maximum number of neighbors in the zero and above-zero layers.
		ef: -> the size of the dynamic list for the nearest neighbors (used during the search). 
			Higher ef leads to more accurate but slower search. ef cannot be set lower than the number of queried nearest neighbors k.
	 		The value ef of can be anything between k and the size of the dataset. """
	M = 80
	EF = 1000 # could be increased if datasets increase,. current low ball

	thread_count = 4 # can be changed if important

	""" nmslib.init acts act the main entry point into NMS lib. This function should be called first before calling any other method.
	# 	Parameters:	
	#
		#	space (str optional) – The metric space to create for this index
		#	method (str optional) – The index method to use
		#	data_type (nmslib.DataType optional) – The type of data to index (dense/sparse/string vectors)
			dist_type (nmslib.DistType optional) – The type of index to create (float/double/int)

		Return type: A new NMSLIB Index. 
	"""
	index = nmslib.init(method='simple_invindx', space='negdotprod_sparse_fast', data_type=nmslib.DataType.SPARSE_VECTOR)
	index.addDataPointBatch(data_matrix) # Adds multiple datapoints to the index
	
	'''
		Create index, make available for query
			* Keep index creation time in case needed for comparisons *
	'''
	
	start = time.time()
	index.createIndex()
	end = time.time()
	print('Time taken for indexing: %f' % (end-start))
	
	''' 	Perform queries on second matrix
		K: number of neighbors to return (for now, 1)
		knnQueryBatch() ->
			* Performs multiple queries on the index, distributing the work over a thread pool
			 :param input: A list of queries to query for :type input: list 
			 :param k: The number of neighbours to return :type k: int optional 
			 :param num_threads: The number of threads to use :type num_threads: int optional
			
			Returns: A list of tuples of (ids, distances) -> nbrs
		* Keep index query time in case needed for comparisons *
	'''
	K=1
	query_matrix = tf_idf_matrix_2
	query_count = query_matrix.shape[0]
	data_count = data_matrix.shape[0]
	
	print("# of queries %d, # of data points %d"  % (query_count, data_matrix.shape[0]) )
	
	start = time.time()
	nbrs = index.knnQueryBatch(query_matrix, k = K, num_threads = thread_count)
	end = time.time()

	print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' % 
      		(end-start, float(end-start)/query_count, thread_count*float(end-start)/query_count))

	'''
		Next step is to poll matches
	'''
	mts =[]
	print(len(nbrs))
	for i in range(len(nbrs)):
		origional_nm = doc_words_2[i]
		try:
			matched_nm   = doc_words[nbrs[i][0][0]]
			conf         = nbrs[i][1][0]
			if(conf < -0.6):
				print(origional_nm)
				print(matched_nm)
				print(conf)
		except:
			matched_nm   = "no match found"
			conf         = None
		mts.append([origional_nm,matched_nm,conf])

	mts = pd.DataFrame(mts,columns=['origional_name','matched_name','conf'])
	results = words_2.merge(mts,left_on='buyer',right_on='origional_name')

	results.conf.hist()
	plt.show()
	#print(results.conf)
	#hist = results.conf.hist() # TODO: need to figure out how else to interpret results
	#plt.show()
	#print('All 3-grams in "Department":')
	#print(ngrams('Department'))

'''
  strips filename to first reverse occurance of sep, 
  params: 
        filename
        sep
        reverse (bool) -> set to False if forward order towards last occurance of sep
  ex. f/hello.txt 
                reverse=True -> hello.txt
                reverse=False -> f      
'''
def strip_filename(filename: str, sep: str, reverse: bool) -> str:
  if(reverse): filename_base = filename.split(sep).pop()
  else: filename_base = (filename.rpartition(sep))[0]
  return filename_base
'''
  reads input binary, creates joined string rep, ingests it via pickle to ./pickled_files
  returns: dict of filename (base representation) and joined string made from input binary
'''
def read_and_ingest() -> dict:
  filename = sys.argv[1]
  filename_base = strip_filename(filename,'/',True)
  try:
    p = subprocess.check_output(['python3', 'Binary_Bouncer.py', filename])
  except subprocess.CalledProcessError as e:
     print(e.output)
  file_hash = hashlib.md5(open(filename,'rb').read()).hexdigest()

  bouncer_read_df= pd.read_csv("buckets/ml_bucket.txt",header=0,delimiter='\r')
  
  input_bounced_in = bouncer_read_df.values.tolist()
  _joined = ''
  for i in range(len(input_bounced_in)):
    _joined += ''.join(input_bounced_in[i]) # now we have one single string representing file
  
  with open('pickled_files/%s-%s.pkl' %(file_hash, filename_base), 'wb') as ingested_pickle: # save combined string into output/ represents file
    pickle.dump(_joined, ingested_pickle)
  
  _input_dict = {'filename': filename_base ,
                 'file_hash': file_hash  ,
                 'joined_str': _joined}
  return _input_dict

'''
  params: _library: List of library binary strings
          _ingested: input binary string
'''
def compare_these(_library: list[str], _ingested: str):
  #print(_library)
  for i in range(len(_library)):
    #vectorize_and_query(_library[i], _ingested)
    break
def main():
  _ingested_dict = read_and_ingest()
  set_of_strings = []

  for filename in os.listdir(pickle_directory):
    current_file_prefix = os.path.basename(filename)
    current_file_prefix = strip_filename(current_file_prefix,'.',False)
    
    f = os.path.join(pickle_directory, filename)

    '''
     As long as file exists, doesnt start with '.' and isn't the recently-ingested file
    '''
    if (not(filename.startswith('.')) and os.path.isfile(f)
    and (_ingested_dict['file_hash'] not in filename)):

      with open('%s/%s/%s'%(script_path,pickle_directory,filename), 'rb') as p:
        print("loading %s" %(filename))
        lib_string = pickle.load(p) # append to list of strings
        set_of_strings.append(lib_string)

  print("done loading in library, created sets")
  '''
  pass in this list of strings and the input string to comparison functions
  '''
  compare_these(set_of_strings, _ingested_dict['joined_str'])
  
if __name__ == '__main__':
  main()





