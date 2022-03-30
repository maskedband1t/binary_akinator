#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## load libraries and set-up:
from cgi import test
import time
from pathlib import Path
from posixpath import basename
from telnetlib import EL
import pandas as pd
pd.set_option('display.max_colwidth', None)
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import os
import pickle #optional - for saving outputs
import re
from tqdm import tqdm # used for progress bars (optional)
import time
from matplotlib import pyplot as plt
import sys
import csv
import nmslib


import re
import subprocess
import hashlib

from ftfy import fix_text

import click

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

def vectorize_and_query(_library : list[str], _hashnames: list[str], _ingested_str : str, _kNN: int, _benchmark: int):

  ###FIRST TIME RUN - takes about 5 minutes... used to build the matching table
  from sklearn.feature_extraction.text import TfidfVectorizer
  import time
  t1 = time.time() # used for timing - can delete
 

  #Building the TFIDF off the clean dataset - takes about 5 min
  vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
  #tf_idf_matrix = vectorizer.fit_transform(org_names)
  tf_idf_matrix = vectorizer.fit_transform(_library) 
  t = time.time()-t1
  print("Time:", t) # used for timing - can delete
  print(tf_idf_matrix.shape)




  
  t1 = time.time()
  ##### Create a list of messy items to match here:
  #df_CF = pd.read_csv(input2_csv) # file containing messy supplier names to match against
  #messy_names = list(df_CF[input2_column].unique()) #unique list of names

  #Creation of vectors for the messy names

  # #FOR LOADING ONLY - only required if items have been saved previously
  # vectorizer = pickle.load(open("Data/vectorizer.pkl","rb"))
  # tf_idf_matrix = pickle.load(open("Data/Comp_tfidf.pkl","rb"))
  # org_names = pickle.load(open("Data/Comp_names.pkl","rb"))

  #messy_tf_idf_matrix = vectorizer.transform(messy_names)
  _ingested_list = [_ingested_str]
  messy_tf_idf_matrix = vectorizer.transform(_ingested_list)

  


  # create a random matrix to index
  data_matrix = tf_idf_matrix#[0:1000000]

  # Set index parameters
  # These are the most important ones
  M = 80
  efC = 1000

  num_threads = 4 # adjust for the number of threads
  # Intitialize the library, specify the space, the type of the vector and add data points 
  index = nmslib.init(method='simple_invindx', space='negdotprod_sparse_fast', data_type=nmslib.DataType.SPARSE_VECTOR) 

  index.addDataPointBatch(data_matrix)
  # Create an index
  start = time.time()
  index.createIndex() 
  end = time.time() 
  print('Indexing time = %f' % (end-start))
  # Number of neighbors 
  num_threads = 4
  K=_kNN
  query_matrix = messy_tf_idf_matrix
  start = time.time() 
  query_qty = query_matrix.shape[0]
  nbrs = index.knnQueryBatch(query_matrix, k = K, num_threads = num_threads)
  end = time.time() 
  print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' % 
        (end-start, float(end-start)/query_qty, num_threads*float(end-start)/query_qty))

  

  mts =[]
  '''
  print(len(nbrs))
  print(len(nbrs[0][1]))
  print(nbrs[0])
  print(nbrs[0][1])
  print(nbrs[0][0]) 
  '''
  # prints the N closest neighbors to this file.
  if(_kNN > 1):
    print('\n')
    print('Printing kNN -> (%s) matches: ' %(_kNN))
    for i in range(_kNN):
      print(_hashnames[nbrs[0][0][i]])
      print(nbrs[0][1][i])

  print('\n')
  print('Printing matches passing confidence benchmark percentage (default is 60): %s ' %(_benchmark))
  for i in range(_kNN): #TODO: make the range the number of possible files.
    try:
      #_matched = _hashnames[nbrs[i][0][0]] ## need this to be the hash
      _matched = _hashnames[nbrs[0][0][i]] ## need this to be the hash
      _conf = nbrs[0][1][i]
      
    except:
      _matched = "no match found"
      _conf = None
    
    # report only if adjusted confidence above benchmark
    if _conf != None:
      if(abs(_conf)*100 >= _benchmark):
        print("closest file is: %s" %(_matched))
        print("with confidence: %s" %(_conf)) 
  
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
def bounce_and_ingest(bounce, filename):
  
  filename_base = strip_filename(filename,'/',True)
  file_hash = hashlib.md5(open(filename,'rb').read()).hexdigest()
  hashname = '%s-%s.pkl' %(file_hash, filename_base)

  b_bounce_time = time.time()

  if not(pickle_fs_contains(file_hash)) or bounce:
    try:
      p = subprocess.check_output(['python3', 'Binary_Bouncer.py', filename])
    except subprocess.CalledProcessError as e:
      print(e.output)
    
    '''
      next 4 lines are experimental to avoid bucket reads ahead of the bounce termination.
      delete if causing issues.
    '''  
    bounce_poll = p.poll()
    while(bounce_poll is None):
      time.sleep(3)
      bounce_poll = p.poll()

    a_bounce_time = time.time()
    print("Time to bounce file: %s sec." %(a_bounce_time-b_bounce_time))
    read_and_ingest(file_hash, filename_base)
 
    print("returning newly ingested file_hash:%s" %(file_hash))
  
  else:
    _joined = read_pickle(hashname)
    if not(_joined):
       print("Nothing is in this pickle :/")
    print("returning previously ingested file_hash:%s" %(file_hash))

'''
  returns True if filesystem contains a file with :param <file_hash> in the pickled_files directory  
'''
def pickle_fs_contains(file_hash : str) -> bool:
  for filename in os.listdir(pickle_directory):
    if(file_hash in filename):
      return True
  return False

'''
  reads pickle and returns object, in our case we are storing strings
'''
def read_pickle(filename : str) -> str:
  full_path = '%s/%s/%s'%(script_path,pickle_directory,filename)
  with open(full_path, 'rb') as p:
        unpickled = pickle.load(p) # append to list of strings
        print("read pickle @ %s" %(full_path))
        return unpickled

'''
  params: _library: List of library binary strings
          _hashnames: List of corresponding hashes
          _ingested: input binary string
'''
def compare_these(_library: list[str], _hashnames: list[str], _ingested: str, _kNN: int, _benchmark: int):
  print("entering vectorize and query")
  vectorize_and_query(_library, _hashnames,_ingested,_kNN, _benchmark)

'''
  reads the buckets from bounce and pickles the combined contents
'''
def read_and_ingest(file_hash: str, filename_base: str):
  bouncer_read_df= pd.read_csv("buckets/ml_bucket.txt",header=0,delimiter='\r',quoting=csv.QUOTE_NONE)
  print('read bucket(s)') # TODO: read from all buckets in bucket folder, combine
  input_bounced_in = bouncer_read_df.values.tolist()
    
  #TODO: remove late-game
  print(len(input_bounced_in))
  print(input_bounced_in)

  _joined = ''
  for i in range(len(input_bounced_in)):
    _joined += ''.join(input_bounced_in[i]) # now we have one single string representing file
    

  new_pickle = 'pickled_files/%s-%s.pkl' %(file_hash, filename_base)  
  with open(new_pickle, 'wb') as ingested_pickle: # save combined string into output/ represents file
    print("creating pickle. check pickled_files folder...")
    pickle.dump(_joined, ingested_pickle)

@click.command()
@click.option('--bounce', is_flag=True, help="set this flag if a previously bounced file needs to get re-bounced")
@click.option('--n',default=1,type=int , help="sets the number of nearest neighbors to return, kNN, default 1")
@click.option('--benchmark',default=60,type=int , help="return all kNN with a confidence over this benchmark, default: 60")
@click.argument('incoming_binary')
def main(bounce, n, benchmark, incoming_binary):
  filename = incoming_binary #sys.argv[1]
  filename_base = strip_filename(filename,'/',True)
  file_hash = hashlib.md5(open(filename,'rb').read()).hexdigest()
  
  # kNN correction. should be positive integer
  if(n < 0):
    n = 1

  # benchmark norm. correction. should be between (0,100]
  if(benchmark < 0):
    benchmark = 60
  benchmark = benchmark % 100

  try:
    bounce_and_ingest(bounce, filename)
  except:
    print("ingestion failed")
    # try to create the pickled file from the incoming binary once again, must have failed during bounce_and_ingest
    read_and_ingest(file_hash,filename_base)

  set_of_strings = []
  set_of_hashnames = []
  for filename in os.listdir(pickle_directory):
    current_file_prefix = os.path.basename(filename)
    current_file_prefix = strip_filename(current_file_prefix,'.',False)
    
    f = os.path.join(pickle_directory, filename)

    '''
     As long as file exists, doesnt start with '.' and isn't the recently-ingested file
    '''
    if (not(filename.startswith('.'))) and os.path.isfile(f):
      if (file_hash not in filename):
        print("loading %s" %(filename))
        try:
          lib_string = read_pickle(filename)
          set_of_strings.append(lib_string)
          set_of_hashnames.append(current_file_prefix)
        except:
          print('failed to load pickle: %s' %(filename))
      else:
        imported_lib_string = read_pickle(filename)
  print("done loading in library, created sets")
  '''
  pass in this list of strings and the input string to comparison functions
  '''
  compare_these(set_of_strings, set_of_hashnames, imported_lib_string, n, benchmark)
  print("\ncompared file with hash %s to rest of files in set" %(file_hash))
  print("run to validate pickle: python3 /helper_files/pickle_validator.py %s" %(file_hash))
if __name__ == '__main__':
  main()