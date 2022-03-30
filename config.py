#!/usr/bin/env python3
import string
config = {
	'all_letters': string.printable ,
	'n_letters' : len(string.printable) ,
	'n_categories' : 2 ,
	'all_categories' : ['neg','pos'] , # 0 , 1 
	'category_lines' : {} ,
	'n_iters' : 20000 ,
	'print_every' : 1 ,
	'plot_every' : 1000,
	'seq_count' : 0,
    'hidden_sz' : 212,
    'learning_rate': 0.0055,
    'random_seed' : 'False',
    'ml_bucket': './buckets/ml_bucket.txt',
	'import_bucket': './buckets/import_bucket.txt',
	'tempData': './data/tempData.txt'
}