#!/usr/bin/env python3
from __future__ import division, print_function, unicode_literals
from LSTM_model import LSTMClassifier, train, preset, randomTrainingExample, categoryFromOutput, timeSince, stringHandler,saveModel


import csv
import glob
import math
import os
import pdb
import string
import subprocess
import time
from io import open

import click
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
import random
preset = config.config

def executeModel(_seed='False') -> LSTMClassifier:
    preset['category_lines']['pos'] = stringHandler('./data/pos.txt')
    preset['category_lines']['neg'] = stringHandler('./data/neg.txt')
    start = time.time()
    current_loss = 0
    all_losses = []
    record_last_x = 2000
    n_correct = 0

    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(preset['n_categories'], preset['n_categories'])
    n_confusion = preset['n_iters']

    if(_seed == 'True'):
        print('pyTorch and random seeded')
        random.seed(65)
        torch.manual_seed(65)
    for iter in range(1, preset['n_iters'] + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

    # Print iter number, loss, name and guess

        if iter % preset['print_every'] == 0:
            guess, guess_i = categoryFromOutput(output)

            category_i = preset['all_categories'].index(category)
            confusion[category_i][guess_i] += 1

            # uncontaminated as this is still before it learns on that data point
            if guess == category:
                correct = '✓'
                if(iter > (preset['n_iters'] + 1 - record_last_x) and iter < (preset['n_iters'] + 1)):
                    n_correct = n_correct + 1
            else:
                correct = '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / preset['n_iters'] * 100, timeSince(start), loss, line, guess, correct))
            
        # Add current loss avg to list of losses
        if iter % preset['plot_every'] == 0:
            all_losses.append(current_loss / preset['plot_every'])
            current_loss = 0

    accuracy_last_x = n_correct / record_last_x
    
    print('Accuracy over last ' + str(record_last_x) + ' is: ' + str(accuracy_last_x))

    # Set up plot        
    plt.figure()
    plt.plot(all_losses)
    plt.show()
    lstm = saveModel()

    return lstm