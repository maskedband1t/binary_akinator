#!/usr/bin/env python3
from __future__ import unicode_literals, print_function, division

import pandas as pd
import torch
import torch.nn as nn
import random, time, math
from io import open
import pdb
import config

preset = config.config


#### helper functions  ###############################################################
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def letterToIndex(letter):
    return preset['all_letters'].find(letter)
def letterToTensor(letter):
    tensor = torch.zeros(1, preset.n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return preset['all_categories'][category_i], category_i
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]
def randomTrainingExample():
    category = randomChoice(preset['all_categories'])
    line = randomChoice(preset['category_lines'][category])
    category_tensor = torch.tensor([preset['all_categories'].index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, preset['n_letters'])
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
def randomValidationExample(validation_List):
    line = randomChoice(validation_List)
    line_tensor = lineToTensor(line)
    return line, line_tensor
 
#returns line and line_tensor depending on seq_count // possibly unnecessary
def sequentialValidation(validation_List, answers):
	n_iters = len(validation_List)
	n_answers = len(answers)
	if(n_answers == n_iters):
		if(preset['seq_count'] < n_iters):
			line = validation_List[preset['seq_count']]
			line_tensor = lineToTensor(line)
			answer = answers[preset['seq_count']]
			#increment
			preset['seq_count'] += 1
			return line, line_tensor, answer
	else:
		print('Your validation file and answers file are misaligned\n')

def stringHandler(binary_in):
    file_path = str(binary_in)
    book = []
    with open(file_path,'r') as txt:
            if txt:
                text = txt.read()
                strings = [item.replace(" ", "") for item in text.splitlines()]
                book = strings
            txt.close()
    return book
##################################################################################################

class LSTMClassifier(nn.Module):
    def __init__(self,n_letters,hidden_sz,n_layers,n_categories=2) -> None:
        super().__init__()
        self.network = nn.LSTM(n_letters, hidden_sz, n_layers,batch_first=True) # batch_sz X 1 X hidden_sz
        self.linear = nn.Linear(hidden_sz,n_categories) # hidden_sz X n_categories, gives us unnormalized probabilit
        # we want something that is batch_sz X n_categories --> HAS TO BE PROBABILITY
    def forward(self, input):
        #torch.manual_seed(60)
        hn = torch.randn(1,batch_size,preset['hidden_sz'])
        cn = torch.randn(1,batch_size,preset['hidden_sz'])
        
        for i in range(input.size()[0]):
            #pdb.set_trace()
            letter = torch.unsqueeze(input[i],dim=0)
            output, (hn,cn) = self.network(letter, (hn,cn))
        output = output.squeeze(dim=1)
        logit = self.linear(output)
        return logit       # logit is standard name for NN before passed through softmax



n_layers = 1

criterion = nn.CrossEntropyLoss()
batch_size = 1


def train(category_tensor, line_tensor):
    lstm.zero_grad()
      
    output = lstm(line_tensor)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in lstm.parameters():
        p.data.add_(p.grad.data, alpha=-preset['learning_rate'])

    return output, loss.item()

lstm = LSTMClassifier(preset['n_letters'],preset['hidden_sz'],n_layers)

def saveModel() -> LSTMClassifier:
    for p in lstm.parameters():
        print(p)
        break
    torch.save(lstm, './model/savedModel.pt')
    return lstm
def loadModel() -> LSTMClassifier :
    lstm = torch.load('./model/savedModel.pt')
    for p in lstm.parameters():
        print(p)
        break
    lstm.eval()
    return lstm
def executeModel_tune():
    preset['category_lines']['pos'] = stringHandler('./data/pos.txt')
    preset['category_lines']['neg'] = stringHandler('./data/neg.txt')
    start = time.time()
    current_loss = 0
    all_losses = []
    record_last_x = 2000
    n_correct = 0

    for iter in range(1, preset['n_iters'] + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

    # Print iter number, loss, name and guess

        if iter % preset['print_every'] == 0:
            guess, guess_i = categoryFromOutput(output)

            category_i = preset['all_categories'].index(category)

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
    #pdb.set_trace()
    log = open('./data/tuning_results','a')
    log.write('\n\nAccuracy over last ' + str(record_last_x) + ' is: ' + str(accuracy_last_x))
    log.write('\n lr: ' + str(preset['learning_rate']))
    log.write('\n hidden_sz: ' + str(preset['hidden_sz']))
    print('Accuracy over last ' + str(record_last_x) + ' is: ' + str(accuracy_last_x))
    print('\n lr: ' + str(preset['learning_rate']))
    print('\n hidden_sz: ' + str(preset['hidden_sz']))
    time.sleep(2)
     
#seeds = [7,35,23,77,99]
#for i in range(1):
#    preset['hidden_sz'] = 212
#    preset['learning_rate'] = 0.0055
#    for j in range(5):
#        random.seed(seeds[j])
#        torch.manual_seed(seeds[j])
        
#        lstm = LSTMClassifier(preset['n_letters'],preset['hidden_sz'],n_layers)
#        executeModel_tune()      

def evaluate(line_tensor, lstm):
    for i in range(line_tensor.size()[0]):
        output = lstm(line_tensor)
    for p in lstm.parameters():
        #print(p)
        break
    #pdb.set_trace()
    return output
def predict(input_line, n_predictions=1):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, preset['all_categories'][category_index]))
            predictions.append([value, preset['all_categories'][category_index]])

    return output