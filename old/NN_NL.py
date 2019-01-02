from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import copy
import time
import math
from lib import support
from lib import tensorlib

import argparse
parser = argparse.ArgumentParser(description='PyTorch LSTM')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    print('Using CUDA Mode.')
    args.device = torch.device('cuda')
else:
    print('Using CPU Mode.')
    args.device = torch.device('cpu')

tensorlib.torch_device = args.device

class LSTM(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTMCell(input_size,hidden_size)
        self.mid = nn.Linear(hidden_size,hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size,hidden_size)
        self.final = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

        self.softmax = nn.LogSoftmax(dim=1)

        self.hidden_size = hidden_size
        self.criterion = nn.NLLLoss().to(device=tensorlib.torch_device)
        self.learning_rate = 0.0005
        self.bestLoss = math.inf
        self.bestNN = None
    def forward(self, input, hidden, state):

        hidden1 = hidden[1:].to(device=tensorlib.torch_device)
        hidden2 = hidden[:1].to(device=tensorlib.torch_device)
        state1 = state[1:].to(device=tensorlib.torch_device)
        state2 = state[:1].to(device=tensorlib.torch_device)

        hidden1, state1 = self.lstm(input,(hidden1,state1))
        mid = self.mid(state1)
        hidden2,state2 = self.lstm2(mid,(hidden2,state2))
        final = self.final(state2)
        output = self.softmax(self.dropout(final))
        return output,torch.cat((hidden1,hidden2)),torch.cat((state1,state2))
    def initHidden(self):
        # First number is the number of layers
        return torch.Tensor((),device=tensorlib.torch_device).new_zeros(2, self.hidden_size)
    def train(self,input_line_tensor, target_line_tensor):
        target_line_tensor.unsqueeze_(-1)
        hidden = self.initHidden()
        state = self.initHidden()

        self.zero_grad()

        for i in range(input_line_tensor.size(0)):
            support.printProgress(i,i_max=input_line_tensor.size(0))
            output, hidden, state = self(input_line_tensor[i], hidden, state)
            loss = self.criterion(output, target_line_tensor[i])
            if (i < input_line_tensor.size(0) - 1):
                loss.backward(retain_graph=True)
            else:
                loss.backward()

            for p in self.parameters():
                p.data.add_(-self.learning_rate, p.grad.data)

            lossFactor = loss.item() / input_line_tensor.size(0)
            if (lossFactor < self.bestLoss):
                self.bestNN = self.state_dict()
                self.bestLoss = lossFactor

            yield i, output, lossFactor

    def sample(self,start_letter='A'):
        with torch.no_grad():  # no need to track history in sampling
            input = tensorlib.inputTensor(start_letter)
            hidden = self.initHidden()
            state = self.initHidden()

            output_pickup = start_letter

            for i in range(max_length):
                output, hidden,state = self(input[0], hidden,state)
                topv, topi = output.topk(1)
                topi = topi[0][0]
                if topi == support.n_letters - 1:
                    break
                else:
                    letter = support.all_letters[topi]
                    output_pickup += letter
                input = tensorlib.inputTensor(letter)

            return output_pickup
    def samples(self, start_letters='ABC'):
        for start_letter in start_letters:
            woah = self.sample(start_letter)
            print(woah)
            yield woah


pickups = support.readLines('./fun/HPSS.txt', sentenceDelimiter='CHAPTER')
'''
print('Length of data: ' + str(len(pickups)))
pickups.extend(support.readLines('./fun/HPCS.txt', sentenceDelimiter='CHAPTER'))
print('Length of data: ' + str(len(pickups)))
pickups.extend(support.readLines('./fun/HPPA.txt', sentenceDelimiter='CHAPTER'))
print('Length of data: ' + str(len(pickups)))
pickups.extend(support.readLines('./fun/HPGF.txt', sentenceDelimiter='CHAPTER'))
print('Length of data: ' + str(len(pickups)))
pickups.extend(support.readLines('./fun/HPOP.txt', sentenceDelimiter='CHAPTER'))
print('Length of data: ' + str(len(pickups)))
pickups.extend(support.readLines('./fun/HPHB.txt', sentenceDelimiter='CHAPTER'))
print('Length of data: ' + str(len(pickups)))
pickups.extend(support.readLines('./fun/HPDH.txt', sentenceDelimiter='CHAPTER'))
print('Length of data: ' + str(len(pickups)))
'''
#pickups.extend(readLines('./fun/BMovie.txt', sentenceDelimiter='.'))
#print('Length of data: ' + str(len(pickups)))

lstm = LSTM(support.n_letters,512,support.n_letters)
lstm.to(device=tensorlib.torch_device)


n_iters = 1000
print_every = 100
plot_every = 20
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

print('Testing')
for iter in range(1, n_iters + 1):

    for i, output, loss in lstm.train(*tensorlib.randomTrainingExample(pickups)):
        total_loss += loss
        if i*iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (support.timeSince(start), i*iter, i*iter / n_iters * 100, loss))

        if i*iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

plt.figure()
plt.scatter(range(len(all_losses)), all_losses, s=1)
plt.savefig('train.png')

with open('output.txt','w') as file:
    try:
        for sample in lstm.samples(('ABCDEFGHIJKLMNOPQRSTUVWXYZ')):
            file.write(sample)
        lstm.load_state_dict(lstm.bestNN)
        print('==================== Best Network State ====================')
        file.write('==================== Best Network State ====================')
        for sample in lstm.samples(('ABCDEFGHIJKLMNOPQRSTUVWXYZ')):
            file.write(sample)
    finally:
        file.close()
