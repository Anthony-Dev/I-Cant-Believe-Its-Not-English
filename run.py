#! /usr/bin/python3

import argparse
from lib import support
from lib import tensorlib
from lib import read
from NN import LSTM_NN
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch LSTM')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--experimental-trainer',action='store_true',help='Use experimental training algorithm')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    print('Using CUDA Mode.')
    args.device = torch.device('cuda')
else:
    print('Using CPU Mode.')
    args.device = torch.device('cpu')

tensorlib.torch_device = args.device

#text = support.readBook('./a.txt')
#text = support.readLines('./pickuplines.txt')


lstm = LSTM_NN(support.n_letters,[512,512,512],support.n_letters,device=tensorlib.torch_device)

n_iters = 10000
print_every = 50
plot_every = 10
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()
if args.experimental_trainer:
    text = support.readBook('./fun/shakespeare.txt')
    for i,i_max, output, loss in lstm.train_2(text,num_epochs=1000,backprop_interval=1):
        support.printProgress(i,i_max=i_max)
        total_loss += loss

        if i % print_every < print_every/2:
            support.printOW('%s (%d %d%%) %.4f' % (support.timeSince(start), i, i / i_max * 100, loss))

        if i % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    plt.figure()
    plt.scatter(range(len(all_losses)), all_losses, s=1)
    plt.savefig('train.png')

    print('') #Clear the progress bar
    lstm = lstm.eval()
    for sample in lstm.samples(start_letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        print('===================')
    plt.show()

else:

    text = read.toArray('./fun/shakespeare.txt')
    for iter in range(1, n_iters + 1):
        support.printProgress(iter,i_max=n_iters)
        output, loss = lstm.teach(*tensorlib.randomTrainingExample(text))
        total_loss += loss
        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (support.timeSince(start), iter, iter / n_iters * 100, loss))
        if iter % plot_every == 0:
            avg_loss = total_loss / plot_every
            if avg_loss < 5:
                # We don't want to plot outliers.
                all_losses.append(total_loss / plot_every)
            total_loss = 0

    torch.save(lstm.state_dict(), './network.data')

    plt.figure()
    plt.scatter(range(len(all_losses)), all_losses, s=1)
    plt.savefig('train.png')

    print('') #Clear the progress bar
    lstm = lstm.eval()
    for sample in lstm.samples(start_letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        print('===================')
    plt.show()
