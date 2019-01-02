import argparse
from lib import support
from lib import tensorlib
from NN import LSTM_NN
import torch
import time
import matplotlib.pyplot as plt

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

#text = support.readLines('./fun/HPSS.txt', sentenceDelimiter='.')
#text = support.readLines('./fun/HPSS.txt', mode='sequence')
#text = support.readLines('./e.txt',mode='sequence')
text = support.readBook('./e.txt')

lstm = LSTM_NN(support.n_letters,[256,256],support.n_letters,device=tensorlib.torch_device)

n_iters = 2000
print_every = 50
plot_every = 10
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

for i,i_max, output, loss in lstm.train_2(text,num_epochs=1,backprop_interval=100):
    support.printProgress(i,i_max=i_max)
    total_loss += loss

    if i % print_every == 0:
        print('%s (%d %d%%) %.4f' % (support.timeSince(start), i, i / i_max * 100, loss))

    if i % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

''' OLDER TRAINING ALGO
for iter in range(1, n_iters + 1):
    support.printProgress(iter,i_max=n_iters)
    output, loss = lstm.train(*tensorlib.randomTrainingExample(text))
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (support.timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0
'''
plt.figure()
plt.scatter(range(len(all_losses)), all_losses, s=1)
plt.savefig('train.png')
print('') #Clear the progress bar
for sample in lstm.samples(start_letters='T'):
    print(sample)
