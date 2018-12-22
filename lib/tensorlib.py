import torch
from random import randint
from lib import support

torch_device = None

# One hot Kronecker Delta Tensor of first to last letters (not including EOS) for input
def inputTensor(line,device=torch_device):
    tensor = torch.zeros(len(line), 1, support.n_letters,device=torch_device)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][support.all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line,device=torch_device):
    letter_indices = [support.all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indices.append(support.n_letters - 1) # EOS
    return torch.LongTensor(letter_indices).to(device=torch_device)

def getTrainingBatches(string,batchSize=1000):
    inTensor = torch.split(inputTensor(string),batchSize)
    outTensor = torch.split(targetTensor(string),batchSize)
    for i in range(inTensor.size(0)):
        yield inTensor[i],outTensor[i]

def randomChoice(l):
    return l[randint(0, len(l) - 1)]

# Make input, and target tensors from a random line
def randomTrainingExample(data):
    line = randomChoice(data)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return input_line_tensor, target_line_tensor
