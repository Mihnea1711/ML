import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv

if torch.cuda.is_available():
    print('GPU is available!')
    device = torch.device("cuda")  # Use GPU
else:
    print('GPU is not available. Running on CPU.')
    device = torch.device("cpu")  # Use CPU

dataFile = open('iris.csv', 'r')
dataset = csv.reader(dataFile)
# skip first row which contains csv header
nrAttributes = len(next(dataset))-1
dataset = list(dataset)
nrInstances = len(dataset)

instances = np.empty([nrInstances, nrAttributes])
labelStrings = []
labels = np.empty(nrInstances)

idx = 0
for row in dataset:
    instances[idx] = np.array(row[:nrAttributes])
    labelStrings.append(row[-1])
    idx += 1

uniqueLabelStrings = sorted(set(labelStrings))
labelDict = {}
labelIdx = 0
for label in uniqueLabelStrings:
    labelDict[label] = labelIdx
    labelIdx += 1

for i in range(len(labelStrings)):
    labels[i] = labelDict[labelStrings[i]]

#shuffle data
randomIdx = np.random.permutation(len(instances))
instances = instances[randomIdx]
labels = labels[randomIdx]

inputs = torch.Tensor(instances).to(device)  # Move tensors to device
targets = torch.Tensor(labels).long().to(device)  # Move tensors to device

class SimpleNN(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super(SimpleNN, self).__init__()
        
        self.fc1 = torch.nn.Linear(inputSize, hiddenSize)
        self.fc2 = torch.nn.Linear(hiddenSize, outputSize)
                
    def forward(self, input):
        output = self.fc1(input)
        output = F.relu(output)
        output = self.fc2(output)
        return output
        
    def train_model(self, inputs, targets):
        lossFunc = nn.CrossEntropyLoss()
        nrEpochs = 10
        learnRate = 0.01
        optimizer = torch.optim.SGD(self.parameters(), learnRate)
        loss = None
        for epoch in range(nrEpochs):
            correct = 0
            total = 0
            for input, target in zip(inputs, targets):            
                optimizer.zero_grad()
                predicted = self.forward(input.unsqueeze(0))
                loss = lossFunc(predicted, target.unsqueeze(0))
                loss.backward()
                optimizer.step()

                _, predicted_label = torch.max(predicted.data, 1)
                target = torch.tensor(target)
                total += 1
                correct += (predicted_label == target).sum().item()

            accuracy = 100 * correct / total
            print('Epoch', epoch, 'loss', loss.item(), 'accuracy', accuracy, '%')


myNet = SimpleNN(4, 5, 3).to(device)  # Move model to device
myNet.train_model(inputs, targets)
