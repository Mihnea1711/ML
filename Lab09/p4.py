import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    print('GPU is available!')
    device = torch.device("cuda")  # Use GPU
else:
    print('GPU is not available. Running on CPU.')
    device = torch.device("cpu")  # Use CPU

MAX_EPOCHS = 100

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

    def train_model(self, inputs, targets, num_epochs):
        lossFunc = nn.CrossEntropyLoss()
        nrEpochs = num_epochs
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
            print('Epoch', epoch, '/' , num_epochs ,'loss', loss.item(), 'accuracy', accuracy, '%')

    def evaluate(self, test_inputs, test_targets):
        correct = 0
        total = 0
        with torch.no_grad():
            for input, target in zip(test_inputs, test_targets):
                predicted = self.forward(input.unsqueeze(0))
                _, predicted_label = torch.max(predicted.data, 1)
                target = torch.tensor(target)
                total += 1
                correct += (predicted_label == target).sum().item()

        accuracy = 100 * correct / total
        print('Test accuracy:', accuracy, '%')
        return accuracy

def train_and_evaluate(model: SimpleNN, inputs_train, targets_train, inputs_test, targets_test, num_epochs: int):
    model.train_model(inputs_train, targets_train, num_epochs=num_epochs)
    test_accuracy = model.evaluate(inputs_test, targets_test)
    return test_accuracy

if __name__ == '__main__':
    dataFile = open('iris.csv', 'r')
    dataset = csv.reader(dataFile)
    # skip first row which contains csv header
    nrAttributes = len(next(dataset)) - 1
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

    # shuffle data
    randomIdx = np.random.permutation(len(instances))
    instances = instances[randomIdx]
    labels = labels[randomIdx]

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(instances, labels, test_size=0.34, random_state=42)

    # Convert numpy arrays to PyTorch tensors
    inputs_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    targets_train = torch.tensor(y_train, dtype=torch.long).to(device)
    inputs_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    targets_test = torch.tensor(y_test, dtype=torch.long).to(device)

    my_net = SimpleNN(4, 5, 3).to(device)  # Move model to device
    max_accuracy = 0
    epochs_for_max_accuracy = 0
    for num_epochs in range(10, MAX_EPOCHS + 1, 10):
        test_accuracy = train_and_evaluate(my_net, inputs_train, targets_train, inputs_test, targets_test, num_epochs)
        if test_accuracy >= max_accuracy:
            max_accuracy = test_accuracy
            epochs_for_max_accuracy = num_epochs

    print(f"Max Accuracy: {max_accuracy} acieved with {epochs_for_max_accuracy} epochs")
