import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
from sklearn.model_selection import train_test_split, KFold


def get_device():
    if torch.cuda.is_available():
        print('GPU is available!')
        return torch.device("cuda")  # Use GPU
    else:
        print('GPU is not available. Running on CPU.')
        return torch.device("cpu")  # Use CPU

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

    def train_model(self, inputs, targets, num_epochs, kfold=5):
        kf = KFold(n_splits=kfold, shuffle=True)
        fold_accuracies = []

        for fold, (train_indices, val_indices) in enumerate(kf.split(inputs)):
            inputs_train, inputs_val = inputs[train_indices], inputs[val_indices]
            targets_train, targets_val = targets[train_indices], targets[val_indices]

            optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
            loss_func = nn.CrossEntropyLoss()

            for epoch in range(num_epochs):
                self.train()
                optimizer.zero_grad()
                outputs = self.forward(inputs_train)
                loss = loss_func(outputs, targets_train)
                loss.backward()
                optimizer.step()

                self.eval()
                with torch.no_grad():
                    val_outputs = self.forward(inputs_val)
                    _, predicted_labels = torch.max(val_outputs, 1)
                    targets_val = torch.tensor(targets_val)
                    correct = (predicted_labels == targets_val).sum().item()
                    total = len(val_indices)
                    accuracy = 100 * correct / total
                    print(f"Fold {fold + 1}, Epoch {epoch + 1}, Validation Accuracy: {accuracy:.2f}%")

                fold_accuracies.append(accuracy)

        average_accuracy = np.mean(fold_accuracies)
        print(f"Average Validation Accuracy across all folds: {average_accuracy:.2f}%")

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

def read_preprocess_data(datafile_path):
    dataFile = open(datafile_path, 'r')
    csv_data_file = csv.reader(dataFile)

    nrAttributes = len(next(csv_data_file)) - 1
    dataset = list(csv_data_file)
    nrInstances = len(dataset)

    instances = np.empty((nrInstances, nrAttributes))
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

    randomIdx = np.random.permutation(len(instances))
    instances = instances[randomIdx]
    labels = labels[randomIdx]

    return instances, labels

def split_and_convert_to_tensors(instances, labels, test_size=0.2, random_state=42, device=None):
    X_train, X_test, y_train, y_test = train_test_split(instances, labels, test_size=test_size, random_state=random_state)

    inputs_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    targets_train = torch.tensor(y_train, dtype=torch.long).to(device)
    inputs_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    targets_test = torch.tensor(y_test, dtype=torch.long).to(device)

    return inputs_train, targets_train, inputs_test, targets_test

if __name__ == '__main__':
    device = get_device()

    file_path = 'iris.csv'
    instances, labels = read_preprocess_data(file_path)

    inputs_train, targets_train, inputs_test, targets_test = split_and_convert_to_tensors(instances, labels, device=device)

    myNet = SimpleNN(4, 5, 3).to(device)  # Move model to device
    myNet.train_model(inputs_train, targets_train, 100, 7)

    # Evaluate the model on the test set
    test_accuracy = myNet.evaluate(inputs_test, targets_test)
    if test_accuracy > 0.9:
        torch.save(myNet.state_dict(), 'trained_model.pth')