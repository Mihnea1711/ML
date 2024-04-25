import torch.nn as nn
import numpy as np
import matplotlib.image as mpimg
import os
import torch

def load_images(datasetDir):
    images = None
    labels = None
    firstImg = True
    imgWidth = None
    imgHeight = None
    nrClasses = len(os.listdir(datasetDir))
    for classDir in os.listdir(datasetDir):
        label = int(classDir)
        imgDir = os.path.join(datasetDir, classDir)
        for imgFile in os.listdir(imgDir):
            img = mpimg.imread(os.path.join(imgDir, imgFile))
            if firstImg:
                imgWidth = img.shape[0]
                imgHeight = img.shape[1]
                images = np.array([img])
                labels = np.array([label])
                firstImg = False
            else:
                images = np.vstack([images, [img]])
                labels = np.append(labels, label)
    nrImages = images.shape[0]

    # shuffle data
    randomIdx = np.random.permutation(len(images))
    images = images[randomIdx]
    labels = labels[randomIdx]

    images = torch.Tensor(images)
    images = images.view([images.shape[0], 1, images.shape[2], images.shape[1]])
    labels = torch.Tensor(labels).long()

    return images, labels, imgWidth, imgHeight


def split_dataset(images, labels, split_ratio=0.8):
    split_idx = int(len(images) * split_ratio)
    train_images, test_images = images[:split_idx], images[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    return train_images, test_images, train_labels, test_labels

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleNN, self).__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]
        # Adjusted the input size of the first linear layer
        prev_size = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:]:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())  # relu activation func
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))  # output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_data, train_labels, epochs=10, lr=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(train_data)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

def test_model(model, test_images, test_labels):
    model.eval()  # set on eval mode
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in zip(test_images, test_labels):
            image = image.view(1, -1)
            outputs = model(image)  # apply model on test data
            _, predicted = torch.max(outputs.data, 1)  # get predictions
            total += label.item()
            label = torch.tensor(label)
            correct += (predicted == label).sum().item()  # nr of correct predictions

    accuracy = correct / total  # get accuracy
    print('Accuracy on test set: {:.2f}%'.format(100 * accuracy))



if __name__ == '__main__':
    datasetDir = 'images/'
    images, labels, imgWidth, imgHeight = load_images(datasetDir)

    train_images, test_images, train_labels, test_labels = split_dataset(images, labels)
    input_size = 28 * 28  # input img size
    hidden_sizes = [128, 64]  # 2 hidden layers of 128 and 64 neurons
    output_size = 10  # 10 output classes
    simple_nn = SimpleNN(input_size, hidden_sizes, output_size)

    simple_nn.train_model(train_images.view(train_images.size(0), -1), train_labels, epochs=150, lr=0.001)

    test_model(simple_nn, test_images.view(test_images.size(0), -1), test_labels)

