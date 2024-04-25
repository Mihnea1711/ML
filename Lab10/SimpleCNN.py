import numpy as np
import matplotlib.image as mpimg
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SimpleCNN(nn.Module):
    def __init__(self, imgWidth, imgHeight):
        super(SimpleCNN, self).__init__()

        inputWidth = imgWidth
        inputHeight = imgHeight
        nrConvFilters = 3
        convFilterSize = 5
        poolSize = 2
        outputSize = 10

        self.convLayer = nn.Conv2d(1, nrConvFilters, convFilterSize)
        self.poolLayer = nn.MaxPool2d(poolSize)
        fcInputSize = (inputWidth - 2*(convFilterSize // 2)) * (inputWidth - 2*(convFilterSize // 2)) * nrConvFilters // (2 * poolSize)
        self.fcLayer = nn.Linear(fcInputSize, outputSize)

    def forward(self, input):
        output = self.convLayer(input)
        output = self.poolLayer(output)
        output = F.relu(output)
        output = output.view([1, -1])
        output = self.fcLayer(output)
        return output

    def train_model(self, images, labels):
        lossFunc = nn.CrossEntropyLoss()
        # played with these params for ex 2
        nrEpochs = 12
        learnRate = 0.008
        optimizer = torch.optim.SGD(self.parameters(), learnRate)

        for epoch in range(nrEpochs):
            misclassified = 0  # initialize counter for misclassified images

            for image, label in zip(images, labels):
                optimizer.zero_grad()
                predicted = self.forward(image.unsqueeze(0))
                loss = lossFunc(predicted, label.unsqueeze(0))
                loss.backward()
                optimizer.step()

                # check if prediction is incorrect
                if torch.argmax(predicted) != label:
                    misclassified += 1

            # ex 1 - calculate classification error
            classification_error = misclassified / len(images)
            print('Epoch', epoch, 'classification error:', classification_error)

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print("Model saved successfully.")

def main():
    datasetDir = 'images/'
    images, labels, imgWidth, imgHeight = load_images(datasetDir)

    # ex 3 - split dataset into train, test
    train_images, test_images, train_labels, test_labels = split_dataset(images, labels)

    myCNN = SimpleCNN(imgWidth, imgHeight)
    myCNN.train_model(train_images, train_labels)

    # ex 3 - evaluate the network on the test set
    test_misclassified = 0
    for image, label in zip(test_images, test_labels):
        predicted = myCNN.forward(image.unsqueeze(0))
        if torch.argmax(predicted) != label:
            test_misclassified += 1

    test_classification_error = test_misclassified / len(test_images)
    print('Classification error on test set:', test_classification_error)

    # save model for future use in case of good test err
    if test_classification_error < 0.1:
        save_model(myCNN, "my_model.pth")

if __name__ == "__main__":
    main()
