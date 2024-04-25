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
        nrEpochs = 15
        learnRate = 0.01
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

            classification_error = misclassified / len(images)  # calculate classification error
            print('Epoch', epoch, 'classification error:', classification_error)

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print("Model loaded successfully.")
    return model

def load_single_image(image_path):
    img = mpimg.imread(image_path)
    # convert the image to grayscale if it has multiple channels
    if img.shape[-1] == 4:  # RGBA image
        img = img[..., :3]  # keep only the first 3 channels (RGB)
    img = np.mean(img, axis=-1)  # convert to grayscale by averaging color channels
    img = np.array([img])
    img = torch.Tensor(img)
    img = img.view([1, 1, img.shape[1], img.shape[2]])  # reshape the image to match model input shape
    return img


def test_single_image(model, image_path):
    test_image = load_single_image(image_path)
    print(test_image.shape)
    predicted = model.forward(test_image)
    return torch.argmax(predicted)

# ex 5 - load the model and test it on some random test images
# 44.png is from the dataset
# 3.png and 7.png are from web
# 3 is recognized correctly, 7 is recognized as 5 :P
if __name__ == '__main__':
    datasetDir = 'images/'
    _, _, imgWidth, imgHeight = load_images(datasetDir)

    # load the saved model
    loaded_model = SimpleCNN(imgWidth, imgHeight)
    loaded_model = load_model(loaded_model, 'my_model.pth')

    # test the trained model on a single test image
    test_image_path = 'test/3.png'
    predicted_label = test_single_image(loaded_model, test_image_path)
    print('Predicted label for test image:', predicted_label.item())
