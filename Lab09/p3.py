import torch
import torch.nn.functional as F
from torch import nn

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
        nrEpochs = 100
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

if __name__ == '__main__':

    if torch.cuda.is_available():
        print('GPU is available!')
        device = torch.device("cuda")  # Use GPU
    else:
        print('GPU is not available. Running on CPU.')
        device = torch.device("cpu")  # Use CPU

    # Re-create the model with the same architecture
    loaded_model = SimpleNN(4, 5, 3).to(device)
    # Load the saved parameters into the model instance
    loaded_model.load_state_dict(torch.load('trained_model.pth'))
    # Set the model to evaluation mode
    loaded_model.eval()

    sepal_length = float(input("Enter sepal length: "))
    sepal_width = float(input("Enter sepal width: "))
    petal_length = float(input("Enter petal length: "))
    petal_width = float(input("Enter petal width: "))

    # Create input tensor
    input_tensor = torch.tensor([[sepal_length, sepal_width, petal_length, petal_width]], dtype=torch.float32).to(device)

    # Forward pass to get the output
    output = loaded_model(input_tensor)

    # Apply softmax to get probabilities
    probabilities = F.softmax(output, dim=1)

    # Display the probabilities for each class
    print("Probabilities for each class:")
    print("Setosa:", probabilities[0][0].item())
    print("Versicolor:", probabilities[0][1].item())
    print("Virginica:", probabilities[0][2].item())