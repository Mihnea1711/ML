import torch as tr
import torch.nn as nn


# Check if GPU is available and set the device accordingly
device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

class SimpleRNN(nn.Module):
    def __init__(self, inputSize, outputSize, lstmLayerSize, noLSTMLayers):
        super(SimpleRNN, self).__init__()
        self.inputSize = inputSize
        self.lstmLayerSize = lstmLayerSize
        self.outputSize = outputSize
        self.noLSTMLayers = noLSTMLayers

        self.lstmLayer = nn.LSTM(self.inputSize, self.lstmLayerSize, self.noLSTMLayers)
        self.outLayer = nn.Linear(self.lstmLayerSize, self.outputSize)

    def forward(self, input):
        input = input.view(-1, 1, 1).to(device)
        lstmOut, hidden = self.lstmLayer(input)
        outLayerInput = lstmOut[-1, 0, :] 
        predictedOut = self.outLayer(outLayerInput)
        return predictedOut

    def train(self, inputs, targets):
        noEpochs = 750
        learnRate = 0.0002
        optimizer = tr.optim.Adam(self.parameters(), learnRate)
        lossFunc = nn.CrossEntropyLoss()

        for epoch in range(noEpochs):
            total_loss = 0
            correct_predictions = 0

            for input, target in zip(inputs, targets):
                optimizer.zero_grad()
                predicted = self.forward(input)
                loss = lossFunc(predicted.unsqueeze(0), target.unsqueeze(0))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Calculate accuracy
                _, predicted_class = tr.max(predicted, 0)
                if predicted_class == target:
                    correct_predictions += 1

            avg_loss = total_loss / len(inputs)
            accuracy = correct_predictions / len(inputs)

            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

