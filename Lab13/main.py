import random

import torch as tr
from torch.nn.functional import softmax

from SimpleRNN import SimpleRNN


desiredTextLength = 20
# Check if GPU is available and set the device accordingly
device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

def main():
    textFile = open("rnn_text.txt")
    text = textFile.read()

    # separate punctuation by spaces
    punctuation = [',', '.', ':', ';', '?', '!', '"', "'"]
    tempCharList = [' ' + c if c in punctuation else c for c in text]
    text = ''.join(tempCharList)
    text = text.lower()

    # build vocabulary
    words = text.split()
    vocabulary = list(set(words))
    vocabulary.sort()

    # labels of words from training text:
    wordLabels = [vocabulary.index(w) for w in words]

    # build training data
    sequenceLength = 5
    noSequences = 100
    # random indices from training text:
    indices = random.sample(range(len(words) - sequenceLength - 1), noSequences)

    inputs = [wordLabels[i: i + sequenceLength] for i in indices]
    targets = [wordLabels[i + sequenceLength] for i in indices]

    inputs = tr.tensor(inputs, dtype=tr.float).to(device)
    targets = tr.tensor(targets, dtype=tr.long).to(device)

    myRNN = SimpleRNN(1, len(vocabulary), 64, 4).to(device)  # Adjusted LSTM size and number of layers, and moved to device
    myRNN.train(inputs, targets)

    while True:
        sentence = input('Introduceti %s cuvinte sau _quit: ' % sequenceLength)
        sentence.strip()
        if sentence == "_quit":
            break
        words = sentence.split()
        if len(words) != sequenceLength:
            continue

        try:
            inputLabels = [vocabulary.index(w) for w in words]
        except:
            print('Cuvintele introduse trebuie sa faca parte din vocabular.')
            continue

        sentence += ' '
        rnnInput = tr.tensor(inputLabels, dtype=tr.float).to(device)
        for i in range(desiredTextLength - sequenceLength):
            rnnOut = myRNN(rnnInput)
            # outputLabel = rnnOut.argmax().item()

            probs = softmax(rnnOut, dim=-0)
            outputLabel = tr.multinomial(probs, 1).item()

            outputWord = vocabulary[outputLabel]
            # print(outputWord)
            sentence += outputWord
            sentence += ' '
            rnnInput = tr.cat([rnnInput[1:], tr.tensor([outputLabel]).to(device)])

        print(sentence)


if __name__ == '__main__':
    main()