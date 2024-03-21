import pandas as pd
import math

class DatasetProcessor:
    def __init__(self, ds):
        self.ds = ds
        self.className = ds.columns[-1]
        self.classLabels = set(ds[self.className])
        self.attributes = ds.columns[0:len(ds.columns)-1]
        self.instanceCount = len(self.ds)

        self.labelCount = {}
        self.labelProb = {}

        for label in self.classLabels:
            self.labelCount[label] = len(ds[ds[self.className] == label])
            self.labelProb[label] = self.labelCount[label] / self.instanceCount

    def getAttribValues(self, A):
        return list(set(self.ds[A]))

    # submultimea care contine doar instantele cu valoarea Aval a atributului A
    def getSubset(self, A, Aval):
        return self.ds[self.ds[A] == Aval]

    # clasa majoritara
    def getLabelWithMaxCount(self):
        maxLabel = ''
        maxCount = 0
        for label in self.classLabels:
            if self.labelCount[label] > maxCount:
                maxLabel = label
                maxCount = self.labelCount[label]
        return maxLabel

    # entropia intregului set de date
    def getEntropy(self):
        entropy = 0
        for label in self.classLabels:
            p = self.labelProb[label]
            entropy -= p * math.log2(p) if p > 0 else 0
        return entropy

    # entropia subsetului de date corespunzatoare lui A
    def getSubsetEntropy(self, A):
        overall_entropy = 0
        for val in self.getAttribValues(A):
            subset = self.getSubset(A, val)
            subsetProbability = len(subset) / self.instanceCount
            subsetLabelProbability = subset[self.className].value_counts(normalize=True)
            subsetEntropy = sum(-p * math.log2(p) for p in subsetLabelProbability)
            overall_entropy += subsetProbability * subsetEntropy
        return overall_entropy

