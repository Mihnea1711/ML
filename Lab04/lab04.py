import pandas as pd
from datasetprocessor import *

class Node:
    def __init__(self, name = 'root', branchName = '', children = []):
        self.name = name
        self.branchName = branchName
        self.children = children.copy()

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.name) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return str(self)

def printNode(node, parentStrLen = 0):
    indent = ' ' * parentStrLen
    nodeStr = ''
    if node.branchName == '': #root node
        nodeStr = node.name
        print(indent, nodeStr)
    else:
        nodeStr = '--'+node.branchName+'--> '+node.name
        print(indent, nodeStr)

    for child in node.children:
        printNode(child, parentStrLen + len(nodeStr))


def ID3(df, branchName, attribList):
    # branchName este numele ramurii dintre nodul curent si parintele sau
    # attribList este o lista ce contine numele atributelor
    dp = DatasetProcessor(df)
    node = Node()
    node.branchName = branchName

    # daca toate instantele din ds au aceeasi clasa, atunci
    # node.name = numele acelei clase
    if len(set(df[df.columns[-1]])) == 1:
        node.name = df[df.columns[-1]].iloc[0]  # numele acelei clase
        return node

    # daca lista atributelor este goala, atunci
    # node.name = clasa care apare cel mai frecvent in ds
    if not attribList:
        node.name = dp.getLabelWithMaxCount()  # clasa care apare cel mai frecvent in ds
        return node

    # Alegem atributul cu entropia minima
    A = min(attribList, key=lambda x: dp.getEntropy()) # atributul cu entropia minima
    node.name = A
    Avalues = dp.getAttribValues(A)  # valorile posibile ale atributului A

    for val in Avalues:
        subset = dp.getSubset(A, val)  # submultimea lui df care conține doar instantele cu valoarea val a atributului A

        if len(subset) == 0:  # daca submultimea este goala
            node.children.append(Node(name=dp.getLabelWithMaxCount()))  # un nou nod cu numele dat de clasa care apare cel mai frecvent în df
        else:
            newAttribList = [attr for attr in attribList if attr != A]  # o noua lista ce contine atributele din attribList, mai putin atributul A
            node.children.append(ID3(subset, val, newAttribList))  # se apeleaza recursiv functia pentru generarea nodului descendent

    return node


def classify_instance(node, instance):
    # verif daca nodul frunza
    if not node.children:
        return node.name  # ret clasa din nodul frunza

    # numele atr
    split_attribute = node.name

    # val atr din instanta
    attribute_value = instance[split_attribute]

    # cautam val atr in copii nodului curent
    for child in node.children:
        if child.branchName == attribute_value:
            return classify_instance(child, instance)  # cont clasif cu nodul copil

    # ret clasa majoraitara daca nu am gasit nimic
    return node.children[0].name


if __name__ == '__main__':
    df = pd.read_csv('data_vreme3.csv')
    dp = DatasetProcessor(df)

    Alist = list(dp.attributes)
    root = ID3(df, '', Alist)

    printNode(root)

    print("# Predictions #")

    # citim setul de date de test
    df_test = pd.read_csv('data_vreme4.csv')

    # clasif fiecare inst din setul de date si calc err de clasificare
    classification_errors = 0
    for index, instance in df_test.iterrows():
        predicted_class = classify_instance(root, instance)
        actual_class = instance.iloc[-1]
        print(f"Instance:", instance.values[:-1])
        print("Predicted class:", predicted_class)
        print("Actual class:", actual_class)
        if predicted_class != actual_class:
            classification_errors += 1
    classification_error_rate = classification_errors / len(df_test)

    print("Eroarea de clasificare a arborelui ID3 este:", classification_error_rate)


