import pandas as pd
from datasetprocessor import *
from sklearn.metrics import accuracy_score

class Node:
    def __init__(self, name='root', branchName='', children=None):
        if children is None:
            children = []
        self.name = name
        self.branchName = branchName
        self.children = children.copy()


def printNode(node, parentStrLen=0):
    indent = ' ' * parentStrLen
    if node.branchName == '':  # root node
        nodeStr = node.name
        print(indent, nodeStr)
    else:
        nodeStr = '--' + node.branchName + '--> ' + node.name
        print(indent, nodeStr)

    for child in node.children:
        printNode(child, parentStrLen + len(nodeStr))

def BuildTree(ds, branchName, attribList):
    # branchName este numele ramurii dintre nodul curent si parintele sau
    # attribList este o lista ce contine numele atributelor

    node = Node()
    node.branchName = branchName
    dp = DatasetProcessor(ds)

    # daca toate instantele din ds au aceeasi clasa, atunci
    if len(dp.classLabels) == 1:
        node.name = list(dp.classLabels)[0]
        return node

    # daca lista atributelor este goala, atunci
    if len(attribList) == 0:
        node.name = dp.getLabelWithMaxCount()
        return node

    # cautam atr cu indice gini minim
    minGini = float('inf')
    bestAttribute = None
    for attribute in attribList:
        attributeGini = dp.getGini()
        if attributeGini < minGini:
            minGini = attributeGini
            bestAttribute = attribute

    node.name = bestAttribute
    Avalues = dp.getAttribValues(bestAttribute)

    for val in Avalues:
        subset = dp.getSubset(bestAttribute, val)

        if len(subset) == 0:
            node.children.append(Node(name=dp.getLabelWithMaxCount()))
        else:
            newAttribList = [attr for attr in attribList if attr != bestAttribute]
            node.children.append(BuildTree(subset, val, newAttribList))

    return node


def predict(node, instance):
    # daca nodul e frunza, ret numele nodului
    if not node.children:
        return node.name

    # daca nu, cautam in instanta de test
    attribute_name = node.name
    attribute_value = instance[attribute_name]

    # cautam copilul coresp pentr val atr in arbore
    for child in node.children:
        if child.branchName == attribute_value:
            # facem recursiv predictia
            return predict(child, instance)

    # # daca val atr nu a fost gasita in copii, at ret clasa majoritara
    # return _dp.getLabelWithMaxCount()


# fct pt constr unui arbore
def BuildTreeWithAttributes(ds, attribList):
    node = BuildTree(ds, '', attribList)
    return node

# fct pt gen padure
def generateForest(ds, num_subsets, num_trees):
    dp = DatasetProcessor(ds)
    forest = []
    for _ in range(num_subsets):
        subset = dp.generateRandomSubset()
        for _ in range(num_trees):
            num_attributes = random.randint(2, 3)
            selected_attributes = random.sample(list(ds.columns[:-1]), num_attributes)
            tree = BuildTreeWithAttributes(subset, selected_attributes)
            forest.append(tree)
    return forest


if __name__ == '__main__':
#1 ---------------------------------------------------------
    # test data
    ds_test = pd.read_csv('data_vreme5.csv')

    # build tree using train data
    ds_train = pd.read_csv('data_vreme3.csv')
    _dp = DatasetProcessor(ds_train)
    Alist = list(_dp.attributes)
    root = BuildTree(ds_train, '', Alist)

    # print the tree
    printNode(root)

    # make predictions
    correct_predictions = 0
    total_instances = len(ds_test)
    for _, _instance in ds_test.iterrows():
        prediction = predict(root, _instance)
        if prediction == _instance[_dp.className]:
            correct_predictions += 1

    # calc classification err
    classification_error = 1 - (correct_predictions / total_instances)
    print("Classification error:", classification_error)

#2 ---------------------------------------------------------
    _ds = pd.read_csv('data_vreme3.csv')

    _num_subsets = 10  # nr subm aleatoare
    _num_trees = 5  # nr arbori in padure
    _forest = generateForest(_ds, _num_subsets, _num_trees)

    # constr lista de atr pt fiec arbore
    attributes_list = []
    for _tree in _forest:
        attributes_list.append(_tree.name)

    # det pred pt fiec ex din setul de test
    predictions = []
    for _, _instance in ds_test.iterrows():
        votes = []
        for _tree in _forest:
            prediction = predict(_tree, _instance)
            votes.append(prediction)
        # pred are cele mai multe voturi
        final_prediction = max(set(votes), key=votes.count)
        predictions.append(final_prediction)

    # calc err de clasif pt padure
    accuracy_forest = accuracy_score(ds_test[_dp.className], predictions)
    print("Eroare de clasificare pentru padurea de arbori:", accuracy_forest)


