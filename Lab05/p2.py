import pandas as pd
import random
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    # Citirea setului de date
    ds = pd.read_csv('data_vreme3.csv')
    dp = DatasetProcessor(ds)

    # Generare pădure de arbori
    num_subsets = 10  # Numărul de submulțimi aleatoare
    num_trees = 5  # Numărul de arbori în pădure
    forest = generateForest(ds, num_subsets, num_trees)

    # Construirea listei de atribute pentru fiecare arbore
    attributes_list = []
    for tree in forest:
        attributes_list.append(tree.name)

    # Determinarea predicțiilor pentru fiecare exemplu din setul de test
    predictions = []
    for _, instance in ds_test.iterrows():
        votes = []
        for tree in forest:
            prediction = predict(tree, instance)
            votes.append(prediction)
        # Predicția este clasa cu cele mai multe voturi
        final_prediction = max(set(votes), key=votes.count)
        predictions.append(final_prediction)

    # Calculul erorii de clasificare pentru "pădurea" de arbori
    accuracy_forest = accuracy_score(ds_test[dp.className], predictions)

    # Compararea cu eroarea de clasificare a arborelui determinat anterior
    print("Eroare de clasificare pentru padurea de arbori:", accuracy_forest)
