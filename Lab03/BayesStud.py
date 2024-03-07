import pandas as pd

# dataset in stored in a DataFrame
df = pd.read_csv('data_vreme1.csv')

attributes = df.columns[0:4] #collection of attribute names
className = df.columns[-1] #name of column with class labels (Joc)

classLabels = set(df[className]) # the class labels are Da, Nu

instanceCount = len(df)

#iterate through attributes
print('\nAttributes :')
for attribName in attributes:
    print(attribName)

#iterate through values of a certain attribute (a column of the dataset):
attribName = attributes[0] # Starea vremii
print('\nValues of', attribName, ':')
for val in df[attribName]:
    print(val)

#iterate through available class labels:
print('\nClasses : ')
for label in classLabels:
    print(label)

#iterate through class labels from data set (last column)
print('\nClass labels in dataset : ')
for label in df[className]:
    print(label)

# a new unclassified instance might look like this:
testInstance = ['Soare', 'Mare', 'Normala', 'Absent']

# goal: determine probability values for the testInstance using Naive Bayes (see documentation) ... 







