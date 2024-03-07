import pandas as pd

# %%
# dataset in stored in a DataFrame
df = pd.read_csv('data_vreme1.csv')
df.head()
# %%
attributes = df.columns[0:4]  # collection of attribute names
className = df.columns[-1]  # name of column with Da/Nu

print(attributes)
print(className)
# %%
classLabels = set(df[className])  # the class labels are Da, Nu
instanceCount = len(df)

print(classLabels)
print(instanceCount)
# %%
# iterate through attributes
print('\nAttributes :')
for attribName in attributes:
    print(attribName)

# iterate through values of a certain attribute (a column of the dataset):
attribName = attributes[0]  # Starea vremii
print('\nValues of', attribName, ':')
for val in df[attribName]:
    print(val)

# iterate through available class labels:
print('\nClasses : ')
for label in classLabels:
    print(label)

# iterate through class labels from data set (last column)
print('\nClass labels in dataset : ')
for label in df[className]:
    print(label)

# a new unclassified instance might look like this:
testInstance = ['Soare', 'Mare', 'Normala', 'Absent']
# %%
# calc initial probs for Da/Nu
prior_probabilities = {}
for label in classLabels:
    prior_probabilities[label] = df[df[className] == label].shape[0] / instanceCount

print(prior_probabilities)


# %%
# func to calc conditionals P(A|B)
def calculate_conditional_probability_with_laplace(attribute, value, label):
    # get df subset for the label
    subset = df[df[className] == label]

    # count attr occurrences for the class
    count_with_value = subset[subset[attribute] == value].shape[0]

    # total count of instances for hte given class
    total_count = subset.shape[0]

    # With LaPlace correction
    return (count_with_value + 1) / (total_count + len(df[attribute].unique()))


# %%
# func to calc conditionals P(A|B)
def calculate_conditional_probability_without_laplace(attribute, value, label):
    # get df subset for the label
    subset = df[df[className] == label]

    # count attr occurrences for the class
    count_with_value = subset[subset[attribute] == value].shape[0]

    # total count of instances for hte given class
    total_count = subset.shape[0]

    # Without LaPlace correction
    return count_with_value / total_count


# %%
# func to calc overall prob for an instance
def calculate_instance_probability(instance, use_laplace=True):
    probabilities = {}
    for label in classLabels:
        # we first add the probabilities for the Da/Nu label
        probabilities[label] = prior_probabilities[label]
        for i, value in enumerate(instance):
            # calc cond prob for each attr and multiply it with initial prob
            if use_laplace:
                probabilities[label] *= calculate_conditional_probability_with_laplace(attributes[i], value, label)
            else:
                probabilities[label] *= calculate_conditional_probability_without_laplace(attributes[i], value, label)

    return probabilities


# %%
# predict a test instance
testInstance = ['Soare', 'Mare', 'Normala', 'Absent']
probabilities = calculate_instance_probability(testInstance)

# Print probabilities for each class
for label, probability in probabilities.items():
    print(f"Probability for class '{label}': {probability}")

    # Choose class with maximum probability as the predicted class
predicted_class = max(probabilities, key=probabilities.get)
print(f"\nPredicted class: {predicted_class}")


# %%
################ console ui
# %%
# Function to print probabilities for each class
def print_probabilities(probabilities):
    for label, probability in probabilities.items():
        print(f"Probability for class '{label}': {probability}")


# %%
# Function to choose the class with maximum probability
def predict_class(probabilities):
    return max(probabilities, key=probabilities.get)


# %%
# while True:
#     testInstance = input("Enter the test instance separated by commas (e.g., Soare, Mare, Normala, Absent): ").strip().split(',')
#     use_laplace = input("Do you want to use Laplace correction? (yes/no): ").lower().strip()
#
#     if use_laplace == 'yes':
#         probabilities = calculate_instance_probability(testInstance, use_laplace=True)
#     elif use_laplace == 'no':
#         probabilities = calculate_instance_probability(testInstance, use_laplace=False)
#     else:
#         print("Invalid input. Please enter 'yes' or 'no' for Laplace correction.")
#         continue
#
#     print("\nProbabilities for each class:")
#     print_probabilities(probabilities)
#
#     predicted_class = predict_class(probabilities)
#     print(f"\nPredicted class: {predicted_class}")
#
#     continue_execution = input("Do you want to classify another instance? (yes/no): ").lower().strip()
#     if continue_execution != 'yes':
#         break