import pandas as pd
import numpy as np

# %%
# Citim datele
df = pd.read_csv('data_vreme2.csv')
df.head()
# %%
attributes = df.columns[0:-1]  # collection of attribute names
className = df.columns[-1]  # name of column with Da/Nu

print(attributes)
print(className)
# %%
classLabels = df[className].unique()
print(classLabels)

instanceCount = len(df)
print(instanceCount)
# %%
# calc initial probs for Da/Nu
prior_probabilities = {}
for label in classLabels:
    prior_probabilities[label] = df[df[className] == label].shape[0] / instanceCount
# %%
# calc mean and std dev for temp attribute for each Da/Nu class
temp_stats = {}
for label in classLabels:
    temp_stats[label] = {
        'mean': df[df[className] == label]['Temperatura'].mean(),
        'stdev': df[df[className] == label]['Temperatura'].std()
    }


# %%
def calculate_probability_gaussian(x, mean, stdev):
    exponent = np.exp(-((x - mean) ** 2) / (2 * stdev ** 2))
    return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent


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
# func to calc overall prob for an instance
def calculate_instance_probability(instance, use_laplace=True):
    probabilities = {}
    for label in classLabels:
        # Initialize probability with prior probability
        probabilities[label] = prior_probabilities[label]
        for i, value in enumerate(instance[:-1]):  # Skip last attribute (class label)
            if attributes[i] == 'Temperatura':
                # Calculate conditional probability using Gaussian probability density function
                mean = temp_stats[label]['mean']
                stdev = temp_stats[label]['stdev']
                probabilities[label] *= calculate_probability_gaussian(float(value), mean, stdev)
            else:
                # Calculate conditional probability for other attributes
                probabilities[label] *= calculate_conditional_probability_with_laplace(attributes[i], value, label)
    return probabilities


# %%
# predict a test instance
testInstance = ['Soare', '24', 'Mare', 'Prezent']
probabilities = calculate_instance_probability(testInstance)

# Print probabilities for each class
for label, probability in probabilities.items():
    print(f"Probability for class '{label}': {probability}")

# Choose class with maximum probability as the predicted class
predicted_class = max(probabilities, key=probabilities.get)
print(f"\nPredicted class: {predicted_class}")
