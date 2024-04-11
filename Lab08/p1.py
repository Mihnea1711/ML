import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# predict probability of passing exam with a chosen value
def predict_probability(hours):
    return sigmoid(w0 + w1 * hours)

# training data
hours_studied = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5])
passed_exam = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# init weights
w0 = np.random.uniform(-0.5, 0.5)
w1 = np.random.uniform(-0.5, 0.5)

# learning rate and number of epochs
alpha = 0.01
epochs = 10000

# gradient descent
errors = []
for epoch in range(epochs):
    # predicted values
    predicted = sigmoid(w0 + w1 * hours_studied)

    # cross-entropy error
    error = -np.mean(passed_exam * np.log(predicted) + (1 - passed_exam) * np.log(1 - predicted))
    errors.append(error)

    # gradients
    dw0 = np.mean(predicted - passed_exam)
    dw1 = np.mean((predicted - passed_exam) * hours_studied)

    # update weights
    w0 -= alpha * dw0
    w1 -= alpha * dw1

# learned coefs
print("Learned coefficients:")
print("w0 =", w0)
print("w1 =", w1)

# error evolution plot
plt.plot(range(epochs), errors)
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Error')
plt.title('Error Evolution over Epochs')
plt.show()

# logistic regression curve
x_values = hours_studied
y_values = sigmoid(w0 + w1 * hours_studied)  # Compute probabilities for each point
plt.plot(x_values, y_values, label='Logistic Regression Curve', color='red')
# original training data points
plt.scatter(hours_studied, passed_exam, label='Original Data', color='blue')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()

# test the model
new_hours = 3.75
print("Probability of passing the exam with {} hours of study: {:.2f}".format(new_hours, predict_probability(new_hours)))
