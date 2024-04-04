import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5])
y = np.array([4, 3, 2.5, 1, 2, 3.5, 6, 4, 7, 1.5, 5, 2.5, 5.5, 3, 8, 7, 7.5, 6, 8.5, 9.5])

N = len(X)

def plot_line(X, y, b0, b1, label, filename):
    plt.scatter(X, y, color='blue', label='Date de antrenare')
    y_pred = b0 + b1 * X
    plt.plot(X, y_pred, color='red', label=label)
    plt.xlabel('Nr ore de studiu')
    plt.ylabel('Nota obtinuta')
    plt.title('Regr liniara')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)  # Save the plot as an image
    plt.show()

def calculate_coefficients_analytical(X, y):
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    b1 = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean) ** 2)
    b0 = y_mean - b1 * X_mean

    # calc val prezise
    y_pred = b0 + b1 * X
    # calc err medie patr (MSE)
    MSE = np.sum((y - y_pred) ** 2) / N
    print("Err medie patr (MSE) analitic:", MSE)

    return b0, b1


# regresie liniara folosind metoda gradientului descendent
def gradient_descent(X, y, alpha, iterations):
    b0 = 0
    b1 = 0
    N = len(X)
    for i in range(iterations):
        y_pred = b0 + b1 * X
        # deriv partiale
        d_b0 = (-2/N) * np.sum(y - y_pred)
        d_b1 = (-2/N) * np.sum(X * (y - y_pred))
        # actualiz coef
        b0 = b0 - alpha * d_b0
        b1 = b1 - alpha * d_b1

    # calc val prezise
    y_pred_gd = b0 + b1 * X
    # calc MSE
    MSE_gd = np.sum((y - y_pred_gd) ** 2) / N
    print("Err medie patr (MSE) folosind gradient descent:", MSE_gd)

    return b0, b1

b0_analytical, b1_analytical = calculate_coefficients_analytical(X, y)
print("coef calc analitic: y = {:.3f} + {:.3f} * x".format(b0_analytical, b1_analytical))
filename = "regr_analitic.png"
plot_line(X, y, b0_analytical, b1_analytical, label='Regr liniara (sol analitica)', filename=filename)

print("----------------------------------------------------------")

alpha = 0.01  # lr
iterations = 1000  # nr de iter
b0_gd, b1_gd = gradient_descent(X, y, alpha, iterations)
print("coef calc prin gradient descent: y = {:.3f} + {:.3f} * x".format(b0_gd, b1_gd))
filename = "regr_grad_descend.png"
plot_line(X, y, b0_gd, b1_gd, label='Regr liniara (gradient descent)', filename=filename)