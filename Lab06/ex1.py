import numpy as np
import matplotlib.pyplot as plt


def f1(x):
    return x ** 4 - 7 * x ** 3 + 14 * x ** 2 - 8 * x


def f1Grad(x):
    return 4 * x ** 3 - 21 * x ** 2 + 28 * x - 8


f1Limits = [-0.2, 4.4]

if __name__ == '__main__':
    f = lambda x: f1(x)
    fGrad = lambda x: f1Grad(x)
    xMin, xMax = f1Limits

    ### Gradient Descent

    x = xMin + 0.1
    alpha = 0.01  # lr
    nrIter = 1000
    epsilon = 1e-6  # crit de conv

    xGD = np.array([x])

    for i in range(nrIter):
        x_new = x - alpha * fGrad(x)
        xGD = np.append(xGD, x_new)
        if abs(f(x_new) - f(x)) < epsilon:
            print("Converged at iteration", i + 1)
            break
        x = x_new

    print('Minimum found at x =', x, ', f(x) =', f(x))

    ### Plotting

    plt.rc('font', size=15)
    lineWidth = 2
    dotSize = 12

    LOD = 100  # Number of points to plot the function
    stepSize = (xMax - xMin) / (LOD - 1)
    x = np.arange(xMin, xMax + stepSize, stepSize)
    y = f(x)
    plt.plot(x, y, '-', linewidth=lineWidth)  # Plotting the function

    ## Plotting the steps taken by gradient descent
    yGD = f(xGD)
    plt.plot(xGD, yGD, '--o', color='orange', linewidth=lineWidth, markersize=dotSize)
    plt.plot(xGD[0], yGD[0], 'o', color='blue', markersize=dotSize)
    plt.plot(xGD[-1], yGD[-1], 'o', color='red', markersize=dotSize)

    plt.xlim((xMin, xMax))
    plt.tight_layout()
    plt.show()
