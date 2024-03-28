import numpy as np
import matplotlib.pyplot as plt


def f2(x):
    return np.sin(np.sqrt(x)) / x


def finite_difference(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


f2Limits = [1, 40]

if __name__ == '__main__':
    f = f2
    xMin, xMax = f2Limits

    ### Gradient Descent

    x = xMin + 0.1
    alpha = 0.95  # lr
    nrIter = 100000
    epsilon = 1e-6  # crit de conv

    xGD = np.array([x])

    for i in range(nrIter):
        grad_approx = finite_difference(f, x)
        x_new = x - alpha * grad_approx
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

    LOD = 1000  # Number of points to plot the function
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
