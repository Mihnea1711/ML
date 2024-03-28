import numpy as np
import matplotlib.pyplot as plt


def f3(x, y):
    return x ** 4 + 2 * x ** 2 * y - 21 * x ** 2 + 2 * x * y ** 2 - 14 * x + y ** 4 - 16 * y ** 2 - 22 * y + 170


def f3Grad(x, y):
    df_dx = 4 * x ** 3 + 4 * x * y - 42 * x + 2 * y ** 2 - 14
    df_dy = 2 * x ** 2 + 4 * x * y + 4 * y ** 3 - 32 * y - 22
    return df_dx, df_dy


f3Limits = [[-4, 4], [-4, 4]]

if __name__ == '__main__':
    f = f3
    xMin, xMax = f3Limits[0]
    yMin, yMax = f3Limits[1]
    epsilon = 1e-9  # crit de conv

    plt.rc('font', size=15)
    lineWidth = 2
    dotSize = 12

    fstep = 0.2  # pasul folosit pentru afisare (valori mai mici = rezolutie mai mare)
    x = np.arange(xMin, xMax, fstep)
    y = np.arange(yMin, yMax, fstep)
    x, y = np.meshgrid(x, y)
    z = f(x, y)

    ## afisare sub forma unei imagini 2D
    plt.imshow(np.flip(z, 0), cmap=plt.get_cmap('gray'), extent=(xMin, xMax, yMin, yMax))

    alpha = 0.001  # lr
    nrIter = 100000
    x_init = xMin + 0.1
    y_init = yMin + 0.1
    xGD = [x_init]
    yGD = [y_init]
    for i in range(nrIter):
        grad_x, grad_y = f3Grad(xGD[-1], yGD[-1])
        x_new = xGD[-1] - alpha * grad_x
        y_new = yGD[-1] - alpha * grad_y
        xGD.append(x_new)
        yGD.append(y_new)
        if abs(f(x_new, y_new) - f(xGD[-2], yGD[-2])) < epsilon:
            print("Converged at iteration", i + 1)
            break

    print('Minimum found at (x, y) =', (xGD[-1], yGD[-1]), ', f(x, y) =', f(xGD[-1], yGD[-1]))

    # afisarea valorilor gradientului descendent, presupunand ca acestea se retin in xGD, yGD
    plt.plot(xGD, yGD, '--o', color='orange', linewidth=lineWidth, markersize=dotSize)
    plt.plot(xGD[0], yGD[0], 'o', color='blue', markersize=dotSize)
    plt.plot(xGD[-1], yGD[-1], 'o', color='red', markersize=dotSize)

    plt.xlim((xMin, xMax))
    plt.ylim((yMin, yMax))
    plt.tight_layout()

    ## convert xGD and yGD to numpy arrays
    xGD = np.array(xGD)
    yGD = np.array(yGD)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z,
                    rstride=1, cstride=1,
                    cmap='viridis',
                    linewidth=0, antialiased=True,
                    zorder=0)

    ## afisarea valorilor gradientului descendent, presupunand ca acestea se retin in xGD, yGD
    zGD = f(xGD, yGD)
    ax.plot(xGD, yGD, zGD, '--o', color='orange', zorder=10, linewidth=lineWidth, markersize=dotSize)
    ax.plot([xGD[0]], [yGD[0]], [zGD[0]], 'o', color='blue', zorder=10, markersize=dotSize)
    ax.plot([xGD[-1]], [yGD[-1]], [zGD[-1]], 'o', color='red', zorder=10, markersize=dotSize)

    # Plotting the steps taken by gradient descent
    ax.scatter(xGD, yGD, f(np.array(xGD), np.array(yGD)), color='red')

    ax.view_init(30, 40)  # stabilirea unghiurilor din care se priveste scena 3D (in grade)
    plt.tight_layout()
    plt.show()
