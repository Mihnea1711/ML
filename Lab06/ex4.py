import numpy as np
import matplotlib.pyplot as plt


def f4(x, y):
    return (1 - x ** 2 - y ** 3) * np.exp((-x ** 2 - y ** 2) / 2)


def finite_difference(f, x, y, h=1e-6):
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return df_dx, df_dy


f4Limits = [[-3, 3], [-3, 3]]

if __name__ == '__main__':
    f = f4
    xMin, xMax = f4Limits[0]
    yMin, yMax = f4Limits[1]
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
    nrIter = 1000000
    x_init = xMin + 0.1
    y_init = yMin + 0.1
    xGD = [x_init]
    yGD = [y_init]
    for i in range(nrIter):
        grad_x, grad_y = finite_difference(f, xGD[-1], yGD[-1])
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