import numpy as np
import matplotlib.pyplot as plt

def f1(x, y):
    return (1 - (x**2 + y**3)) * np.exp(-(x**2 + y**2) / 2)

def gradient_f1(x, y):
    df_dx = (-2 * x * (1 - (x**2 + y**3)) + 3 * x**2 * (x**2 + y**3)) * f1(x, y)
    df_dy = (-3 * y**2 * (1 - (x**2 + y**3)) + 2 * y * (x**2 + y**3)) * f1(x, y)
    return df_dx, df_dy

f1Limits = [[-3, 3],[-3, 3]]

if __name__ == '__main__':
    f = f1
    xMin, xMax = f1Limits[0]
    yMin, yMax = f1Limits[1]
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

    # Gradient Descent
    alpha = 0.01
    nrIter = 10000
    xGD = [xMin + 0.1]
    yGD = [yMin + 0.1]
    for i in range(nrIter):
        grad_x, grad_y = gradient_f1(xGD[-1], yGD[-1])
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

    ### afisare sub forma unei suprafete intr-o scena 3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(x, y, z,
                    rstride=1, cstride=1,
                    cmap=plt.get_cmap('gray'),
                    linewidth=0, antialiased=True,
                    zorder=0)

    ## afisarea valorilor gradientului descendent, presupunand ca acestea se retin in xGD, yGD
    zGD = f(xGD, yGD)
    ax.plot(xGD, yGD, zGD, '--o', color='orange', zorder=10, linewidth=lineWidth, markersize=dotSize)
    ax.plot([xGD[0]], [yGD[0]], [zGD[0]], 'o', color='blue', zorder=10, markersize=dotSize)
    ax.plot([xGD[-1]], [yGD[-1]], [zGD[-1]], 'o', color='red', zorder=10, markersize=dotSize)

    ax.view_init(30, 40)  # stabilirea unghiurilor din care se priveste scena 3D (in grade)

    ## rotirea automata a scenei 3D
    ## (posibil doar daca matplotlib genereaza o fereastra separata)
    ## (nu functioneaza pentru ferestre 'inline')

    # for angle in range(0, 360):
    #    ax.view_init(30, angle)
    #    plt.draw()
    #    plt.pause(0.01)
    #
    plt.tight_layout()
    plt.show()
