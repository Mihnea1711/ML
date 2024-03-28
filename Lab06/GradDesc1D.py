import numpy as np
import matplotlib.pyplot as plt

def f1(x): # functia care trebuie minimizata
    return x**2 - 3*x

def f1Grad(x): # derivata
    return 2*x-3

f1Limits = [-1, 4] # domeniul functiei
    
if __name__ == '__main__':
    
    f = f1
    fGrad = f1Grad
    xMin, xMax = f1Limits
    
    ### gradientul descendent

    x = xMin+0.1 # valoarea initiala a lui x
    alpha = 0.3 # rata de invatare 
    nrIter = 10
    epsilon = 1e-6  # crit de conv

    xGD = np.array([x]) # valorile obtinute pe parcursul iteratiilor
    
    for i in range(nrIter):
        x_new = x - alpha * fGrad(x)
        xGD = np.append(xGD, x_new)
        # print('Iter {0}: x = {1} , f(x) = {2}'.format(i+1, x, f(x)))
        # ... aici ar trebui adaugat un criteriu de convergenta
        if abs(f(x_new) - f(x)) < epsilon:
            print("Converged at iteration", i + 1)
            break
        x = x_new
    
    yGD = f(xGD)

    ### afisare grafica
    
    plt.rc('font', size=15)
    lineWidth = 2
    dotSize = 12

    LOD = 30 # nr de puncte prin care se traseaza graficul functiei
    stepSize = (xMax - xMin) / (LOD-1)
    x = np.arange(xMin, xMax + stepSize, stepSize)
    y = f(x)
    plt.plot(x, y, '-', linewidth = lineWidth) # afisarea graficului functiei
    
    ## afisarea valorilor obtinute cu gradientul descendent
    gradx = fGrad(x)
    plt.plot(xGD, yGD, '--o', color='orange', linewidth = lineWidth, markersize=dotSize)
    # primul si ultimul punct sunt evidentiate cu alte culori
    plt.plot(xGD[0], yGD[0], 'o', color='blue', markersize = dotSize)
    plt.plot(xGD[-1], yGD[-1], 'o', color='red', markersize = dotSize)

    plt.xlim((xMin, xMax))
    plt.tight_layout()
    plt.show()
    




