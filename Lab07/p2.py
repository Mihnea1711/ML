import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

X = np.array([0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5])
y = np.array([4, 3, 2.5, 1, 2, 3.5, 6, 4, 7, 1.5, 5, 2.5, 5.5, 3, 8, 7, 7.5, 6, 8.5, 9.5])

# fct pt regr polinomiala
def polynomial_regression(X, y, degree):
    # polinom de grad degree
    coeffs = np.polyfit(X, y, degree)

    # model polinomial
    poly_model = np.poly1d(coeffs)

    # calc val
    y_pred = poly_model(X)

    # calc err medie patr (MSE)
    mse = mean_squared_error(y, y_pred)

    return poly_model, mse

# grad maxim (ipoteza)
max_degree = 8
mse_list = []
poly_models = []

# verificam fiecare grad si cream polinoamele
for degree in range(2, max_degree + 1):
    model, mse = polynomial_regression(X, y, degree)
    mse_list.append(mse)
    poly_models.append(model)

# afisam MSE pt fiec polinom
for degree, mse in enumerate(mse_list, start=2):
    print(f"Gradul {degree}: MSE = {mse}")

# afis polinomul cu cea mai mica err
best_degree = np.argmin(mse_list) + 2
best_model = poly_models[np.argmin(mse_list)]
print(f"Cel mai bun grad polinomial: {best_degree}")

# plotam polinomul cu cea mai mica err
plt.scatter(X, y, color='blue', label='Date de antrenare')
plt.plot(X, best_model(X), color='red', label=f'Regr polinomiala (grad {best_degree})')
plt.xlabel('Nr ore de studiu')
plt.ylabel('Nota obtinuta')
plt.title('Regr polinomiala')
plt.legend()
plt.grid(True)
plt.savefig('regr_polinomial.png')
plt.show()
