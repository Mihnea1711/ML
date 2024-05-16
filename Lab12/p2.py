"""
 User\Items         W       X       Y       Z
    A               ?      4.5     2.0      ?
    B              4.0      ?      3.5      ?
    C               ?      5.0      ?      2.0
    C               ?      3.5     4.0     1.0
"""

import numpy as np

# matr de ratinguri
R = np.array([
    [np.nan, 4.5, 2.0, np.nan],
    [4.0, np.nan, 3.5, np.nan],
    [np.nan, 5.0, np.nan, 2.0],
    [np.nan, 3.5, 4.0, 1.0]
])

alpha = 0.001    # rata de inv
beta = 10e-6    # regularizare
iterations = 1000

# param algoritmului
N = R.shape[0]  # nr de useri
M = R.shape[1]  # nr de prod
K = 2           # numarul liniilor, respectiv al coloanelor din cele doua matrice factor

# init matr U și V cu valori random
np.random.seed(42)
U = np.random.rand(N, K)
V = np.random.rand(K, M)

# fct de calc al err
def calculate_error(rating_matrix, U, V):
    error = 0
    for i in range(N):
        for j in range(M):
            if not np.isnan(rating_matrix[i, j]):
                error += (rating_matrix[i, j] - np.dot(U[i, :], V[:, j])) ** 2
                for k in range(K):
                    error += (beta/2) * (U[i, k] ** 2 + V[k, j] ** 2)
    return error


# estimare ratinguri lipsa pentru userii si prod specifice
def estimate_missing_ratings(rating_matrix, R_pred):
    R_filled = rating_matrix.copy()
    for i in range(N):
        for j in range(M):
            if np.isnan(R_filled[i, j]):
                R_filled[i, j] = R_pred[i, j]
    return R_filled

def main():
    # alg de factorizare al matricii
    for it in range(iterations):
        for i in range(N):
            for j in range(M):
                if not np.isnan(R[i, j]):
                    eij = R[i, j] - np.dot(U[i, :], V[:, j])
                    for k in range(K):
                        U[i, k] = U[i, k] + alpha * (2 * eij * V[k, j] - beta * U[i, k])
                        V[k, j] = V[k, j] + alpha * (2 * eij * U[i, k] - beta * V[k, j])
        error = calculate_error(R, U, V)
        if (it + 1) % 100 == 0:
            print(f"Iteration: {it + 1}; Error: {error:.4f}")

    print()
    # estimam elem lipsa din matricea de ratinguri
    R_pred = np.dot(U, V)
    print("matr de ratinguri estimata:")
    print(R_pred)
    print()

    R_filled = estimate_missing_ratings(R, R_pred)
    print("matr de ratinguri completa:")
    print(R_filled)
    print()

if __name__ == '__main__':
    main()