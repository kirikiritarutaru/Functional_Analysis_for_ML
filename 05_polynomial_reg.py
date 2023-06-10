import matplotlib.pyplot as plt
import numpy as np

# 多項式回帰の例


def generate_training_data(n=4, d=2):
    np.random.seed(1)
    x_data = 6 * np.random.rand(n) - 3
    lam_data = 1 - 1.5 * x_data + np.sin(x_data) + np.cos(3 * x_data)
    x = np.linspace(-3, 3, 100)
    lam = 1 - 1.5 * x + np.sin(x) + np.cos(3 * x)

    return x_data, lam_data, x, lam


def Phi(x, d):
    Phi = 1
    for m in range(d):
        Phi = np.hstack((Phi, x**(m + 1)))
    return Phi


def Phi_matrix(x1, x2, d):
    K = np.empty((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            K[i, j] = Phi(x1[i], d).T @ Phi(x2[j], d)
    return K


def polynomial_reg(x_data, lam_data, d):
    K = Phi_matrix(x_data, x_data, d)
    c = np.linalg.pinv(K) @ lam_data
    lam_sol = Phi_matrix(x, x_data.T, d) @ c

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, lam, ls='--', label='True Function $f_T$')
    ax.plot(x, lam_sol, label='Function $f$ (Solution)')
    ax.scatter(x_data, lam_data, marker='+', label='Training Data', s=100, c='black')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-4, 4)
    ax.legend(loc='lower left')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.show()


if __name__ == '__main__':
    n = 15
    d = 10
    x_data, lam_data, x, lam = generate_training_data(n, d)
    polynomial_reg(x_data, lam_data, d)
