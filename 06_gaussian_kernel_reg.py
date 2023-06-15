import math

import matplotlib.pyplot as plt
import numpy as np

# ガウスカーネル回帰


def generate_training_data(n=4, add_noise=False):
    np.random.seed(1)
    x_data = 6 * np.random.rand(n) - 3
    if add_noise:
        lam_data = 1 - 1.5 * x_data + np.sin(x_data) + \
            np.cos(3 * x_data) + np.random.normal(0, 0.1, n)
    else:
        lam_data = 1 - 1.5 * x_data + np.sin(x_data) + np.cos(3 * x_data)
    x = np.linspace(-3, 3, 100)
    lam = 1 - 1.5 * x + np.sin(x) + np.cos(3 * x)

    return x_data, lam_data, x, lam


def kernel_func(x1, x2):
    gamma = 1 / 2
    k = math.exp(-gamma * np.sum((x1 - x2)**2))
    return k


def kernel_matrix(x1, x2):
    K = np.empty((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            K[i, j] = kernel_func(x1[i], x2[j])
    return K


def gaussian_kernel_reg(x_data, lam_data, x, lam):
    K = kernel_matrix(x_data, x_data)
    c = np.linalg.inv(K) @ lam_data

    k_s = kernel_matrix(x, x_data)
    lam_sol = k_s @ c

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, lam, ls='--', label='True Function $f_T$')
    ax.plot(x, lam_sol, label='Function $f$ (Solution)')
    ax.scatter(x_data, lam_data, label='Training Data', marker='+', s=100, c='black')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-5.5, 5.5)
    ax.legend(loc='lower left')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.show()


if __name__ == '__main__':
    n = 8
    add_noise = True
    x_data, lam_data, x, lam = generate_training_data(n, add_noise)
    gaussian_kernel_reg(x_data, lam_data, x, lam)
