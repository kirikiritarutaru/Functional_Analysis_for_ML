import math

import matplotlib.pyplot as plt
import numpy as np

# リッジ回帰
# 学習すべきパラメータの大きさに制約を加える「正則化」を施した回帰


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


def ridge_reg(x_data, lam_data, x, lam):
    K = kernel_matrix(x_data, x_data)

    # パラメータへの制約が小さい場合
    alpha = 5e-7
    c_1 = np.linalg.inv(K.T @ K + alpha * np.identity(n)) @ K.T @ lam_data

    # パラメータへの制約が大きい場合
    alpha = 1
    c_2 = np.linalg.inv(K.T @ K + alpha * np.identity(n)) @ K.T @ lam_data

    k_s = kernel_matrix(x, x_data)
    lam_sol_1 = k_s @ c_1
    lam_sol_2 = k_s @ c_2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, lam, ls='--', label='True Function $f_T$', color='red', alpha=0.6)
    ax.plot(x, lam_sol_1, label=r'Function $f$ ($\alpha=5e-7$)')
    ax.plot(x, lam_sol_2, label=r'Function $f$ ($\alpha=1.0$)')
    ax.scatter(x_data, lam_data, label='Training Data', marker='+', s=100, c='black')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-5.5, 5.5)
    ax.legend(loc='lower left')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.show()


if __name__ == '__main__':
    n = 5
    add_noise = True
    x_data, lam_data, x, lam = generate_training_data(n, add_noise)
    ridge_reg(x_data, lam_data, x, lam)
