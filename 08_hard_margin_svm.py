import warnings

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

warnings.simplefilter("ignore")

# ハードマージンサポートベクトルマシン


def generate_training_data(N_minus=20, N_plus=20):
    np.random.seed(3)

    # 負例
    r1 = np.sqrt(0.9 * np.random.rand(N_minus))
    t1 = 2 * np.pi * np.random.rand(N_minus)
    x_minus_data = np.array([1.5 * r1 * np.cos(t1) - 2, r1 * np.sin(t1) - 1])

    # 正例
    r2 = np.sqrt(2 * 1.1 * np.random.rand(N_plus) + 1.1)
    t2 = np.arange(N_plus) * 2 * np.pi / N_plus
    x_plus_data = np.array([1.5 * r2 * np.cos(t2) - 2, r2 * np.sin(t2) - 1])

    # トータル
    x_data = np.hstack((x_plus_data, x_minus_data))

    lam_data = np.hstack((np.ones(N_plus), -np.ones(N_minus)))
    return x_data, x_plus_data, x_minus_data, lam_data


def show_data(x_plus_data, x_minus_data):
    fig, ax = plt.subplots()
    ax.scatter(x_plus_data[0], x_plus_data[1], marker='x')
    ax.scatter(x_minus_data[0], x_minus_data[1], marker='o')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()


def kernel_func(x1, x2, d):
    x1d = x2d = 0
    for m in range(d):
        x1d = np.hstack((x1d, x1**(m + 1)))
        x2d = np.hstack((x2d, x2**(m + 1)))
    return x1d @ x2d


def kernel_matrix(x1, x2, d):
    K = np.empty((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            K[i, j] = kernel_func(x1[i], x2[j], d)
    return K


def hard_margin_svm(N, x_data, lam_data):
    c = cp.Variable(N)
    v0 = cp.Variable(1)
    K = kernel_matrix(x_data.T, x_data.T, 2)
    cons = [np.diag(lam_data) @ (K @ c + v0 * np.ones(N)) >= np.ones(N)]
    Kcost = cp.Parameter(shape=K.shape, value=K, PSD=True)
    obj = cp.Minimize(cp.quad_form(c, Kcost))
    P = cp.Problem(obj, cons)
    P.solve(verbose=False)

    c = c.value
    v0 = v0.value
    cons = np.diag(lam_data) @ (K @ c + v0 * np.ones(N)) - 1
    sv_index = (np.where(np.abs(cons) < 1e-7))[0].tolist()
    sv = x_data[:, sv_index]

    return sv, c, v0


def show_training_result(x_data, x_plus_data, x_minus_data, sv, c, v0):
    x1 = np.linspace(-5, 1, 50)
    x2 = np.linspace(-3, 1, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.c_[np.ravel(X1), np.ravel(X2)]
    kx = kernel_matrix(X, x_data.T, 2)
    f = kx @ c + v0

    fig, ax = plt.subplots()
    ax.scatter(x_plus_data[0], x_plus_data[1], marker='x')
    ax.scatter(x_minus_data[0], x_minus_data[1], marker='o')
    ax.scatter(sv[0], sv[1], marker='s', color='k', fc='none', label='Support Vector')
    plt.contour(X1, X2, f.reshape(X1.shape), [0])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()


if __name__ == '__main__':
    N = 50
    x_data, x_plus_data, x_minus_data, lam_data = generate_training_data(N_minus=N, N_plus=N)
    # show_data(x_plus_data, x_minus_data)
    sv, c, v0 = hard_margin_svm(N + N, x_data, lam_data)
    show_training_result(x_data, x_plus_data, x_minus_data, sv, c, v0)
