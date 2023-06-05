import matplotlib.pyplot as plt
import numpy as np


def generate_training_data(a0=2, a1=0.3, a2=-0.5, n=20):
    np.random.seed(1)
    x_data = 6 * np.random.rand(n, 2) - 3
    lam_data = a2 * x_data[:, 1] + a1 * x_data[:, 0] + a0 + np.random.normal(0, 0.5, n)
    return x_data, lam_data


def show_data(x_data, lam_data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(x_data[:, 0], x_data[:, 1], lam_data, marker='+')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$\\lambda$')
    ax.view_init(azim=235)
    plt.show()


def multiple_reg(x_data, lam_data, n):
    X_data = np.stack((np.ones(n), x_data[:, 0], x_data[:, 1]), 1)
    c = np.linalg.pinv(X_data) @ lam_data
    return c


def show_multi_reg(c, a0, a1):
    x1 = x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.c_[np.ravel(X1), np.ravel(X2)]

    Lam = a2 * X[:, 1] + a1 * X[:, 0] + a0
    lam = Lam.reshape(X1.shape)
    Lam_sol = c[2] * X[:, 1] + c[1] * X[:, 0] + c[0]
    lam_sol = Lam_sol.reshape(X1.shape)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot_surface(X1, X2, lam, alpha=0.5, label='True Function $f_T$')
    ax.plot_surface(X1, X2, lam_sol, label='Function $f$ (Solution)')
    ax.scatter(x_data[:, 0], x_data[:, 1], lam_data, marker='+', s=100, label='Training Data')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$\\lambda$')
    ax.view_init(azim=235)
    plt.savefig('./figures/multiple_reg.png')
    plt.show()


if __name__ == '__main__':
    a0, a1, a2, n = 3, 0.5, -0.5, 20
    x_data, lam_data = generate_training_data(a0, a1, a2, n)
    # show_data(x_data, lam_data)
    c = multiple_reg(x_data, lam_data, n)
    show_multi_reg(c, a0, a1)
