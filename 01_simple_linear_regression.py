import matplotlib.pyplot as plt
import numpy as np

# 1次関数の周りにデータ点が分布する時に、1次関数で回帰するケース
# そりゃうまくいくよね


def generate_training_data(a0=3, a1=0.5, n=10):
    np.random.seed(3)
    x_data = 6 * np.random.rand(n) - 3
    lam_data = a1 * x_data + a0 + np.random.normal(0, 0.2, n)
    return x_data, lam_data


def show_data(x_data, lam_data):
    fig, ax = plt.subplots()
    ax.scatter(x_data, lam_data, marker='+')
    plt.xlabel('$x$')
    plt.ylabel('$\\lambda$')
    plt.show()


def linear_reg(x_data, lam_data, n):
    X_data = np.stack((np.ones(n), x_data), 1)
    c = np.linalg.pinv(X_data) @ lam_data
    return c


def show_linear_reg(c, a0, a1):
    x = np.linspace(-3, 3, 100)
    lam = a1 * x + a0
    lam_sol = c[1] * x + c[0]
    fig, ax = plt.subplots()
    ax.plot(x, lam, label='True Function $f_T$', linestyle='--')
    ax.plot(x, lam_sol, label='Function $f$ (Solution)')
    ax.scatter(x_data, lam_data, marker='+', label='Training Data', s=100)
    ax.legend()
    plt.xlabel('$x$')
    plt.ylabel('$\\lambda$')
    plt.savefig('./figures/linear_reg.png')
    plt.show()


if __name__ == '__main__':
    a0, a1, n = 3, 0.5, 10
    x_data, lam_data = generate_training_data(a0, a1, n)
    # show_data(x_data, lam_data)
    c = linear_reg(x_data, lam_data, n)
    show_linear_reg(c, a0, a1)
