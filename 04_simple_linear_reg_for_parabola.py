import matplotlib.pyplot as plt
import numpy as np

# 放物線の周りにデータ点が分布する時に、1次関数で回帰するケース
# 適切なモデリングでないことが見て取れる


def generate_training_data(n=20):
    np.random.seed(3)
    x_data = 6 * np.random.rand(n) - 3
    lam_data = 0.4 * x_data**2 + 0.2 * x_data + 3 + np.random.normal(0, 0.2, n)

    return x_data, lam_data


def show_data(x_data, lam_data):
    fig, ax = plt.subplots()
    ax.scatter(x_data, lam_data, marker='+')
    plt.xlabel('$x$')
    plt.ylabel('$\\lambda$')
    plt.show()


def linear_reg(x_data, lam_data, n):
    X_data = np.stack((np.ones(n), x_data), axis=1)
    c = np.linalg.pinv(X_data) @ lam_data
    return c


def show_linear_reg(x_data, lam_data, c):
    x = np.linspace(-3, 3, 100)
    lam = 0.4 * x**2 + 0.2 * x + 3
    lam_sol = c[1] * x + c[0]

    fig, ax = plt.subplots()
    ax.plot(x, lam, ls='--', label='True Function')
    ax.plot(x, lam_sol, label='Function $f$ (Solution)')
    ax.scatter(x_data, lam_data, marker='+', label='Training Data', s=100)
    plt.xlabel('$x$')
    plt.ylabel('$\\lambda$')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    n = 20
    x_data, lam_data = generate_training_data(n=n)
    # show_data(x_data, lam_data)
    c = linear_reg(x_data, lam_data, n)
    show_linear_reg(x_data, lam_data, c)
