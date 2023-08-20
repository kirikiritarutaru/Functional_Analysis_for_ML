import math

import GPy
import matplotlib.pyplot as plt
import numpy as np
from japanize_matplotlib import japanize

japanize()

# ガウス過程回帰による2変数関数予測


def true_func(x):
    y = np.exp(-np.sum((x - np.array([-1.2, 1.2]))**2, 1)) + \
        np.exp(-np.sum((x - np.array([1.2, -1.2]))**2, 1))
    return y


def generate_training_data(n):
    np.random.seed(1)
    x1_data = 4 * np.random.rand(40)[0:n] - 2
    x2_data = 4 * np.random.rand(40)[0:n] - 2
    x1x2_data = np.vstack([x1_data, x2_data]).T
    z_data = true_func(x1x2_data) + np.random.normal(0, 0.01, n)
    return x1_data, x2_data, x1x2_data, z_data


def kernel_func(x1, x2, i, j, hp):
    if i == j and all(x1 == x2):
        k = hp[0]**2 + hp[2]**2
    else:
        k = hp[0]**2 * math.exp(-(1 / (2 * hp[1]**2)) * np.sum((x1 - x2)**2))
    return k


def kernel_matrix(x1, x2, hyperparam):
    K = np.empty((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            K[i, j] = kernel_func(x1[i], x2[j], i, j, hyperparam)
    return K


def train_hparams(x_data, z_data):
    kernel = GPy.kern.RBF(2)
    model = GPy.models.GPRegression(x_data, z_data.reshape(-1, 1), kernel=kernel)
    hparam_priors = 3 * [None]
    hparam_priors[0] = GPy.priors.Gaussian(mu=0, sigma=1)
    hparam_priors[1] = GPy.priors.Gaussian(mu=0, sigma=1)
    hparam_priors[2] = GPy.priors.Gaussian(mu=0, sigma=0.001)
    param_name = model.parameter_names()
    for i in range(3):
        hparam_priors[i].domain = "positive"
        model[param_name[i]].set_prior(hparam_priors[i])
    model.optimize(messages=False, optimizer='scg', max_iters=1e5)
    return model


def show_training_result(model, x1_data, x2_data, x1x2_data, z_data):
    x1 = x2 = np.linspace(-2, 2, 50)
    X1, X2 = np.meshgrid(x1, x2)
    x1x2 = np.c_[np.ravel(X1), np.ravel(X2)]
    y = true_func(x1x2)

    sigma_f = np.sqrt(model.rbf.variance[0])
    q = model.rbf.lengthscale[0]
    sigma_n = np.sqrt(model.Gaussian_noise.variance[0])
    hyperparam = [sigma_f, q, sigma_n]
    K = kernel_matrix(x1x2_data, x1x2_data, hyperparam)
    invK = np.linalg.inv(K)
    K_ss = kernel_matrix(x1x2, x1x2, hyperparam)
    k_s = kernel_matrix(x1x2, x1x2_data, hyperparam)

    c = invK @ z_data
    z_mean = k_s @ c
    z_var = K_ss - k_s @ invK @ k_s.T
    z_stdv = np.sqrt(np.diag(z_var))

    z_samples = np.random.multivariate_normal(z_mean, z_var, 10)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X1, X2, z_mean.reshape(50, 50))
    ax.scatter(x1_data, x2_data, z_data, marker='+', color='black')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$\\mu(\\mathbf{x})$')
    ax.set_title('平均の描画')
    ax.view_init(azim=235)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X1, X2, z_stdv.reshape(50, 50)**2)
    ax.scatter(x1_data, x2_data, 0, marker='+', color='black')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$\\sigma^2(\\mathbf{x})$')
    ax.set_title('分散の描画')
    ax.view_init(azim=235)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for z in z_samples:
        surf = ax.plot_surface(X1, X2, z.reshape(50, 50))
    ax.scatter(x1_data, x2_data, 0, marker='+', color='black')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$\\mu(\\mathbf{x})$')
    ax.set_title('予測分布にしたがうベクトルの描画')
    ax.view_init(azim=235)

    plt.show()


if __name__ == '__main__':
    n = 25
    x1_data, x2_data, x1x2_data, z_data = generate_training_data(n)
    model = train_hparams(x1x2_data, z_data)
    show_training_result(model, x1_data, x2_data, x1x2_data, z_data)
