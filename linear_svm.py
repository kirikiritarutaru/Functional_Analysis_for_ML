import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np


def generate_train_data(n=10):
    np.random.seed(0)
    x1_data = -5 + 10 * np.random.rand(n)
    x2_data = -5 + 10 * np.random.rand(n)
    lam_data = np.empty(0)
    D_plus = D_minus = np.empty(0)

    for i in range(n):
        if x1_data[i] >= 1.5 * x2_data[i]:
            D_plus = np.append(D_plus, [x1_data[i], x2_data[i]])
            lam_data = np.append(lam_data, 1)
        else:
            D_minus = np.append(D_minus, [x1_data[i], x2_data[i]])
            lam_data = np.append(lam_data, -1)
    D_plus = D_plus.reshape(-1, 2).T
    D_minus = D_minus.reshape(-1, 2).T

    return x1_data, x2_data, lam_data, D_plus, D_minus


def draw_boundary(D_plus, D_minus):
    x1 = np.linspace(-5, 5, 100)
    x2 = (1 / 1.5) * x1

    fig, ax = plt.subplots()
    ax.scatter(D_plus[0], D_plus[1], marker='x')
    ax.scatter(D_minus[0], D_minus[1], marker='o')
    ax.plot(x1, x2, ls='--', c='black')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()


def solve_quadratic(x1_data, x2_data, lam_data, n):
    c = cp.Variable(3)
    H = np.diag([2, 2, 0])
    A = np.diag(lam_data) @ np.vstack((x1_data, x2_data, np.ones(n))).T
    b = np.ones(n)
    cons = [A @ c >= b]
    obj = cp.Minimize(cp.quad_form(c, H))
    P = cp.Problem(obj, cons)
    P.solve(verbose=False)

    # サポートベクトルの算出
    c = c.value
    cons = A @ c - 1
    sv_index = (np.where(np.abs(cons) < 1e-7))[0].tolist()
    sv = np.array([x1_data[sv_index], x2_data[sv_index]])
    return c, sv


def draw_result(D_plus, D_minus, c, sv):
    x1 = np.linspace(-5, 5, 100)
    x2 = (1 / 1.5) * x1
    x2_sol = -(c[0] / c[1]) * x1 - (c[2] / c[1])

    fig, ax = plt.subplots()
    ax.plot(x1, x2, ls='--', c='black')
    ax.plot(x1, x2_sol, c='red')
    ax.scatter(D_plus[0], D_plus[1], marker='x')
    ax.scatter(D_minus[0], D_minus[1], marker='o')
    ax.scatter(sv[0], sv[1], marker='s', color='k', fc='none')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()


if __name__ == '__main__':
    n = 200
    x1_data, x2_data, lam_data, D_plus, D_minus = generate_train_data(n=n)
    # draw_boundary(D_plus, D_minus)
    c, sv = solve_quadratic(x1_data, x2_data, lam_data, n)
    draw_result(D_plus, D_minus, c, sv)
