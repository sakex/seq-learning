import numpy as np
import GPy

DEBUG = True  # Uncomment this line for debug purposes

if DEBUG:
    import sys
    import os

    sys.path.append(os.path.abspath('Debug'))
from sequential_learning import ASGP


def c_consistence():
    n = 200
    n_var = 2

    design = np.random.uniform(size=n * n_var).reshape((-1, n_var))
    response = np.array([np.sin(np.sum(row)) for row in design])

    real_c = np.full((n_var, n_var), 1 / 8 * (3 + 2 * np.cos(2) - np.cos(4)))

    kernel = GPy.kern.RBF(n_var, lengthscale=np.array([1.0 for _ in range(n_var)]), ARD=True,
                          name="rbf") + GPy.kern.White(n_var, variance=1)
    model = GPy.models.GPRegression(design, response.reshape(-1, 1), kernel)
    model.optimize_restarts(3)
    model.optimize(messages=True)

    sigma_f = model.kern.rbf.variance.values[0]

    K = model.kern.K(design) / sigma_f
    k_inv = np.linalg.pinv(K)
    theta = model.kern.rbf.lengthscale.values
    print(theta)

    c_ = ASGP(design, response, theta, k_inv, "RBF")
    c_.compute()
    mat = c_.mat
    print(mat)
    norm = np.linalg.norm(mat - real_c)
    print("NORM:", norm)
    assert norm < 1e-3


def c_consistence_matern32():
    n = 200
    n_var = 2

    design = np.random.uniform(size=n * n_var).reshape((-1, n_var))
    response = np.array([np.sin(np.sum(row)) for row in design])

    real_c = np.full((n_var, n_var), 1 / 8 * (3 + 2 * np.cos(2) - np.cos(4)))

    lengthscale = np.array([1.0 for _ in range(n_var)])

    kernel = GPy.kern.Matern32(n_var, lengthscale=lengthscale, ARD=True,
                               name="rbf") + GPy.kern.White(n_var, variance=1)
    model = GPy.models.GPRegression(design, response.reshape(-1, 1), kernel)
    model.optimize_restarts(3)
    model.optimize(messages=True)

    sigma_f = model.kern.rbf.variance.values[0]

    print(model.kern)
    K = model.kern.K(design) / sigma_f
    k_inv = np.linalg.pinv(K)
    theta = model.kern.rbf.lengthscale.values
    print(theta)

    c_ = ASGP(design, response, theta, k_inv, "Matern32")
    c_.compute()
    mat = c_.mat
    print(mat)
    norm = np.linalg.norm(mat - real_c)
    assert norm < 1e-3


def subspace_dist(A, B):
    return np.linalg.norm(np.dot(A, A.T) - np.dot(B, B.T), ord=2)


def test_precomputed():
    from data import design, response, kinv, theta
    response, theta = map(lambda item: np.array(item), (response, theta))
    kinv = np.array(kinv).reshape(400, 400, order="F")
    design = np.array(design).reshape(-1, 2, order="F")
    true_sub = np.array((1, 1)) / np.sqrt(2)
    c_ = ASGP(design, response, theta, kinv, "Gaussian")
    c_.compute()
    sub_est = np.linalg.eig(c_.mat)[1][:, 0]
    assert subspace_dist(sub_est.reshape(1, 2), true_sub.reshape(1, 2)) < 1e-5


if __name__ == "__main__":
    c_consistence()
    # c_consistence_matern32()
    test_precomputed()
