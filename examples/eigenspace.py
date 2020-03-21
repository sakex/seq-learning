import numpy as np
from GPy.kern import RBF, Matern32, White
from GPy.models import GPRegression
from GPy.util import diag
from GPy.util.linalg import pdinv, dpotrs, tdot

DEBUG = True  # Uncomment this line for debug purposes

if DEBUG:
    import sys
    import os

    sys.path.append(os.path.abspath('Debug'))
import sequential_learning as sl


def c_consistence():
    n = 400
    n_var = 2

    real_c = np.full((n_var, n_var), 1 / 8 * (3 + 2 * np.cos(2) - np.cos(4)))
    # design = np.random.uniform(size=n * n_var).reshape(-1, n_var)
    design = np.random.uniform(size=n * n_var).reshape((-1, n_var))

    response = np.array([np.sin(np.sum(row)) for row in design])

    kernel = Matern32(input_dim=n_var, ARD=True) + White(2, 1e-8)

    model = GPRegression(design, response.reshape(-1, 1), kernel, normalizer=True)
    # model.optimize_restarts(5, max_iters=5000, robust=True)
    model.optimize(messages=True, optimizer="lbfgsb")

    theta = kernel.rbf.lengthscale
    print(model, "\n")
    print(theta)

    # TODO: Cholesky that MF
    k = kernel.K(design)
    k_inv = np.linalg.inv(k)

    print(kernel.variance)

    mat = sl.C_gp(design, response, theta, k_inv, "Matern32")
    print(real_c)
    print()
    print(mat)
    norm = np.linalg.norm(mat - real_c)
    print(norm)


def subspace_dist(A, B):
    return np.linalg.norm(np.dot(A, A.T) - np.dot(B, B.T), ord=2)


def test_precomputed():
    from data import design, response, kinv, theta
    response, theta = map(lambda item: np.array(item), (response, theta))
    kinv = np.array(kinv).reshape(400, 400, order="F")
    design = np.array(design).reshape(-1, 2, order="F")
    true_sub = np.array((1, 1)) / np.sqrt(2)
    mat = sl.C_gp(design, response, theta, kinv, "Gaussian")
    sub_est = np.linalg.eig(mat)[1][:, 0]
    print(mat)
    assert subspace_dist(sub_est.reshape(1, 2), true_sub.reshape(1, 2)) < 1e-5

    assert dist < 1e-5


if __name__ == "__main__":
    c_consistence()
    # test_precomputed()