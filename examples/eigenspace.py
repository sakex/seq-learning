import numpy as np
from GPy.kern import RBF, White, WhiteHeteroscedastic
from GPy.models import GPRegression

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

    kernel = RBF(input_dim=n_var, variance=1e-8, lengthscale=1e-8)

    model = GPRegression(design, response.reshape(-1, 1), kernel, noise_var=1e-8)
    model.optimize_restarts(5, max_iters=1000, robust=True)
    model.optimize(messages=True)

    theta = kernel.lengthscale[0]

    print(theta, "\n")
    print(model)
    # print(model.likelihood.variance)

    # TODO: Cholesky that MF
    k = kernel.K(design) + np.eye(n) * kernel.variance

    k_inv = np.linalg.inv(k)
    mat = sl.C_gp(design, response, np.array([theta for _ in range(n_var)]), k_inv, "Gaussia")
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
    mat = sl.C_gp(design, response, theta, kinv)
    sub_est = np.linalg.eig(mat)[1][:, 0]
    assert subspace_dist(sub_est.reshape(1, 2), true_sub.reshape(1, 2)) < 1e-5


def eigenspace_consistence():
    n = 400
    n_var = 2

    def func(arr: np.array):
        return np.sin(np.sum(arr))

    design = np.random.uniform(size=n * n_var).reshape((-1, n_var))

    test = np.random.uniform(size=n * n_var).reshape((-1, n_var))

    response = np.apply_along_axis(func, 1, design)

    kernel = RBF(input_dim=2, ARD=True)

    model = GPRegression(design, response, kernel)
    model.optimize(messages=True, max_f_eval=1000)

    theta = kernel.lengthscale

    k_inv = kernel.K
    mat = sl.C_gp(design, response, theta, k_inv)
    print(mat)
    true_sub = np.array((1, 1)) / np.sqrt(2)
    sub_est = np.linalg.eig(mat)[1][:, 0]
    dist = subspace_dist(sub_est.reshape(1, 2), true_sub.reshape(1, 2))
    assert dist < 1e-5


if __name__ == "__main__":
    c_consistence()
    # test_precomputed()
    # eigenspace_consistence()
