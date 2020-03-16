import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial.distance import cdist

DEBUG = True  # Uncomment this line for debug purposes

if DEBUG:
    import sys
    import os

    sys.path.append(os.path.abspath('Debug'))
import sequential_learning as sl


def subspace_dist(A, B):
    return np.linalg.norm(np.dot(A, A.T) - np.dot(B, B.T), ord=2)


def test_precomputed():
    from data import design, response, k_inv, theta
    design, response, k_inv, theta = map(lambda item: np.array(item), (design, response, k_inv, theta))
    true_sub = np.array((1, 1)) / np.sqrt(2)
    mat = sl.C_gp(design, response, theta, k_inv)
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

    kernel = RBF(length_scale=(1, 1))
    # The only way to use this is to look at their code, no explaination anywhere in the docs
    # Spent hours on this

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                   alpha=.2,
                                   optimizer="fmin_l_bfgs_b").fit(design, response)
    gpr.predict(test, return_std=True)
    # Here again, I need to run predict to have _K_inv
    # How do I know ? sklearn/gaussian_process/_gpr.py lines 344-351
    theta = gpr.kernel_.get_params()["length_scale"]
    k_inv = gpr._K_inv
    mat = sl.C_gp(design, response, theta, k_inv)
    true_sub = np.array((1, 1)) / np.sqrt(2)
    sub_est = np.linalg.eig(mat)[1][:, 0]
    dist = subspace_dist(sub_est.reshape(1, 2), true_sub.reshape(1, 2))
    print(dist)
    assert dist < 1e-5


if __name__ == "__main__":
    test_precomputed()
    eigenspace_consistence()
