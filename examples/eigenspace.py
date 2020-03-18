import numpy as np
import GPy

DEBUG = True  # Uncomment this line for debug purposes

if DEBUG:
    import sys
    import os

    sys.path.append(os.path.abspath('Debug'))
import sequential_learning as sl


def subspace_dist(A, B):
    return np.linalg.norm(np.dot(A, A.T) - np.dot(B, B.T), ord=2)


def test_precomputed():
    from data import design, response, kinv, theta
    response, theta = map(lambda item: np.array(item), (response, theta))
    kinv = np.array(kinv).reshape(400, 400)
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

    kernel = GPy.kern.RBF(input_dim=2, ARD=True)

    model = GPy.models.GPRegression(design, response, kernel)
    model.optimize(messages=True, max_f_eval=1000)

    theta = kernel.lengthscale

    k_inv = kernel.K
    mat = sl.C_gp(design, response, theta, k_inv)
    print(mat)
    true_sub = np.array((1, 1)) / np.sqrt(2)
    sub_est = np.linalg.eig(mat)[1][:, 0]
    dist = subspace_dist(sub_est.reshape(1, 2), true_sub.reshape(1, 2))
    assert dist < 1e-5


def c_consistence():
    n = 400
    n_var = 2
    real_c = np.full((2, 2), 1 / 8 * (3 + 2 * np.cos(2) - np.cos(4)))
    design = np.random.uniform(size=n * n_var).reshape(-1, 2)
    response = np.apply_along_axis(lambda x: np.sin(np.sum(x)), 1, design)

    kernel = GPy.kern.RBF(input_dim=2, ARD=True)

    model = GPy.models.GPRegression(design, response.reshape(-1, 1), kernel)
    model.optimize(messages=True, max_f_eval=1000)

    theta = kernel.lengthscale

    test = np.random.uniform(size=n * n_var).reshape(-1, 2)
    test_response = np.apply_along_axis(lambda x: np.sin(np.sum(x)), 1, test)

    k_inv = np.linalg.inv(model.kern.K(model.X))
    print(k_inv)
    mat = sl.C_gp(design, response, theta, k_inv)
    norm = np.linalg.norm(mat - real_c)
    print(mat)


if __name__ == "__main__":
    c_consistence()
    # test_precomputed()
    # eigenspace_consistence()
