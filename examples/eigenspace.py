import numpy as np
import GPy

DEBUG = True  # Uncomment this line for debug purposes

if DEBUG:
    import sys
    import os

    sys.path.append(os.path.abspath('Debug'))
from sequential_learning import ASGP


def c_consistence():
    n = 800
    n_var = 10

    design = np.random.uniform(size=n * n_var).reshape((-1, n_var))
    new_design = np.random.uniform(size=(n * n_var // 2)).reshape((-1, n_var))

    response = np.array([np.sin(np.sum(row)) for row in design])

    real_c = np.full((n_var, n_var), 1 / 8 * (3 + 2 * np.cos(2) - np.cos(4)))

    kernel = GPy.kern.Bias(n_var) * GPy.kern.RBF(n_var, lengthscale=np.array([1.0 for _ in range(n_var)]), ARD=True,
                                                 name="rbf") + GPy.kern.White(n_var, variance=1)
    model = GPy.models.GPRegression(design, response.reshape(-1, 1), kernel)
    model.optimize_restarts(2)
    model.optimize(messages=True)

    sigma_f = model.kern.mul.bias.variance.values[0] * model.kern.mul.rbf.variance.values[0]

    K = model.kern.K(design) / sigma_f
    k_inv = np.linalg.pinv(K)
    theta = model.kern.mul.rbf.lengthscale.values

    c_ = ASGP(design, response, theta, k_inv, "RBF")
    c_.compute()
    mat = c_.mat
    print(mat)
    norm = np.linalg.norm(mat - real_c)
    assert norm < 1e-3


def test_precomputed():
    from data import design, response, kinv, theta
    response, theta = map(lambda item: np.array(item), (response, theta))
    kinv = np.array(kinv).reshape(400, 400, order="F")
    design = np.array(design).reshape(-1, 2, order="F")
    true_sub = np.array((1, 1)) / np.sqrt(2)
    mat = sl.C_gp(design, response, theta, kinv, "Gaussian")
    sub_est = np.linalg.eig(mat)[1][:, 0]
    assert subspace_dist(sub_est.reshape(1, 2), true_sub.reshape(1, 2)) < 1e-5


if __name__ == "__main__":
    c_consistence()
    # test_precomputed()
