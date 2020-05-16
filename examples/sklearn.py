import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, Matern
import scipy
from scipy.spatial.distance import pdist, squareform

DEBUG = True  # Uncomment this line for debug purposes

if DEBUG:
    import sys
    import os

    sys.path.append(os.path.abspath('Debug'))
from sequential_learning import ASGP


def _check_length_scale(X, length_scale):
    length_scale = np.squeeze(length_scale).astype(float)
    if np.ndim(length_scale) > 1:
        raise ValueError("length_scale cannot be of dimension greater than 1")
    if np.ndim(length_scale) == 1 and X.shape[1] != length_scale.shape[0]:
        raise ValueError("Anisotropic kernel must have the same number of "
                         "dimensions as data (%d!=%d)"
                         % (length_scale.shape[0], X.shape[1]))
    return length_scale


def c_consistence(n_var, kernel, name):
    n = 400

    design = np.random.uniform(size=n * n_var).reshape((-1, n_var))
    response = np.array([np.sin(np.sum(row)) for row in design])

    real_c = np.full((n_var, n_var), 1 / 8 * (3 + 2 * np.cos(2) - np.cos(4)))

    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10)
    gpr.fit(design, response)

    L_inv = scipy.linalg.solve_triangular(gpr.L_.T, np.eye(gpr.L_.shape[0]))
    k_inv = L_inv.dot(L_inv.T)
    sigma_f = gpr.kernel_.k1.get_params()['k1__constant_value']

    theta = gpr.kernel_.k1.get_params()['k2__length_scale']

    Ki = k_inv * sigma_f

    c_ = ASGP(design, response, theta, Ki, name)
    c_.compute()
    mat = c_.mat
    print(mat)
    norm = np.linalg.norm(mat - real_c)
    # assert norm < 1e-3


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
    def main():
        n_var = 2
        rbf = ConstantKernel(1e-8) * RBF(length_scale=np.array([1.0, 1.0])) + WhiteKernel(noise_level=1)
        matern32 = ConstantKernel(1e-8) * Matern(length_scale=np.array([1.0, 1.0]), nu=1.5) + WhiteKernel(noise_level=.5)
        matern52 = ConstantKernel(1e-8) * Matern(length_scale=np.array([1.0, 1.0]), nu=2.5) + WhiteKernel(noise_level=1)
        kerns = [rbf, matern32, matern52]
        names = ("RBF", "Matern32", "Matern52")
        for kern, name in zip(kerns, names):
            c_consistence(n_var, kern, name)


    main()
    # test_precomputed()
