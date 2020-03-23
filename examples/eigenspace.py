import numpy as np
from GPy.kern import Matern52, RBF
from GPy.models import GPRegression
from sklearn.metrics.pairwise import euclidean_distances
from functools import reduce

DEBUG = True  # Uncomment this line for debug purposes

if DEBUG:
    import sys
    import os

    sys.path.append(os.path.abspath('Debug'))
import sequential_learning as sl


def Sq_Euclid_DistMat(X1, X2):
    """
    L2-norm applicable for both vectors and matrices (useful for high dimension input features.)
    Parameter Description:
    X1,X2: When X1==X2, then calculate autocorrelation. Otherwise cross-correlations.
    DistMat: Pairwise Squared Distance Matrix of size nXn
    """
    if X1.shape[1] == 1:
        n = X1.shape[0]
        m = X2.shape[0]
        r1 = X1.reshape(n, 1) * np.ones([1, m])
        r2 = X2.reshape(1, m) * np.ones([n, 1])
        sed = ((r1 - r2) ** 2)
    elif X1.shape[1] == 2:  # matrices for 2D feature space.
        n = X1.shape[0]
        m = X2.shape[0]
        r1x = X1[:, 0].reshape(n, 1) * np.ones([1, m])
        r1y = X1[:, 1].reshape(n, 1) * np.ones([1, m])
        r2x = X2[:, 0].reshape(1, m) * np.ones([n, 1])
        r2y = X2[:, 1].reshape(1, m) * np.ones([n, 1])
        sed = ((r1x - r2x) ** 2 + (r1y - r2y) ** 2)
    else:
        print("too many dimensions in X matrices", X1.shape)
        return None

    return sed


def c_consistence():
    from data import design, response, kinv, theta
    kinv = np.array(kinv).reshape(400, 400, order="F")
    response, theta = map(lambda item: np.array(item), (response, theta))
    design = np.array(design).reshape(-1, 2, order="F")
    n = 400
    n_var = 2

    real_c = np.full((n_var, n_var), 1 / 8 * (3 + 2 * np.cos(2) - np.cos(4)))

    """design = np.random.uniform(size=n * n_var).reshape((-1, n_var))

    response = np.array([np.sin(np.sum(row)) for row in design])"""

    kernel = Matern52(input_dim=n_var, name="main", ARD=True)

    model = GPRegression(design, response.reshape(-1, 1), kernel, noise_var=1e-8)

    model.optimize(messages=True, optimizer="lbfgsb")

    print(model)

    theta = kernel.lengthscale

    r = euclidean_distances(design / kernel.lengthscale)

    """K = np.exp(-dist / theta) + np.diag(model.likelihood.variance ** 2 * np.ones(n))
    c = np.linalg.inv(np.linalg.cholesky(K))
    k_inv = np.dot(c.T, c)
    print(K, "\n")"""

    K = reduce(lambda reducer, t: (1+np.sqrt(5.)/t*abs(r)+5./(3*t**2*r**2))*np.exp(-np.sqrt(5.)*r/t) * reducer, theta, np.full((n, n), 1))
    print(K)
    k_inv = np.linalg.inv(K)

    print(K)
    print(k_inv)

    mat = sl.C_gp(design, response, theta, k_inv, "Matern52")
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


if __name__ == "__main__":
    c_consistence()
    # test_precomputed()
