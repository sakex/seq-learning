import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt


DEBUG = True  # Uncomment this line for debug purposes

if DEBUG:
    import sys
    import os

    sys.path.append(os.path.abspath('Debug'))
from sequential_learning import ASGP


def c_consistence_rbf():
    n = 200
    n_var = 2

    design = np.random.uniform(size=n * n_var).reshape((-1, n_var))
    response = np.array([np.sin(np.sum(row)) for row in design])

    real_c = np.full((n_var, n_var), 1 / 8 * (3 + 2 * np.cos(2) - np.cos(4)))

    with pm.Model() as marginal_gp_model:
        ls = pm.Gamma('ℓ', alpha=1, beta=10, shape=design.shape[1])
        cov_func = pm.gp.cov.ExpQuad(input_dim=2, ls=ls)
        gp = pm.gp.Marginal(cov_func=cov_func)

        noise = pm.gp.cov.WhiteNoise(sigma=0.2)
        f = gp.marginal_likelihood("f", X=design, y=response, noise=noise)
        k = pm.Deterministic("k_inv", tt.nlinalg.matrix_inverse(cov_func(design) + np.eye(n) * (np.finfo(float).eps + 1.490116e-08)))
        mp = pm.find_MAP()
        k_inv = mp["k_inv"]
        theta = np.array(mp["ℓ"])
        print("theta:", theta)
        c_ = ASGP(design, response, theta, k_inv, "RBF")
        c_.compute()
        mat = c_.mat
        print(mat)
        norm = np.linalg.norm(mat - real_c)
        assert norm < 5e-3
        print("NORM:", norm)


if __name__ == "__main__":
    c_consistence_rbf()
