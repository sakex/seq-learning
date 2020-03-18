import GPy
import numpy as np

n = 400
n_var = 2
real_c = np.full((2, 2), 1 / 8 * (3 + 2 * np.cos(2) - np.cos(4)))
design = np.random.uniform(size=n * n_var).reshape(-1, 2)
response = np.apply_along_axis(lambda x: np.sin(np.sum(x)), 1, design).reshape(-1, 1)

ker = GPy.kern.RBF(input_dim=2, ARD=True)
m = GPy.models.GPRegression(design, response, ker)

m.optimize(messages=True, max_f_eval=1000)

print(m)
print(ker.lengthscale/2)