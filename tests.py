import unittest
import Debug.sequential_learning as sl
import numpy as np
import GPy


class ImportTest(unittest.TestCase):
    def import_test(self):
        text = "Hello World"
        output = sl.test_import(text)
        self.assertEqual(output, "Input: Hello World")


class ActiveSubspaceGPEstimationTest(unittest.TestCase):
    @staticmethod
    def subspace_dist(A, B):
        return np.linalg.norm(np.dot(A, A.T) - np.dot(B, B.T), ord=2)

    def test_c_eigen(self):
        kernels = [
            (GPy.kern.RBF, "RBF"),
            (GPy.kern.Matern32, "Matern32"),
            (GPy.kern.Matern52, "Matern52")
        ]
        for kern, name in kernels:
            # ==============================
            # GENEREATE DATA
            n = 200  # Number of samples
            n_var = 2  # Number of variables (dimensions)
            design = np.random.uniform(size=n * n_var).reshape((-1, n_var))  # Creates the random data

            # Define the function
            def f(x):
                return np.sin(np.sum(x))

            response = np.fromiter(map(f, design), dtype=float)  # Applies the random data to the design
            # and creates the response

            # ==============================
            # RUN REGRESSION (RBF KERNEL)
            kernel = kern(2, lengthscale=[1., 1.], ARD=True, name="rbf") + GPy.kern.White(2, variance=1.)
            model = GPy.models.GPRegression(design, response.reshape(-1, 1), kernel)
            model.optimize(messages=False)  # Run the regression

            # ==============================
            # EXTRACT MATRIX K⁻¹ AND LENGTH SCALE PARAMETERS
            variance = model.kern.rbf.variance.values[0]  # Extract variance
            K = model.kern.K(design) / variance  # Maths explained in the paper
            k_inv = np.linalg.pinv(K)
            lengthscales = model.kern.rbf.lengthscale.values

            # ==============================
            # RUN THE C++ ALGORITHM
            cpp_lib = sl.ASGP(design, response, lengthscales, k_inv, name)  # Constructor
            cpp_lib.compute()  # Compute
            c_hat = cpp_lib.mat  # Get matrix

            # ==============================
            # TEST THAT OUR MATRIX HAS THE RIGHT EIGENVALUES
            true_subspace = np.array((1, 1)) / np.sqrt(2)  # True subspace
            sub_est = np.linalg.eig(c_hat)[1][:, 0]  # Computes Eigenvalues
            distance = self.subspace_dist(sub_est.reshape(1, 2), true_subspace.reshape(1, 2))  # Get subspace distance
            self.assertTrue(distance < 1e-5)

    def test_c_estimation(self):
        # ==============================
        # GENEREATE DATA
        n = 400  # Number of samples
        n_var = 2  # Number of variables (dimensions)
        design = np.random.uniform(size=n * n_var).reshape((-1, n_var))  # Creates the random data

        # Define the function
        def f(x):
            return np.sin(np.sum(x))

        response = np.fromiter(map(f, design), dtype=float)  # Applies the random data to the design
        # and creates the response

        # ==============================
        # RUN REGRESSION (RBF KERNEL)
        kernel = GPy.kern.RBF(2, lengthscale=[1., 1.], ARD=True, name="rbf") + GPy.kern.White(2, variance=1.)
        model = GPy.models.GPRegression(design, response.reshape(-1, 1), kernel)
        model.optimize(messages=False)  # Run the regression

        # ==============================
        # EXTRACT MATRIX K⁻¹ AND LENGTH SCALE PARAMETERS
        variance = model.kern.rbf.variance.values[0]  # Extract variance
        K = model.kern.K(design) / variance  # Maths explained in the paper
        k_inv = np.linalg.pinv(K)
        lengthscales = model.kern.rbf.lengthscale.values

        # ==============================
        # RUN THE C++ ALGORITHM
        cpp_lib = sl.ASGP(design, response, lengthscales, k_inv, "RBF")  # Constructor
        cpp_lib.compute()  # Compute
        c_hat = cpp_lib.mat  # Get matrix

        # ==============================
        # TEST THAT OUR MATRIX HAS THE RIGHT EIGENVALUES
        true_c = np.full((n_var, n_var), 1. / 8. * (3. + 2. * np.cos(2.) - np.cos(4.)))  # True C matrix
        distance = np.linalg.norm(c_hat - true_c)  # Get distance
        self.assertTrue(distance < 1e-3)


if __name__ == '__main__':
    unittest.main()
