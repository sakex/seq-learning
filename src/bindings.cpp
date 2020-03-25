#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "python_to_armadillo.hpp"
#include "activegp/types/eCovTypes.h"
#include "activegp/GaussianProcess.hpp"

namespace np = boost::python::numpy;
namespace python = boost::python;


template<activegp::eCovTypes cov_type>
inline np::ndarray
c_gp(np::ndarray const &design, np::ndarray const &response, np::ndarray const &theta, np::ndarray const &k_inv,
     python::object const &X2) {
    // One of Boost.Python contributors told me to pass np::ndarray by value and not by reference
    // Because they have shared pointer semantics https://github.com/boostorg/python/issues/297
    // However, https://www.mantidproject.org/Some_C%2B%2B_Tips argues that copying the shared_ptr
    // Would increment the ref counter for no good reason
    // I think of it as the Rust borrowing system
    activegp::GP<cov_type> gp;
    activegp::DesignLoader loader;
    // If we didn't provide X2, then we provided None
    if (X2.is_none()) {
        loader.load_matrices(design, response, theta, k_inv);
        gp.compute(loader);
        return python_extractor::arma_to_py(gp.matrix());
    }
    np::ndarray X2_arr = python::extract<np::ndarray>(X2);
    loader.load_matrices(design, response, theta, k_inv, X2_arr);
    gp.update(loader);
    return python_extractor::arma_to_py(gp.matrix());
}


inline constexpr unsigned hashCovType(const char *str, unsigned index = 0) {
    // Hash table computed at compilation for different branches
    return !str[index] ? 0x1505 : (hashCovType(str, index + 1) * 0x21) ^ (unsigned) str[index];
}

// We don't inline bindings
np::ndarray
select_type(np::ndarray const &design, np::ndarray const &response, np::ndarray const &theta, np::ndarray const &k_inv,
            char const *s, python::object const &X2) {
    unsigned const hash = hashCovType(s);
    switch (hash) {
        case hashCovType("Gaussian"):
        case hashCovType("gaussian"):
        case hashCovType("RBF"):
        case hashCovType("rbf"):
            return c_gp<activegp::eCovTypes::gaussian>(design, response, theta, k_inv, X2);
        case hashCovType("Matern3_2"):
        case hashCovType("matern3_2"):
        case hashCovType("Matern32"):
        case hashCovType("matern32"):
            return c_gp<activegp::eCovTypes::matern_3_2>(design, response, theta, k_inv, X2);
        case hashCovType("Matern5_2"):
        case hashCovType("matern5_2"):
        case hashCovType("Matern52"):
        case hashCovType("matern52"):
            return c_gp<activegp::eCovTypes::matern_5_2>(design, response, theta, k_inv, X2);
        default:
            throw activegp::InvalidCovType(s);
    }
}

void invalid_covtype_python(activegp::InvalidCovType const &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

BOOST_PYTHON_MODULE (sequential_learning) {
    using namespace boost::python;
    Py_Initialize();
    np::initialize();
    register_exception_translator<activegp::InvalidCovType>(&invalid_covtype_python);
    def("C_gp", &select_type,
        (python::arg("X"), python::arg("Y"),
                python::arg("theta"), python::arg("Ki"),
                python::arg("type") = "RBF", python::arg("X2") = python::object()));
}