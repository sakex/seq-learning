#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#ifdef __DEBUG__

#include <iostream>

#endif

#include "python_to_armadillo.hpp"
#include "activegp/types/eCovTypes.h"
#include "activegp/GaussianProcess.hpp"

namespace np = boost::python::numpy;


template<activegp::eCovTypes cov_type>
inline np::ndarray c_gp(np::ndarray &design, np::ndarray &response, np::ndarray &theta, np::ndarray &k_inv) {
    activegp::GP<cov_type> gp;
    activegp::DesignLoader loader;
    loader.load_matrices(design, response, theta, k_inv);
    gp.compute(loader);
    return python_extractor::arma_to_py(gp.matrix());
}


inline constexpr unsigned hashCovType(const char *str, unsigned index = 0) {
    // Hash table computed at compilation for different branches
    return !str[index] ? 0x1505 : (hashCovType(str, index + 1) * 0x21) ^ (unsigned)str[index];
}

// We don't inline bindings
np::ndarray
select_type(np::ndarray &design, np::ndarray &response, np::ndarray &theta, np::ndarray &k_inv, char const *s) {
    unsigned const hash = hashCovType(s);
    switch (hash) {
        case hashCovType("Gaussian"):
        case hashCovType("gaussian"):
        case hashCovType("RBF"):
        case hashCovType("rbf"):
            return c_gp<activegp::eCovTypes::gaussian>(design, response, theta, k_inv);
        case hashCovType("Matern3_2"):
        case hashCovType("matern3_2"):
        case hashCovType("Matern32"):
        case hashCovType("matern32"):
            return c_gp<activegp::eCovTypes::matern_3_2>(design, response, theta, k_inv);
        case hashCovType("Matern5_2"):
        case hashCovType("matern5_2"):
        case hashCovType("Matern52"):
        case hashCovType("matern52"):
            return c_gp<activegp::eCovTypes::matern_5_2>(design, response, theta, k_inv);
        default:
            throw activegp::InvalidCovType(s);
    }
}

void translate_exception(activegp::InvalidCovType const &e) {
    // Use the Python 'C' API to set up an exception object
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

BOOST_PYTHON_MODULE (sequential_learning) {
    using namespace boost::python;
    Py_Initialize();
    np::initialize();
    register_exception_translator<activegp::InvalidCovType>(&translate_exception);
    def("C_gp", select_type);
}