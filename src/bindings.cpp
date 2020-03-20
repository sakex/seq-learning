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


constexpr unsigned hashCovType(const char *str, unsigned index = 0) {
    return !str[index] ? 0x1505 : (hashCovType(str, index + 1) * (unsigned)0x21) ^ str[index];
}

np::ndarray
select_type(np::ndarray &design, np::ndarray &response, np::ndarray &theta, np::ndarray &k_inv, char const *s) {
    unsigned hash = hashCovType(s);
    switch (hash) {
        case hashCovType("Gaussian"):
            return c_gp<activegp::eCovTypes::gaussian>(design, response, theta, k_inv);
        case hashCovType("Mattern3_2"):
            return c_gp<activegp::eCovTypes::mattern_3_2>(design, response, theta, k_inv);
        default:
            throw activegp::InvalidCovType(std::string(s));
    }
}

void translate_exception(activegp::InvalidCovType const& e)
{
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