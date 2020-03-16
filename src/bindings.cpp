#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#ifdef __DEBUG__

#include <iostream>

#endif

#include "python_to_armadillo.hpp"
#include "activegp/types/eCovTypes.h"
#include "activegp/GaussianProcess.hpp"

namespace np = boost::python::numpy;

np::ndarray c_gp(np::ndarray &design, np::ndarray &response, np::ndarray &theta, np::ndarray &k_inv) {
    activegp::DesignLoader loader;
    loader.load_matrices(design, response, theta, k_inv);
    activegp::GP<activegp::eCovTypes::gaussian> gp;
    gp.compute(loader);
    return python_extractor::arma_to_py(gp.matrix());
}

BOOST_PYTHON_MODULE (sequential_learning) {
    using namespace boost::python;
    Py_Initialize();
    np::initialize();
    def("C_gp", c_gp);
}