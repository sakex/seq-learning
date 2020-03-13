#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#ifdef __DEBUG__
#include <iostream>
#endif
#include "python_to_armadillo.hpp"

namespace np = boost::python::numpy;

np::ndarray c_gp(np::ndarray & design, np::ndarray & response, np::ndarray & theta) {
#ifdef __DEBUG__
    std::cout << "hi" << std::endl;
#endif
    /*activegp::python_to_armadillo builder(design, response, theta);
    activegp::GP<activegp::eCovTypes::gaussian> gp;
    gp.load(builder);*/
    return design;
}

BOOST_PYTHON_MODULE(sequential_learning)
{
    using namespace boost::python;
    Py_Initialize();
    np::initialize();
    def("C_gp", c_gp);
}