#include <boost/python.hpp>
#include <boost/numpy.hpp>

#ifdef __DEBUG__
#include <iostream>
#endif
#include "cpp/python_loader.hpp"

namespace np = boost::numpy;

np::ndarray c_gp(np::ndarray & design, np::ndarray & response) {
#ifdef __DEBUG__
    std::cout << "hi" << std::endl;
#endif
    activegp::PythonLoader<activegp::cov_types::gaussian> builder(design, response);
}

BOOST_PYTHON_MODULE(sequential_learning)
{
    using namespace boost::python;
    Py_Initialize();
    np::initialize();
    def("C_gp", c_gp);
}