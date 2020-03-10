//
// Created by sakex on 03.03.20.
//

#ifndef SEQ_LEARNING_PYTHON_LOADER_HPP
#define SEQ_LEARNING_PYTHON_LOADER_HPP

#include "gaussian_process.hpp"
#include "cov_types.hpp"
#include <boost/python/numpy.hpp>

namespace activegp {
    namespace np = boost::python::numpy;

    struct PythonLoader {
        uint16_t n;
        uint16_t n_var;
        np::ndarray const &design;
        np::ndarray const &response;
        np::ndarray const &theta;

        PythonLoader(np::ndarray const &_design, np::ndarray const &_response, np::ndarray const &_theta) :
                design(_design), response(_response), theta(_theta) {
            Py_intptr_t const *shape = design.get_strides();
            n = shape[0];
            n_var = shape[1];
            std::cout << n << " " << n_var << std::endl;
        }
    };
}

#endif //SEQ_LEARNING_PYTHON_LOADER_HPP
