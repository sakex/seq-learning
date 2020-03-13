//
// Created by sakex on 03.03.20.
//

#ifndef SEQ_LEARNING_PYTHON_TO_ARMADILLO_HPP
#define SEQ_LEARNING_PYTHON_TO_ARMADILLO_HPP

#include "activegp/types/DesignLoader.h"
#include <armadillo>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace np = boost::python::numpy;
namespace python = boost::python;

namespace python_extractor {
    inline void py_to_arma(np::ndarray const & numpy_array, arma::Mat<double> & target) {
        Py_intptr_t const *shape = numpy_array.get_strides();
        uint16_t const rows = shape[0];
        uint16_t const columns = shape[1];

        target.set_size(rows, columns);

        for(uint16_t i = 0; i < rows; ++i){
            for(uint16_t j = 0; j < columns; ++j) {
                double const num = python::extract<double>(numpy_array(i, j));
                target(i, j) = num;
            }
        }
    }

    activegp::DesignLoader extract(np::ndarray &design, np::ndarray &response, np::ndarray &theta) {
        Py_intptr_t const *shape = design.get_strides();
        uint16_t const n = shape[0];
        uint16_t const n_var = shape[1];
        activegp::DesignLoader loader {n, n_var};
        py_to_arma(design, loader.design);
        py_to_arma(response, loader.response);
        py_to_arma(theta, loader.theta);
        return loader;
    }


}

#endif //SEQ_LEARNING_PYTHON_TO_ARMADILLO_HPP
