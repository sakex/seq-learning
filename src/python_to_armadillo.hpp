//
// Created by sakex on 03.03.20.
//

#ifndef SEQ_LEARNING_PYTHON_TO_ARMADILLO_HPP
#define SEQ_LEARNING_PYTHON_TO_ARMADILLO_HPP

#include "activegp/types/DesignLoader.hpp"
#include <armadillo>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace np = boost::python::numpy;
namespace python = boost::python;

namespace python_extractor {
    inline void py_to_arma(np::ndarray const &numpy_array, arma::Mat<double> &target) {
        uint16_t const nd = numpy_array.get_nd();
        assert(nd == 2); // Has to be 2 dimensional
        Py_intptr_t const *shape = numpy_array.get_shape();
        uint16_t const rows = shape[0];
        uint16_t const columns = shape[1];

        target.set_size(rows, columns);
        for (uint16_t i = 0; i < rows; ++i) {
            for (uint16_t j = 0; j < columns; ++j) {
                double const num = python::extract<double>(numpy_array[i][j]);
                target(i, j) = num;
            }
        }
    }

    inline void py_to_arma(np::ndarray const &numpy_array, arma::vec &target) {
        uint16_t const nd = numpy_array.get_nd();
        assert(nd == 1);
        Py_intptr_t const *shape = numpy_array.get_shape();
        uint16_t const rows = shape[0];

        target.set_size(rows);
        for (uint16_t i = 0; i < rows; ++i) {
            double const num = python::extract<double>(numpy_array[i]);
            target(i) = num;
        }
    }

    inline np::ndarray arma_to_py(arma::Mat<double> const &arma_mat) {
        uint16_t rows = arma_mat.n_rows;
        uint16_t columns = arma_mat.n_cols;
        python::tuple shape = python::make_tuple(rows, columns);
        np::dtype dtype = np::dtype::get_builtin<double>();
        np::ndarray output = np::zeros(shape, dtype);
        for (uint16_t i = 0; i < rows; i++) {
            for (uint16_t j = 0; j < columns; j++) {
                output[i][j] = arma_mat.at(i, j);
            }
        }
        return output;
    }
}

template<>
inline void
activegp::DesignLoader::load_matrices<np::ndarray, np::ndarray, np::ndarray, np::ndarray>(
        np::ndarray const &design,
        np::ndarray const &response,
        np::ndarray const &theta,
        np::ndarray const &k_inv) {
    python_extractor::py_to_arma(design, design_);
    python_extractor::py_to_arma(response, response_);
    python_extractor::py_to_arma(theta, theta_);
    python_extractor::py_to_arma(k_inv, k_inv_);
    // Assertion on shape done before
    Py_intptr_t const *shape = design.get_shape();
    n_ = shape[0];
    n_var_ = shape[1];
}

template<>
inline void
activegp::DesignLoader::load_matrices<np::ndarray, np::ndarray, np::ndarray, np::ndarray, np::ndarray, np::ndarray>(
        np::ndarray const &design,
        np::ndarray const &response,
        np::ndarray const &theta,
        np::ndarray const &k_inv,
        np::ndarray const &design_2,
        np::ndarray const &k_inv_2) {
    load_matrices(design, response, theta, k_inv);
    python_extractor::py_to_arma(k_inv_2, k_inv_2_);
    python_extractor::py_to_arma(design_2, design_2_);
    Py_intptr_t const *shape = design_2.get_shape();
    n_2_ = shape[0];
    assert(n_var_ == shape[1]);
}


#endif //SEQ_LEARNING_PYTHON_TO_ARMADILLO_HPP
