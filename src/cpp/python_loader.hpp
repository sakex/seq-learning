//
// Created by sakex on 03.03.20.
//

#ifndef SEQ_LEARNING_PYTHON_LOADER_HPP
#define SEQ_LEARNING_PYTHON_LOADER_HPP

#include "gaussian_process.hpp"
#include "cov_types.hpp"
#include <boost/numpy.hpp>

namespace activegp {
    namespace np = boost::numpy;

    template<cov_types cov_type>
    class PythonLoader : public GP<cov_type> {
    public:
        PythonLoader(np::ndarray const &design, np::ndarray const &response) {
            Py_intptr_t const * shape = design.get_strides();
            n = shape[0];
            n_var = shape[1];
            load(design, response);
        }

    private:
        uint16_t n;
        uint16_t n_var;
        void load(np::ndarray const &design, np::ndarray const &response) {}
    };

    template<>
    void PythonLoader<cov_types::gaussian>::load(const boost::numpy::ndarray &design,
                                                 const boost::numpy::ndarray &response) {
        for(uint16_t i = 0; i < n_var; ++i) {
            for(uint16_t j = i; j < n_var; ++j) {

            }
        }
    }
}

#endif //SEQ_LEARNING_PYTHON_LOADER_HPP
