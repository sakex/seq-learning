//
// Created by sakex on 02.03.20.
//

#ifndef SEQ_LEARNING_GAUSSIAN_PROCESS_HPP
#define SEQ_LEARNING_GAUSSIAN_PROCESS_HPP

#include "cov_types.hpp"
#include <armadillo>

namespace activegp {
    template<cov_types cov_type, typename Loader>
    class GP {
    public:
        GP() = default;

        arma::Mat<double> C() const {
            return __matrix;
        }

        void set_shape(uint16_t const _n, uint16_t const _n_var) {
            n = _n;
            n = _n_var;
        }

    private:
        uint16_t n = 0;
        uint16_t n_var = 0;
        arma::Mat<double> __matrix;
    };
}
#endif //SEQ_LEARNING_GAUSSIAN_PROCESS_HPP
