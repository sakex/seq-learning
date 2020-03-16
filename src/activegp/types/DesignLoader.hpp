//
// Created by alexandre on 11.03.20.
//

#ifndef SEQ_LEARNING_DESIGNLOADER_HPP
#define SEQ_LEARNING_DESIGNLOADER_HPP

#include <armadillo>

namespace activegp {
    struct DesignLoader {
        uint16_t n_;
        uint16_t n_var_;
        arma::Mat<double> design_;
        arma::vec response_;
        arma::vec theta_;
        arma::Mat<double> k_inv_;

        template <typename ...Ts>
        void load_matrices(Ts&... parameters);
    };
}


#endif //SEQ_LEARNING_DESIGNLOADER_HPP
