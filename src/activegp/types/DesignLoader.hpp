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
        uint16_t n_2_;
        arma::Mat<double> design_;
        arma::Mat<double> design_2_;
        arma::vec response_;
        arma::vec theta_;
        arma::Mat<double> k_inv_;
        arma::Mat<double> k_inv_2_;

        template<typename ...Ts>
        void load_matrices(Ts const &... parameters);
    };
}


#endif //SEQ_LEARNING_DESIGNLOADER_HPP
