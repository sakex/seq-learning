//
// Created by alexandre on 11.03.20.
//

#ifndef SEQ_LEARNING_DESIGNLOADER_H
#define SEQ_LEARNING_DESIGNLOADER_H

#include <armadillo>

namespace activegp {
    struct DesignLoader {
        uint16_t n;
        uint16_t n_var;
        arma::Mat<double> design;
        arma::Mat<double> response;
        arma::Mat<double> theta;
    };
}


#endif //SEQ_LEARNING_DESIGNLOADER_H
