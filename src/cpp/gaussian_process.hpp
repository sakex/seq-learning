//
// Created by sakex on 02.03.20.
//

#ifndef SEQ_LEARNING_GAUSSIAN_PROCESS_HPP
#define SEQ_LEARNING_GAUSSIAN_PROCESS_HPP

#include "cov_types.hpp"
#include "traits/loader.hpp"
#include <armadillo>
#include "traits/loader.hpp"


namespace activegp {

    template<cov_types>
    class GP {
    public:
        GP() = default;

        template<typename Loader>
        void load(Loader const &loader);

        arma::Mat<double> &C() {
            return __matrix;
        }

    private:
        uint16_t n_var = 0;
        uint16_t n = 0;
        arma::Mat<double> __matrix;
        arma::Mat<double> __wij; // Don't reinstantiate

        template<class Loader>
        void w_kappa_ij(Loader const &loader, uint16_t const derivative1, uint16_t const derivative2);
    };

    template<>
    template<class Loader>
    void GP<cov_types::gaussian>::load(Loader const &loader) {
        static_assert(traits::is_loader(loader), "type doesn't implement interface");

        n_var = loader.n_var;
        n = loader.n;

        __matrix.resize(n_var, n_var);
        __wij.resize(n, n);

        for (uint16_t i = 0; i < n_var; ++i) {
            for (uint16_t j = i; j < n_var; ++j) {
                w_kappa_ij(loader, i, j);
            }
        }
    }

    template<>
    template<class Loader>
    void
    GP<cov_types::gaussian>::w_kappa_ij(Loader const &loader, uint16_t const derivative1, uint16_t const derivative2) {
        if (derivative1 == derivative2) {
            for (int i = 0; i < n; i++) {
                for (int j = i; j < n; j++) {
                    __wij.at(i, j) = w_ii_cpp(loader.design(i, derivative1), loader.design(j, derivative1),
                                              loader.theta(derivative1));
                    for (int k = 0; k < n_var; k++) {
                        if (k != derivative1)
                            __wij(i, j) *= Ikk_cpp(loader.design(i, k), loader.design(j, k), loader.theta(k));
                    }
                    __wij.at(j, i) = __wij.at(i, j);
                }
            }
            return;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                __wij(i, j) = w_ij_cpp(loader.design(i, derivative1), loader.design(j, derivative1),
                                       loader.theta(derivative1)) *
                              w_ij_cpp(design(j, derivative2), design(i, derivative2), theta(derivative2));
                if (n_var > 2) {
                    for (int k = 0; k < n_var; k++) {
                        if (k != derivative1 && k != derivative2)
                            __wij(i, j) *= Ikk_cpp(loader.design(i, k), loader.design(j, k), loader.theta(k));
                    }
                }
            }
        }
    }

#endif //SEQ_LEARNING_GAUSSIAN_PROCESS_HPP
}