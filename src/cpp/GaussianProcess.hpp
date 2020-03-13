//
// Created by sakex on 02.03.20.
//

#ifndef SEQ_LEARNING_GAUSSIAN_PROCESS_HPP
#define SEQ_LEARNING_GAUSSIAN_PROCESS_HPP

#include "types/eCovTypes.h"
#include <armadillo>
#include <cmath>
#include "helpers/constexprSqrt.hpp"
#include "types/DesignLoader.h"

namespace activegp {

    template<eCovTypes>
    class GP {
    public:
        GP() = default;

        void load(DesignLoader const &loader);

    private:
        uint16_t n_var = 0;
        uint16_t n = 0;
        arma::Mat<double> __matrix;
        arma::Mat<double> __wij; // Instantiate once

        void w_kappa_ij(DesignLoader const &loader, uint16_t derivative1, uint16_t derivative2);

        [[nodiscard]] double ikk(double a, double b, double t) const;

        [[nodiscard]] double w_ii(double a, double b, double t) const;

        [[nodiscard]] double w_ij(double a, double b, double t) const;
    };

    template<>
    inline double GP<eCovTypes::gaussian>::ikk(double const a, double const b, double const t) const {
        // Original was:
        // return((sqrt(PI)*(erf((b+a)/(2.*t)) - erf((b+a-2)/(2.*t)))*t*exp(-(b-a)*(b-a)/(4.*t*t)))/2.);
        // Dividing sqrt(pi) by 2 at compile time to not have to divide everything again at runtime
        constexpr double PI_SQRT = helpers::sqrt(M_PI) / 2.;
        // Try and get some out of order execution
        const double bma_squared = -std::pow(b - a, 2),
                tx2 = t * 2.,
                apb = a + b,
                erf1 = std::erf(apb / tx2),
                erf2 = std::erf((apb - 2) / (tx2)),
                exp = std::exp(bma_squared / (4. * t * t));
        return (PI_SQRT * t) * exp * (erf1 - erf2);
    }

    template<>
    inline double GP<eCovTypes::gaussian>::w_ii(double const a, double const b, double const t) const {
        constexpr double PI_SQRT = helpers::sqrt(M_PI);
        double const a2 = a * a, b2 = b * b, t2 = t * t;
        return (1 / (8 * t2 * t) * ((2 * (-2 + a + b) * std::exp((-a2 - b2 - 2 + 2 * a + 2 * b) / (2 * t2)) * t +
                                     std::exp(-(a - b) * (a - b) / (4 * t2)) * PI_SQRT * ((a - b) * (a - b) - 2 * t2) *
                                     std::erf((-2 + a + b) / (2 * t))) -
                                    (2 * (a + b) * t * std::exp(-((a2 + b2) / (2 * t2))) +
                                     std::exp(-(a - b) * (a - b) / (4 * t2)) * PI_SQRT *
                                     ((a - b) * (a - b) - 2 * t2) *
                                     std::erf((a + b) / (2 * t)))));
    }

    template<>
    inline double GP<eCovTypes::gaussian>::w_ij(double const a, double const b, double const t) const {
        // Original was:
        // return((sqrt(PI)*(erf((b+a)/(2.*t)) - erf((b+a-2)/(2.*t)))*t*exp(-(b-a)*(b-a)/(4.*t*t)))/2.);
        // Dividing sqrt(pi) by 2 at compile time to not have to divide everything again at runtime
        constexpr double PI_SQRT = helpers::sqrt(M_PI) / 2.;
        // Try and get some out of order execution
        const double bma_squared = -std::pow(b - a, 2),
                tx2 = t * 2.,
                apb = a + b,
                erf1 = std::erf(apb / tx2),
                erf2 = std::erf((apb - 2) / (tx2)),
                exp = std::exp(bma_squared / (4. * t * t));
        return (PI_SQRT * t) * exp * (erf1 - erf2);
    }

    template<>
    inline void
    GP<eCovTypes::gaussian>::w_kappa_ij(DesignLoader const &loader, uint16_t const derivative1,
                                        uint16_t const derivative2) {
        if (derivative1 == derivative2) {
            for (int i = 0; i < n; i++) {
                for (int j = i; j < n; j++) {
                    __wij.at(i, j) = w_ii(loader.design(i, derivative1), loader.design(j, derivative1),
                                          loader.theta(derivative1));
                    for (int k = 0; k < n_var; k++) {
                        if (k != derivative1)
                            __wij(i, j) *= ikk(loader.design(i, k), loader.design(j, k), loader.theta(k));
                    }
                    __wij.at(j, i) = __wij.at(i, j);
                }
            }
            return;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                __wij(i, j) = w_ij(loader.design(i, derivative1), loader.design(j, derivative1),
                                   loader.theta(derivative1)) *
                              w_ij(loader.design(j, derivative2), loader.design(i, derivative2),
                                   loader.theta(derivative2));
                if (n_var > 2) {
                    for (int k = 0; k < n_var; k++) {
                        if (k != derivative1 && k != derivative2)
                            __wij(i, j) *= ikk(loader.design(i, k), loader.design(j, k), loader.theta(k));
                    }
                }
            }
        }
    }

    template<>
    inline void GP<eCovTypes::gaussian>::load(DesignLoader const &loader) {
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



#endif //SEQ_LEARNING_GAUSSIAN_PROCESS_HPP
}