//
// Created by sakex on 02.03.20.
//

#ifndef SEQ_LEARNING_GAUSSIAN_PROCESS_HPP
#define SEQ_LEARNING_GAUSSIAN_PROCESS_HPP

#include "types/eCovTypes.h"
#include <armadillo>
#include <cmath>
#include "helpers/constexprSqrt.hpp"
#include "types/DesignLoader.hpp"

namespace activegp {

    template<eCovTypes>
    class GP {
    public:
        GP() = default;

        void compute(DesignLoader const &loader);

        arma::Mat<double> const &matrix() {
            return matrix_;
        }

    private:
        static const double m_num_;
        uint16_t n_var_ = 0;
        uint16_t n_ = 0;
        arma::Mat<double> theta_;
        arma::Mat<double> matrix_;

        arma::Mat<double> w_kappa_ij(DesignLoader const &loader, uint16_t derivative1, uint16_t derivative2);

        [[nodiscard]] double ikk(double a, double b, double t) const;

        [[nodiscard]] double w_ii(double a, double b, double t) const;

        [[nodiscard]] double w_ij(double a, double b, double t) const;
    };

    template<>
    const double GP<eCovTypes::gaussian>::m_num_ = 1;

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
    inline arma::Mat<double>
    GP<eCovTypes::gaussian>::w_kappa_ij(DesignLoader const &loader, uint16_t const derivative1,
                                        uint16_t const derivative2) {
        arma::Mat<double> wij_temp(n_, n_);
        if (derivative1 == derivative2) {
            for (int i = 0; i < n_; i++) {
                for (int j = i; j < n_; j++) {
                    wij_temp.at(i, j) = w_ii(loader.design_(i, derivative1), loader.design_(j, derivative1),
                                             theta_(derivative1));
                    for (int k = 0; k < n_var_; k++) {
                        if (k != derivative1)
                            wij_temp.at(i, j) *= ikk(loader.design_(i, k), loader.design_(j, k), theta_(k));
                    }
                    wij_temp.at(j, i) = wij_temp.at(i, j);
                }
            }
            return wij_temp;
        }
        for (int i = 0; i < n_; i++) {
            for (int j = 0; j < n_; j++) {
                wij_temp.at(i, j) = w_ij(loader.design_(i, derivative1), loader.design_(j, derivative1),
                                         theta_(derivative1)) *
                                    w_ij(loader.design_(j, derivative2), loader.design_(i, derivative2),
                                          theta_(derivative2));
                if (n_var_ > 2) {
                    for (int k = 0; k < n_var_; k++) {
                        if (k != derivative1 && k != derivative2)
                            wij_temp.at(i, j) *= ikk(loader.design_(i, k), loader.design_(j, k), theta_(k));
                    }
                }
            }
        }
        return wij_temp;
    }

    template<eCovTypes cov_type>
    inline void GP<cov_type>::compute(DesignLoader const &loader) {
        n_var_ = loader.n_var_;
        n_ = loader.n_;

        matrix_.resize(n_var_, n_var_);
        //wij_.resize(n_var_, n_var_); // Note that this is a matrix of matrices
        theta_ = arma::sqrt(loader.theta_ / 2.);

        arma::Mat<double> kir = loader.k_inv_.t() * loader.response_; // Cross product
        arma::Mat<double> t_kir = kir.t();
        for (uint16_t i = 0; i < n_var_; ++i) {
            // Unrolling their loop (first iter)
            arma::Mat<double> wii_temp = w_kappa_ij(loader, i, i);
            double const theta_squared = std::pow(theta_.at(i), 2);
            arma::Mat<double> m =
                    (m_num_ / theta_squared) - arma::accu(loader.k_inv_ % wii_temp) + t_kir * (wii_temp * kir);
            matrix_(i, i) = m(0, 0);
            //wij_(i, i) = wij_temp_;

            for (uint16_t j = i + 1; j < n_var_; ++j) {
                arma::Mat<double> wij_temp = w_kappa_ij(loader, i, j);

                std::cout << wij_temp << std::endl;
                return;

                std::cout << t_kir * (wij_temp * kir) << std::endl;
                std::cout << -arma::accu(loader.k_inv_ % wij_temp) << std::endl;
                m = -arma::accu(loader.k_inv_ % wij_temp) + t_kir * (wij_temp * kir);
                double const m_val = m(0, 0);
                matrix_.at(i, j) = m_val;
                matrix_.at(j, i) = m_val;
                //wij_(i, j) = wij_temp_;
            }
        }
    }


#endif //SEQ_LEARNING_GAUSSIAN_PROCESS_HPP
}