//
// Created by sakex on 02.03.20.
//

#ifndef SEQ_LEARNING_GAUSSIAN_PROCESS_HPP
#define SEQ_LEARNING_GAUSSIAN_PROCESS_HPP

#include "types/eCovTypes.h"
#include <armadillo>
#include <cmath>
#include <vector>
#include "helpers/constexprSqrt.hpp"

namespace activegp {
    typedef std::vector<std::vector<arma::Mat<double>>> MatrixOfMatrix;

    class GPMembers {
    public:
        virtual ~GPMembers() = default;

    protected:
        uint16_t n_var_ = 0;
        uint16_t n_ = 0;
        uint16_t n_2_ = 0;
        arma::Mat<double> design_;
        arma::Mat<double> design_2_;
        arma::vec response_;
        arma::vec theta_;
        arma::Mat<double> k_inv_;
        arma::Mat<double> k_inv_2_;
        arma::Mat<double> matrix_;
        MatrixOfMatrix wij_;
    };

    template<eCovTypes>
    class GpImpl : public GPMembers {
        // Inspired by Rust OOP
    public:
        GpImpl() = default;

        void compute();

        void update();

    public:
        // Getters/Setters

        arma::Mat<double> &matrix() {
            return matrix_;
        }


        arma::Mat<double> &design();

        arma::Mat<double> &design2();

        arma::vec &response();

        arma::vec &theta();

        arma::Mat<double> &k_inv();

        arma::Mat<double> &k_inv2();

        uint16_t n_var() const;

        void shape(uint16_t, uint16_t);

        void n_2(uint16_t);


    private:
        static const double m_num_;

    private:
        [[nodiscard]] arma::Mat<double> w_kappa_ij(uint16_t, uint16_t) const;

        [[nodiscard]] arma::Mat<double>
        w_kappa_ij2(arma::Mat<double> const &, arma::Mat<double> const &, uint16_t, uint16_t) const;

        [[nodiscard]] arma::Mat<double> w_kappa_ii(uint16_t) const;

        [[nodiscard]] arma::Mat<double>
        w_kappa_ii2(arma::Mat<double> const &, arma::Mat<double> const &, uint16_t) const;


        [[nodiscard]] double ikk(double a, double b, double t) const;

        [[nodiscard]] double w_ii(double a, double b, double t) const;

        [[nodiscard]] double w_ij(double a, double b, double t) const;
    };

    // -----------------------------------------------------------------------------------------------------------------
    // GAUSSIAN SPCECIALISATION
    template<>
    inline const double GpImpl<eCovTypes::gaussian>::m_num_ = 1;

    template<>
    inline double GpImpl<eCovTypes::gaussian>::ikk(double const a, double const b, double const t) const {
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
                _exp = std::exp(bma_squared / (4. * t * t));
        return (PI_SQRT * t) * _exp * (erf1 - erf2);
    }

    template<>
    inline double GpImpl<eCovTypes::gaussian>::w_ii(double const a, double const b, double const t) const {
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
    inline double GpImpl<eCovTypes::gaussian>::w_ij(double const a, double const b, double const t) const {
        double a2 = a * a, b2 = b * b, t2 = t * t;
        return (-((2 * (exp(-(a2 + b2) / (2 * t2)) - exp((-a2 - b2 + 2 * (a + b - 1)) / (2 * t2))) * t +
                   (a - b) * exp(-(a - b) * (a - b) / (4 * t2)) * sqrt(M_PI) *
                   (erf((-2 + a + b) / (2 * t)) - erf((a + b) / (2 * t))))) / (4 * t));
    }

    // -----------------------------------------------------------------------------------------------------------------
    // MATERN3_2 SPECIALISATION

    template<>
    inline const double GpImpl<eCovTypes::matern_3_2>::m_num_ = 3;

    template<>
    inline double GpImpl<eCovTypes::matern_3_2>::ikk(double a, double b, double const t) const {
        if (b > a) std::swap(a, b);
        double a2 = a * a, b2 = b * b, t2 = t * t, t3 = t2 * t;
        return ((-6 * std::sqrt(3) * a * b * t - 9 * a * t2 - 9 * b * t2 -
                 5 * std::sqrt(3) * t3) / (12. * exp((std::sqrt(3) * (a + b)) / t) * t2) +
                (std::exp((std::sqrt(3) * (-2 + a + b)) / t) * (-6 * std::sqrt(3) * t + 6 * std::sqrt(3) * a * t +
                                                                6 * std::sqrt(3) * b * t -
                                                                6 * std::sqrt(3) * a * b * t - 18 * t2 + 9 * a * t2 +
                                                                9 * b * t2 - 5 * std::sqrt(3) * t3)) / (12. * t2) +
                (std::exp((std::sqrt(3) * (-a + b)) / t) * (6 * a2 * a - 18 * a2 * b +
                                                            18 * a * b2 - 6 * b2 * b + 12 * std::sqrt(3) * a2 * t -
                                                            24 * std::sqrt(3) * a * b * t
                                                            + 12 * std::sqrt(3) * b2 * t + 21 * a * t2 - 21 * b * t2 +
                                                            5 * std::sqrt(3) * t3)) / (12. * t2) +
                (std::exp((2 * std::sqrt(3) * b) / t -
                          (std::sqrt(3) * (a + b)) / t) * (9 * a * t2 - 9 * b * t2 +
                                                           5 * std::sqrt(3) * t3)) / (12. * t2));
    }

    template<>
    inline double GpImpl<eCovTypes::matern_3_2>::w_ii(double a, double b, double const t) const {
        if (b > a) std::swap(a, b);
        double a2 = a * a, b2 = b * b, t2 = t * t;
        return ((-6 * std::sqrt(3) * a * b * t - 3 * a * t2 - 3 * b * t2 - std::sqrt(3) * t2 * t) /
                (4. * std::exp((std::sqrt(3) * (a + b)) / t) * t2 * t2) + (std::exp((std::sqrt(3) * (-2 + a + b)) / t) *
                                                                           (-6 * std::sqrt(3) * t +
                                                                            6 * std::sqrt(3) * a * t +
                                                                            6 * std::sqrt(3) * b * t -
                                                                            6 * std::sqrt(3) * a * b * t -
                                                                            6 * t2 + 3 * a * t2 + 3 * b * t2 -
                                                                            std::sqrt(3) * t2 * t)) / (4. * t2 * t2) +
                (std::exp((2 * std::sqrt(3) * b) / t - (std::sqrt(3) * (a + b)) / t) *
                 (3 * a * t2 - 3 * b * t2 + std::sqrt(3) * t2 * t)) /
                (4. * t2 * t2) + (std::exp((std::sqrt(3) * (-a + b)) / t) *
                                  (-6 * a2 * a + 18 * a2 * b - 18 * a * b2 + 6 * b2 * b + 3 * a * t2 - 3 * b * t2 +
                                   std::sqrt(3) * t2 * t)) / (4. * t2 * t2));

    }

    template<>
    inline double GpImpl<eCovTypes::matern_3_2>::w_ij(double const a, double const b, double const t) const {
        double a2 = a * a, b2 = b * b, t2 = t * t, t3 = t2 * t;
        if (a > b) {
            return ((-6 * a * b * t - 3 * std::sqrt(3) * a * t2 - sqrt(3) * b * t2 - 2 * t3) /
                    (4. * std::exp((std::sqrt(3) * (a + b)) / t) * t3) + (std::exp((std::sqrt(3) * (-a + b)) / t) *
                                                                          (2 * std::sqrt(3) * a2 * a -
                                                                           6 * std::sqrt(3) * a2 * b +
                                                                           6 * std::sqrt(3) * a * b2 -
                                                                           2 * std::sqrt(3) * b2 * b +
                                                                           6 * a2 * t - 12 * a * b * t + 6 * b2 * t -
                                                                           std::sqrt(3) * a * t2 +
                                                                           std::sqrt(3) * b * t2 - 2 * t3)) /
                                                                         (4. * t3) +
                    (std::exp((2 * std::sqrt(3) * b) / t - (std::sqrt(3) * (a + b)) / t) *
                     (3 * std::sqrt(3) * a * t2 - 3 * std::sqrt(3) * b * t2 + 2 * t3)) / (4. * t3) +
                    (std::exp((std::sqrt(3) * (-2 + a + b)) / t) *
                     (6 * t - 6 * a * t - 6 * b * t + 6 * a * b * t + 4 * std::sqrt(3) * t2 -
                      3 * std::sqrt(3) * a * t2 -
                      std::sqrt(3) * b * t2 + 2 * t3)) / (4. * t3));
        } else {
            return ((std::exp((std::sqrt(3) * (a - b)) / t) * (2 * std::sqrt(3) * a2 * a - 6 * std::sqrt(3) * a2 * b +
                                                               6 * std::sqrt(3) * a * b2 - 2 * std::sqrt(3) * b2 * b -
                                                               6 * a2 * t + 12 * a * b * t -
                                                               6 * b2 * t + 3 * std::sqrt(3) * a * t2 -
                                                               3 * std::sqrt(3) * b * t2 -
                                                               2 * t3)) / (4. * t3) +
                    (-6 * a * b * t - 3 * std::sqrt(3) * a * t2 -
                     std::sqrt(3) * b * t2 - 2 * t3) / (4. * std::exp((std::sqrt(3) * (a +
                                                                                       b)) / t) * t3) +
                    (std::exp((std::sqrt(3) * (-2 + a + b)) / t) * (6 * t - 6 * a * t -
                                                                    6 * b * t + 6 * a * b * t + 4 * sqrt(3) * t2 -
                                                                    3 * std::sqrt(3) * a * t2 -
                                                                    std::sqrt(3) * b * t2 + 2 * t3)) / (4. * t3) +
                    (std::exp((2 * std::sqrt(3) * a) / t - (std::sqrt(3) * (a + b)) / t) * (-(std::sqrt(3) * a * t2) +
                                                                                            std::sqrt(3) * b * t2 +
                                                                                            2 * t3)) / (4. * t3));
        }
    }

    // -----------------------------------------------------------------------------------------------------------------
    // MATERN_5_2 SPECIALISATION

    template<>
    inline const double GpImpl<eCovTypes::matern_5_2>::m_num_ = 5. / 3.;

    template<>
    inline double GpImpl<eCovTypes::matern_5_2>::ikk(double a, double b, double const t) const {
        if (b > a) std::swap(a, b);
        double a2 = a * a, b2 = b * b, t2 = t * t, t3 = t2 * t;
        return ((10 * (a - b) * exp((sqrt(5) * (-a + b)) / t) * (5 * a2 * a2 + 5 * b2 * b2 -
                                                                 15 * sqrt(5) * b2 * b * t + 105 * b2 * t2 -
                                                                 54 * sqrt(5) * b * t3 +
                                                                 54 * t2 * t2 - 5 * a2 * a * (4 * b - 3 * sqrt(5) * t) +
                                                                 15 * a2 * (2 * b2 - 3 * sqrt(5) * b * t + 7 * t2) +
                                                                 a *
                                                                 (-20 * b2 * b + 45 * sqrt(5) * b2 * t - 210 * b * t2 +
                                                                  54 * sqrt(5) * t3)) +
                 (3 * t * ((-50 * sqrt(5) * a2 * b2 - 200 * a2 * b * t - 200 * a * b2 * t - 50 * sqrt(5) * a2 * t2 -
                            170 * sqrt(5) * a * b * t2 - 50 * sqrt(5) * b2 * t2 - 225 * a * t3 - 225 * b * t3 -
                            63 * sqrt(5) * t2 * t2) * exp(-sqrt(5) * (a + b) / t) +
                           exp(sqrt(5) * (b - a) / t) * t2 * (50 * sqrt(5) * a2 + 50 * sqrt(5) * b2 - 225 * b * t +
                                                              63 * sqrt(5) * t2 +
                                                              25 * a * (-4 * sqrt(5) * b + 9 * t)))) +
                 3 * t * (exp((sqrt(5) * (-a + b)) / t) * t2 * (50 * sqrt(5) * a2 +
                                                                50 * sqrt(5) * b2 - 25 * a * (4 * sqrt(5) * b - 9 * t) -
                                                                225 * b * t +
                                                                63 * sqrt(5) * t2) -
                          exp((sqrt(5) * (-2 + a + b)) / t) * (50 * sqrt(5) +
                                                               400 * t + 270 * sqrt(5) * t2 + 450 * t3 +
                                                               63 * sqrt(5) * t2 * t2 +
                                                               50 * b2 * (sqrt(5) + 4 * t + sqrt(5) * t2) -
                                                               5 * b * (20 * sqrt(5) +
                                                                        120 * t + 54 * sqrt(5) * t2 + 45 * t3) +
                                                               50 * a2 * (sqrt(5) +
                                                                          sqrt(5) * b2 + 4 * t + sqrt(5) * t2 -
                                                                          2 * b * (sqrt(5) + 2 * t)) -
                                                               5 * a *
                                                               (20 * sqrt(5) + 120 * t + 54 * sqrt(5) * t2 + 45 * t3 +
                                                                20 * b2 * (sqrt(5) + 2 * t) -
                                                                2 * b * (20 * sqrt(5) + 80 * t +
                                                                         17 * sqrt(5) * t2))))) / (540. * t2 * t2));
    }

    template<>
    inline double GpImpl<eCovTypes::matern_5_2>::w_ii(double a, double b, double const t) const {
        if (b > a) std::swap(a, b);
        double a2 = a * a, b2 = b * b, t2 = t * t, t3 = t2 * t;
        return ((-50 * pow(a - b, 3) * exp((sqrt(5) * (-a + b)) / t) *
                 (a2 - 2 * a * b + b2 + sqrt(5) * a * t - sqrt(5) * b * t + t2) +
                 (3 * t * (exp(-(sqrt(5) * (a + b)) / t) *
                           (-50 * sqrt(5) * a2 * b2 - 100 * a2 * b * t - 100 * a * b2 * t - 10 * sqrt(5) * a2 * t2 -
                            50 * sqrt(5) * a * b * t2 - 10 * sqrt(5) * b2 * t2 - 35 * a * t3 - 35 * b * t3 -
                            7 * sqrt(5) * t2 * t2) +
                           exp(sqrt(5) * (b - a) / t) * t2 *
                           (10 * sqrt(5) * a2 - 20 * sqrt(5) * a * b + 10 * sqrt(5) * b2 + 35 * a * t - 35 * b * t +
                            7 * sqrt(5) * t2))) -
                 3 * t * (-(exp((sqrt(5) * (-a + b)) / t) * t2 *
                            (10 * sqrt(5) * a2 - 20 * sqrt(5) * a * b + 10 * sqrt(5) * b2 + 35 * a * t - 35 * b * t +
                             7 * sqrt(5) * t2)) + exp((sqrt(5) * (-2 + a + b)) / t) *
                                                  (50 * sqrt(5) + 200 * t + 70 * sqrt(5) * t2 + 70 * t3 +
                                                   7 * sqrt(5) * t2 * t2 +
                                                   10 * b2 * (5 * sqrt(5) + 10 * t + sqrt(5) * t2) -
                                                   5 * b * (20 * sqrt(5) + 60 * t + 14 * sqrt(5) * t2 + 7 * t3) +
                                                   10 * a2 * (5 * sqrt(5) + 5 * sqrt(5) * b2 + 10 * t + sqrt(5) * t2 -
                                                              10 * b * (sqrt(5) + t)) - 5 * a * (20 * sqrt(5) + 60 * t +
                                                                                                 14 * sqrt(5) * t2 +
                                                                                                 7 * t3 + 20 * b2 *
                                                                                                          (sqrt(5) +
                                                                                                           t) - 10 * b *
                                                                                                                (4 *
                                                                                                                 sqrt(5) +
                                                                                                                 8 * t +
                                                                                                                 sqrt(5) *
                                                                                                                 t2))))) /
                (108. * t3 * t3));
    }

    template<>
    inline double GpImpl<eCovTypes::matern_5_2>::w_ij(double const a, double const b, double const t) const {
        double a2 = a * a, b2 = b * b, t2 = t * t, t3 = t2 * t;
        if (a > b) {
            return ((10 * pow(a - b, 2) * exp((sqrt(5) * (-a + b)) / t) *
                     (sqrt(5) * a2 * a - sqrt(5) * b2 * b + 10 * b2 * t - 9 * sqrt(5) * b * t2 + 9 * t3 +
                      a2 * (-3 * sqrt(5) * b + 10 * t) + a * (3 * sqrt(5) * b2 - 20 * b * t + 9 * sqrt(5) * t2)) +
                     (3 * t * ((-50 * a2 * b2 - 40 * sqrt(5) * a2 * b * t - 20 * sqrt(5) * a * b2 * t - 50 * a2 * t2 -
                                90 * a * b * t2 - 10 * b2 * t2 - 25 * sqrt(5) * a * t3 - 11 * sqrt(5) * b * t3 -
                                18 * t2 * t2) * exp(-(sqrt(5) * (a + b)) / t) +
                               exp(sqrt(5) * (b - a) / t) * t2 *
                               (50 * a2 + 50 * b2 + 25 * sqrt(5) * a * t + 18 * t2 - 25 * b * (4 * a + sqrt(5) * t)))) +
                     3 * t *
                     (-(exp((sqrt(5) * (-a + b)) / t) * t2 * (10 * a2 - 20 * a * b + 10 * b2 + 11 * sqrt(5) * a * t
                                                              - 11 * sqrt(5) * b * t + 18 * t2)) +
                      exp((sqrt(5) * (-2 + a +
                                      b)) / t) * (10 * b2 * (5 + 2 * sqrt(5) * t + t2) - b * (100 +
                                                                                              80 * sqrt(5) * t +
                                                                                              110 * t2 +
                                                                                              11 * sqrt(5) * t3) +
                                                  2 * (25 +
                                                       30 * sqrt(5) * t + 75 * t2 + 18 * sqrt(5) * t3 + 9 * t2 * t2) +
                                                  10 * a2 * (5 + 5 * b2 + 4 * sqrt(5) * t + 5 * t2 - 2 * b * (5 +
                                                                                                              2 *
                                                                                                              sqrt(5) *
                                                                                                              t)) -
                                                  5 * a * (20 + 20 * sqrt(5) * t + 38 * t2 +
                                                           5 * sqrt(5) * t3 + 4 * b2 * (5 + sqrt(5) * t) - 2 * b * (20 +
                                                                                                                    12 *
                                                                                                                    sqrt(5) *
                                                                                                                    t +
                                                                                                                    9 *
                                                                                                                    t2))))) /
                    (108. * t2 * t2 * t));
        } else {
            return ((10 * pow(a - b, 2) * exp((sqrt(5) * (a - b)) / t) *
                     (sqrt(5) * a2 * a - sqrt(5) * b2 * b - 10 * b2 * t - 9 * sqrt(5) * b * t2 - 9 * t3 -
                      a2 * (3 * sqrt(5) * b + 10 * t) + a * (3 * sqrt(5) * b2 + 20 * b * t + 9 * sqrt(5) * t2)) +
                     (3 * t * ((-t2 * (10 * b2 + 11 * sqrt(5) * b * t + 18 * t2) -
                                10 * a2 * (5 * b2 + 4 * sqrt(5) * b * t + 5 * t2) -
                                5 * a * t * (4 * sqrt(5) * b2 + 18 * b * t + 5 * sqrt(5) * t2)) *
                               exp(-sqrt(5) * (a + b) / t) + exp(sqrt(5) * (a - b) / t) * t2 *
                                                             (10 * b2 + 10 * a2 + 11 * sqrt(5) * b * t + 18 * t2 -
                                                              a * (20 * b + 11 * sqrt(5) * t)))) + 3 * t *
                                                                                                   (-(exp((sqrt(5) *
                                                                                                           (a - b)) /
                                                                                                          t) * t2 *
                                                                                                      (50 * a2 +
                                                                                                       50 * b2 +
                                                                                                       25 * sqrt(5) *
                                                                                                       b * t + 18 * t2 -
                                                                                                       25 * a * (4 * b +
                                                                                                                 sqrt(5) *
                                                                                                                 t))) +
                                                                                                    exp((sqrt(5) *
                                                                                                         (-2 + a + b)) /
                                                                                                        t) * (10 * b2 *
                                                                                                              (5 + 2 *
                                                                                                                   sqrt(5) *
                                                                                                                   t +
                                                                                                               t2) - b *
                                                                                                                     (100 +
                                                                                                                      80 *
                                                                                                                      sqrt(5) *
                                                                                                                      t +
                                                                                                                      110 *
                                                                                                                      t2 +
                                                                                                                      11 *
                                                                                                                      sqrt(5) *
                                                                                                                      t3) +
                                                                                                              2 * (25 +
                                                                                                                   30 *
                                                                                                                   sqrt(5) *
                                                                                                                   t +
                                                                                                                   75 *
                                                                                                                   t2 +
                                                                                                                   18 *
                                                                                                                   sqrt(5) *
                                                                                                                   t3 +
                                                                                                                   9 *
                                                                                                                   t2 *
                                                                                                                   t2) +
                                                                                                              10 * a2 *
                                                                                                              (5 +
                                                                                                               5 * b2 +
                                                                                                               4 *
                                                                                                               sqrt(5) *
                                                                                                               t +
                                                                                                               5 * t2 -
                                                                                                               2 * b *
                                                                                                               (5 + 2 *
                                                                                                                    sqrt(5) *
                                                                                                                    t)) -
                                                                                                              5 * a *
                                                                                                              (20 + 20 *
                                                                                                                    sqrt(5) *
                                                                                                                    t +
                                                                                                               38 * t2 +
                                                                                                               5 *
                                                                                                               sqrt(5) *
                                                                                                               t3 +
                                                                                                               4 * b2 *
                                                                                                               (5 +
                                                                                                                sqrt(5) *
                                                                                                                t) -
                                                                                                               2 * b *
                                                                                                               (20 +
                                                                                                                12 *
                                                                                                                sqrt(5) *
                                                                                                                t + 9 *
                                                                                                                    t2))))) /
                    (108. * t2 * t2 * t));
        }
    }

    // -----------------------------------------------------------------------------------------------------------------
    // GENERAL METHODS

    template<eCovTypes cov_type>
    inline arma::Mat<double>
    GpImpl<cov_type>::w_kappa_ij(uint16_t const derivative1, uint16_t const derivative2) const {
        arma::Mat<double> wij_temp(n_, n_);
        for (int i = 0; i < n_; i++) {
            for (int j = 0; j < n_; j++) {
                wij_temp.at(i, j) = w_ij(design_(i, derivative1), design_(j, derivative1),
                                         theta_(derivative1)) *
                                    w_ij(design_(j, derivative2), design_(i, derivative2),
                                         theta_(derivative2));
                if (n_var_ > 2) {
                    for (int k = 0; k < n_var_; k++) {
                        if (k != derivative1 && k != derivative2)
                            wij_temp.at(i, j) *= ikk(design_(i, k), design_(j, k), theta_(k));
                    }
                }
            }
        }
        return wij_temp;
    }

    template<eCovTypes cov_type>
    inline arma::Mat<double>
    GpImpl<cov_type>::w_kappa_ij2(arma::Mat<double> const &X0, arma::Mat<double> const &X1, uint16_t const derivative1,
                                  uint16_t const derivative2) const {
        arma::Mat<double> wij_temp(n_, n_);
        for (int i = 0; i < n_; i++) {
            for (int j = 0; j < n_2_; j++) {
                wij_temp.at(i, j) = w_ij(X0(i, derivative1), X1(j, derivative1),
                                         theta_(derivative1)) *
                                    w_ij(X0(j, derivative2), X1(i, derivative2),
                                         theta_(derivative2));
                if (n_var_ > 2) {
                    for (int k = 0; k < n_var_; k++) {
                        if (k != derivative1 && k != derivative2)
                            wij_temp.at(i, j) *= ikk(X0(i, k), X1(j, k), theta_(k));
                    }
                }
            }
        }
        return wij_temp;
    }

    template<eCovTypes cov_type>
    inline arma::Mat<double>
    GpImpl<cov_type>::w_kappa_ii(uint16_t const derivative) const {
        arma::Mat<double> wij_temp(n_, n_);
        for (int i = 0; i < n_; i++) {
            for (int j = i; j < n_; j++) {
                wij_temp.at(i, j) = w_ii(design_(i, derivative), design_(j, derivative),
                                         theta_(derivative));
                for (int k = 0; k < n_var_; k++) {
                    if (k != derivative)
                        wij_temp.at(i, j) *= ikk(design_(i, k), design_(j, k), theta_(k));
                }
                wij_temp.at(j, i) = wij_temp.at(i, j);
            }
        }
        return wij_temp;
    }

    template<eCovTypes cov_type>
    inline arma::Mat<double>
    GpImpl<cov_type>::w_kappa_ii2(arma::Mat<double> const &X0, arma::Mat<double> const &X1,
                                  uint16_t const derivative) const {
        size_t const n1 = X0.n_rows;
        size_t const n2 = X1.n_rows;
        arma::Mat<double> wij_temp(n1, n2);
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                wij_temp.at(i, j) = w_ii(X0(i, derivative), X1(j, derivative), theta_(derivative));
                for (int k = 0; k < n_var_; k++) {
                    if (k != derivative)
                        wij_temp.at(i, j) *= ikk(X0(i, k), X1(j, k), theta_(k));
                }
            }
        }
        return wij_temp;
    }

    template<eCovTypes cov_type>
    inline void GpImpl<cov_type>::compute() {
        matrix_.resize(n_var_, n_var_);
        wij_.resize(n_var_); // Note that this is a matrix of matrices
        for(auto & v: wij_) v.resize(n_var_);

        arma::vec kir = k_inv_.t() * response_; // Cross product
        arma::Mat<double> t_kir = kir.t();

        for (uint16_t i = 0; i < n_var_; ++i) {
            // Unrolling their loop (first iter)
            arma::Mat<double> wii_temp = w_kappa_ii(i);
            // Branch miss predictions shouldn't be that bad considering we are dealing with a constant bool
            double const theta_squared = std::pow(theta_.at(i), 2.);
            arma::Mat<double> m =
                    (m_num_ / theta_squared) - arma::accu(k_inv_ % wii_temp) + (t_kir * (wii_temp * kir));
            matrix_.at(i, i) = m.at(0, 0);
            wij_[i][i] = std::move(wii_temp);
            for (uint16_t j = i + 1; j < n_var_; ++j) {
                arma::Mat<double> wij_temp = w_kappa_ij(i, j);
                m = -arma::accu(k_inv_ % wij_temp) + (t_kir * (wij_temp * kir));
                double const m_val = m(0, 0);
                matrix_.at(i, j) = m_val;
                matrix_.at(j, i) = m_val;
                wij_[i][j] = std::move(wii_temp);
            }
        }
    }

    template<eCovTypes cov_type>
    inline void GpImpl<cov_type>::update() {
        //wij_.resize(n_var_, n_var_); // Note that this is a matrix of matrices
        arma::Mat<double> kikn = k_inv_ * k_inv_2_.t(); // T Cross product
        arma::Mat<double> t_kikn = kikn.t();
        arma::Mat<double> rbind = arma::join_vert(design_, design_2_);
        for (uint16_t i = 0; i < n_var_; ++i) {
            arma::vec wa = arma::vectorise(w_kappa_ii2(design_, design_2_, i));
            arma::vec wb = arma::vectorise(w_kappa_ii2(design_2_, rbind, i));
            double const w = wb(wb.n_elem - 1);
            wb.shed_col(wb.n_elem - 1);
            arma::Mat<double> kntKiWij = t_kikn * wij_[i][i];
            arma::Mat<double> tmp = - (wa+wb).t() * gn;
            for (uint16_t j = i + 1; j < n_var_; ++j) {
            }
        }
    }

    template<eCovTypes cov_type>
    inline arma::Mat<double> &GpImpl<cov_type>::design() {
        return design_;
    }

    template<eCovTypes cov_type>
    inline arma::Mat<double> &GpImpl<cov_type>::design2() {
        return design_2_;
    }

    template<eCovTypes cov_type>
    inline arma::vec &GpImpl<cov_type>::response() {
        return response_;
    }

    template<eCovTypes cov_type>
    inline uint16_t GpImpl<cov_type>::n_var() const {
        return n_var_;
    }

    template<eCovTypes cov_type>
    inline arma::vec &GpImpl<cov_type>::theta() {
        return theta_;
    }

    template<eCovTypes cov_type>
    inline arma::Mat<double> &GpImpl<cov_type>::k_inv() {
        return k_inv_;
    }

    template<eCovTypes cov_type>
    inline arma::Mat<double> &GpImpl<cov_type>::k_inv2() {
        return k_inv_2_;
    }

    template<eCovTypes cov_type>
    inline void GpImpl<cov_type>::shape(uint16_t const n, uint16_t const n_var) {
        n_ = n;
        n_var_ = n_var;
    }

    template<eCovTypes cov_type>
    inline void GpImpl<cov_type>::n_2(uint16_t const n) {
        n_2_ = n;
    }
}

#endif //SEQ_LEARNING_GAUSSIAN_PROCESS_HPP