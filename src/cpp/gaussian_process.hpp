//
// Created by sakex on 02.03.20.
//

#ifndef SEQ_LEARNING_GAUSSIAN_PROCESS_HPP
#define SEQ_LEARNING_GAUSSIAN_PROCESS_HPP

#include "cov_types.hpp"
#include <armadillo>
#include <boost/hana.hpp>


namespace activegp {
    template<cov_types cov_type>
    class GP {
    public:
        GP() = default;

        explicit GP(arma::Mat<double> _mat) : __matrix(std::move(_mat)) {}

        template<typename Loader>
        void load(Loader &loader) {
            check_loader_interface(loader);

            n_var = loader.n_var;
            n = loader.n;

            __matrix.resize(n_var, n_var);
            __wij.resize(n, n);

            for (uint16_t i = 0; i < n_var; ++i) {
                for (uint16_t j = i; j < n_var; ++j) {
                    w_kappa_ij(i, j);
                }
            }
        }

        arma::Mat<double> &C() {
            return __matrix;
        }

    private:
        uint16_t n_var = 0;
        uint16_t n = 0;
        arma::Mat<double> __matrix;
        arma::Mat<double> __wij; // Don't reinstanciate

        template<typename Loader>
        static constexpr void check_loader_interface(Loader const &loader) {
            // Check for trait implementation using C++14 idioms and Boost::hana
            constexpr auto valid_design = boost::hana::is_valid(
                    [](auto &&loader) -> decltype(loader.design[0][0]) {
                    });
            constexpr auto valid_response = boost::hana::is_valid(
                    [](auto &&loader) -> decltype(loader.response[0]) {
                    });
            constexpr auto valid_theta = boost::hana::is_valid(
                    [](auto &&loader) -> decltype(loader.theta[0]) {
                    });
            constexpr auto valid_n = boost::hana::is_valid(
                    [](auto &&loader) -> decltype(loader.n) {
                    });
            constexpr auto valid_n_var = boost::hana::is_valid(
                    [](auto &&loader) -> decltype(loader.n_var) {
                    });
            constexpr bool is_valid =
                    valid_design(loader) && valid_response(loader) && valid_theta(loader) && valid_n(loader) &&
                    valid_n_var(loader);
            static_assert(is_valid, "type doesn't implement interface");
        }

        arma::Mat<double> w_kappa_ij(uint16_t, uint16_t);
    };

    template<>
    arma::Mat<double> GP<cov_types::gaussian>::w_kappa_ij(uint16_t const i, uint16_t const j) {

    }
}
#endif //SEQ_LEARNING_GAUSSIAN_PROCESS_HPP
