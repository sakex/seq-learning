//
// Created by sakex on 10.03.20.
//

#ifndef SEQ_LEARNING_LOADER_HPP
#define SEQ_LEARNING_LOADER_HPP

#include <boost/hana.hpp>

namespace activegp::traits {
    template<typename Loader>
    static constexpr bool is_loader(Loader const &loader) {
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
        return is_valid;
    }

}
#endif //SEQ_LEARNING_LOADER_HPP
