//
// Created by alexandre on 11.03.20.
//

#ifndef SEQ_LEARNING_CONSTEXPRSQRT_HPP
#define SEQ_LEARNING_CONSTEXPRSQRT_HPP

#include <limits>

namespace helpers {
    // taken from https://stackoverflow.com/a/34134071

    long double constexpr sqrt_newton_raphson(long double x, long double curr, long double prev) {
        return curr == prev
               ? curr
               : sqrt_newton_raphson(x, 0.5 * (curr + x / curr), curr);
    }

    long double constexpr sqrt(long double x) {
        return x >= 0 && x < std::numeric_limits<long double>::infinity()
               ? sqrt_newton_raphson(x, x, 0)
               : std::numeric_limits<long double>::quiet_NaN();
    }
}


#endif //SEQ_LEARNING_CONSTEXPRSQRT_HPP
