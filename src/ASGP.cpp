//
// Created by alexandre on 25.03.20.
//

#include "ASGP.h"

namespace PythonBindings {

    ASGP::ASGP(np::ndarray design, np::ndarray response, np::ndarray theta, np::ndarray ki, char const *s) :
            x_(std::move(design)), y_(std::move(response)), theta_(std::move(theta)),
            ki_(std::move(ki)), x2_{np::array(python::object())},
            ki2_{np::array(python::object())}, mat_{np::array(python::object())} {
        select_type(s);
    }

    inline constexpr unsigned hashCovType(const char *str, unsigned index = 0) {
        // Hash table computed at compilation for different branches
        return !str[index] ? 0x1505 : (hashCovType(str, index + 1) * 0x21) ^ (unsigned) str[index];
    }

    void ASGP::select_type(char const *s) {
        unsigned const hash = hashCovType(s);
        switch (hash) {
            case hashCovType("Gaussian"):
            case hashCovType("gaussian"):
            case hashCovType("RBF"):
            case hashCovType("rbf"): {
                type_ = activegp::eCovTypes::gaussian;
                break;
            }
            case hashCovType("Matern3_2"):
            case hashCovType("matern3_2"):
            case hashCovType("Matern32"):
            case hashCovType("matern32"): {
                type_ = activegp::eCovTypes::matern_3_2;
                break;
            }
            case hashCovType("Matern5_2"):
            case hashCovType("matern5_2"):
            case hashCovType("Matern52"):
            case hashCovType("matern52"): {
                type_ = activegp::eCovTypes::matern_5_2;
                break;
            }
            default:
                throw activegp::InvalidCovType(s);
        }
    }

    void ASGP::compute() {
        CHOOSE_IMPL(compute_impl);
    }

    void ASGP::update() {
        CHOOSE_IMPL(update_impl);
    }

    np::ndarray ASGP::x() const {
        return x_;
    }

    np::ndarray ASGP::y() const {
        return y_;
    }

    np::ndarray ASGP::theta() const {
        return theta_;
    }

    np::ndarray ASGP::ki() const {
        return ki_;
    }

    np::ndarray ASGP::x2() const {
        return x2_;
    }

    np::ndarray ASGP::ki2() const {
        return ki_;
    }

    np::ndarray ASGP::mat() const {
        return mat_;
    }
}