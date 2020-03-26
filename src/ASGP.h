//
// Created by alexandre on 25.03.20.
//

#ifndef SEQ_LEARNING_ASGP_H
#define SEQ_LEARNING_ASGP_H

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "activegp/types/eCovTypes.h"
#include "activegp/GaussianProcess.hpp"
#include "python_to_armadillo.hpp"

namespace np = boost::python::numpy;
namespace python = boost::python;

namespace PythonBindings {
#define CHOOSE_IMPL(IMPL) {\
        switch (type_) {\
            case activegp::eCovTypes::gaussian: {\
                IMPL<activegp::eCovTypes::gaussian>();\
                break;\
            }\
            case activegp::eCovTypes::matern_3_2: {\
                IMPL<activegp::eCovTypes::matern_3_2>();\
                break;\
            }\
            case activegp::eCovTypes::matern_5_2: {\
                IMPL<activegp::eCovTypes::matern_5_2>();\
                break;\
            }\
    }\
};

    class ASGP {
        // Active subspace Gaussian Process class exposed to Python
    public:
        ASGP(np::ndarray, np::ndarray, np::ndarray, np::ndarray, char const *s);

        ~ASGP();

        void compute();

        void update();

        [[nodiscard]] np::ndarray x() const;

        [[nodiscard]] np::ndarray y() const;

        [[nodiscard]] np::ndarray theta() const;

        [[nodiscard]] np::ndarray ki() const;

        [[nodiscard]] np::ndarray x2() const;

        [[nodiscard]] np::ndarray ki2() const;

        [[nodiscard]] np::ndarray mat() const;

    private:
        activegp::eCovTypes type_;
        activegp::GPMembers * gp_data = nullptr;
        np::ndarray x_;
        np::ndarray y_;
        np::ndarray theta_;
        np::ndarray ki_;
        np::ndarray x2_;
        np::ndarray ki2_;
        np::ndarray mat_;

    private:
        void select_type(char const *);

        template<activegp::eCovTypes cov_type>
        void load_matrices(activegp::GpImpl<cov_type> &gp) {
            python_extractor::py_to_arma(x_, gp.design());
            python_extractor::py_to_arma(y_, gp.response());
            python_extractor::py_to_arma(theta_, gp.theta());
            python_extractor::py_to_arma(ki_, gp.k_inv());
            // Assertion on shape done before
            Py_intptr_t const *shape = x_.get_shape();
            gp.shape(shape[0], shape[1]);
        }

        template<activegp::eCovTypes cov_type>
        void load_matrices(activegp::GpImpl<cov_type> &gp) {
            python_extractor::py_to_arma(x_, gp.design());
            python_extractor::py_to_arma(y_, gp.response());
            python_extractor::py_to_arma(theta_, gp.theta());
            python_extractor::py_to_arma(ki_, gp.k_inv());
            // Assertion on shape done before
            Py_intptr_t const *shape = x_.get_shape();
            gp.shape(shape[0], shape[1]);
        }

        template<activegp::eCovTypes cov_type>
        inline void compute_impl() {
            auto * gp = dynamic_cast<activegp::GpImpl<cov_type> *>(gp_data);
            load_matrices<cov_type>(*gp);
            gp->compute();
            mat_ = python_extractor::arma_to_py(gp->matrix());
        }

        template<activegp::eCovTypes cov_type>
        inline void update_impl() {
            activegp::GpImpl<cov_type> gp;
            gp.load_matrices(x_, y_, theta_, ki_, ki2_, x2_);
            gp.update();
            mat_ = python_extractor::arma_to_py(gp.matrix());
        }
    };

}
#endif //SEQ_LEARNING_ASGP_H
