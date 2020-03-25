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
    class ASGP {
        // Active subspace Gaussian Process class exposed to Python
    public:
        ASGP(np::ndarray, np::ndarray, np::ndarray, np::ndarray, char const *s);

        void compute();

        np::ndarray x() const;
        np::ndarray y() const;
        np::ndarray theta() const;
        np::ndarray ki() const;
        np::ndarray x2() const;
        np::ndarray ki2() const;
        np::ndarray mat() const;

    private:
        activegp::eCovTypes type_;
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
        inline void compute_impl() {
            activegp::GP<cov_type> gp;
            activegp::DesignLoader loader;
            loader.load_matrices(x_, y_, theta_, ki_);
            gp.compute(loader);
            mat_ = python_extractor::arma_to_py(gp.matrix());
        }
    };

}


#endif //SEQ_LEARNING_ASGP_H
