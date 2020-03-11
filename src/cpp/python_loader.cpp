//
// Created by sakex on 11.03.20.
//

#include "python_loader.hpp"

namespace activegp {
    PythonLoader::PythonLoader(np::ndarray const &_design, np::ndarray const &_response, np::ndarray const &_theta) :
            design(_design), response(_response), theta(_theta) {
        Py_intptr_t const *shape = design.get_strides();
        n = shape[0];
        n_var = shape[1];
        std::cout << n << " " << n_var << std::endl;
    }
}