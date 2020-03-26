#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "activegp/types/eCovTypes.h"
#include "activegp/GaussianProcess.hpp"
#include "ASGP.h"

namespace PythonBindings {
    namespace np = boost::python::numpy;
    namespace python = boost::python;

    void invalid_covtype_python(activegp::InvalidCovType const &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }

    char const *greet() {
        return "Hello World";
    }

    BOOST_PYTHON_MODULE (sequential_learning) {
        using namespace boost::python;
        Py_Initialize();
        np::initialize();
        register_exception_translator<activegp::InvalidCovType>(&invalid_covtype_python);
        def("greet", &greet);
        class_<ASGP>("ASGP", "A wrapper for Active Subspaces of Gaussian Process",
                     init<np::ndarray, np::ndarray, np::ndarray, np::ndarray, char const *>(
                             args("X", "Y", "theta", "Ki", "type")
                     ))
                .def("compute", &ASGP::compute)
                .add_property("X", &ASGP::x)
                .add_property("Y", &ASGP::y)
                .add_property("theta", &ASGP::theta)
                .add_property("Ki", &ASGP::ki)
                .add_property("mat", &ASGP::mat);
    }
}