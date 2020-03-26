cmake_minimum_required(VERSION 3.14)
project(seq_learning VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O3 -Werror=return-type -Wl,--no-undefined -D__DEBUG__=1 -flto -funroll-loops -march=native")


# FIND Threads, PYTHON, ARMADILLO AND BOOST
find_package (Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Armadillo REQUIRED)
find_package(Boost REQUIRED python3 numpy3)


# INCLUDE
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${ARMADILLO_INCLUDE_DIRS})

# CREATE LIB
add_library(sequential_learning SHARED src/activegp/helpers/constexprSqrt.hpp src/bindings.cpp src/activegp/GaussianProcess.hpp src/activegp/types/eCovTypes.h src/python_to_armadillo.hpp src/ASGP.cpp src/ASGP.h src/activegp/helpers/thread_n_m.hpp)

# LINK
target_link_libraries(sequential_learning ${CMAKE_THREAD_LIBS_INIT} ${ARMADILLO_LIBRARIES} ${Boost_LIBRARIES} ${NUMPY_INCLUDE_DIRS} ${Python3_LIBRARIES})

set_target_properties(sequential_learning PROPERTIES PREFIX "")