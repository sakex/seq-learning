cmake_minimum_required(VERSION 3.14)
project(seq_learning VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -Wl,--no-undefined -D__DEBUG__=1")


# FIND PYTHON, ARMADILLO AND BOOST
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Armadillo REQUIRED)
find_package(Boost REQUIRED python3 numpy3)


# INCLUDE
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${ARMADILLO_INCLUDE_DIRS})

# CREATE LIB
add_library(sequential_learning SHARED src/cpp/helpers/constexprSqrt.hpp src/bindings.cpp src/cpp/kernel.hpp src/cpp/GaussianProcess.hpp src/cpp/types/eCovTypes.h src/cpp/python_loader.h src/cpp/python_loader.cpp src/cpp/types/DesignLoader.h)

# LINK
target_link_libraries(sequential_learning ${ARMADILLO_LIBRARIES} ${Boost_LIBRARIES} ${NUMPY_INCLUDE_DIRS} ${Python3_LIBRARIES})

set_target_properties(sequential_learning PROPERTIES PREFIX "")