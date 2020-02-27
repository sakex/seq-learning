cmake_minimum_required(VERSION 3.14)
project(seq_learning VERSION 0.1)

# FIND PYTHON AND BOOST
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Boost COMPONENTS python3 REQUIRED)

# INCLUDE
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# CREATE LIB
add_library(sequential_learning SHARED src/bindings.cpp)

# LINK
target_link_libraries(sequential_learning ${Boost_LIBRARIES} ${Python3_LIBRARIES})

set_target_properties(sequential_learning PROPERTIES PREFIX "")