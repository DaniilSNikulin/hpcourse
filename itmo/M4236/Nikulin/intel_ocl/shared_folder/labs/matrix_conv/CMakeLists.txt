cmake_minimum_required (VERSION 3.7)
project(test_tgt)

add_definitions( "-Wall -O2 -fmax-errors=2" )
set (CMAKE_CXX_STANDARD 11)

set(INCLUDE_DIR ../include)
include_directories(${INCLUDE_DIR})

find_package(OpenCL REQUIRED)
add_executable(${PROJECT_NAME} main.cpp ${INCLUDE_DIR}/utils.hpp)
target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)

