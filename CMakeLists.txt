project(ex)
cmake_minimum_required(VERSION 3.0)

find_package(OpenCV REQUIRED)
include(mini-caffe/mini-caffe.cmake)

add_executable(ex ex.hpp ex.cpp main.cpp)
target_link_libraries(ex libcaffe ${OpenCV_LIBS})
