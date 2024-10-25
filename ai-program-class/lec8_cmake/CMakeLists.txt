cmake_minimum_required(VERSION 3.0)
project(lec8_cmake)

add_subdirectory(pybind11)
pybind11_add_module(example src/main.cpp)