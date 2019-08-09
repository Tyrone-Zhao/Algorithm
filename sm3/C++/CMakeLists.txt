cmake_minimum_required(VERSION 3.14)
project(sm3)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(sm3
        sm3.cpp
        sm3.h
        sm3test.cpp)
