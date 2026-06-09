cmake_minimum_required(VERSION 3.15)
project(test_package CXX)

find_package(improc 1.0.3 CONFIG REQUIRED)

add_executable(test_package src/test_package.cpp)
target_link_libraries(test_package PRIVATE improc::improc)
