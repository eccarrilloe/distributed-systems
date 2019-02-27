cmake_minimum_required(VERSION 3.13)
project(sequence)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable(sequence main.cpp)
target_link_libraries( sequence ${OpenCV_LIBS} )