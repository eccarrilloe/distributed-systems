cmake_minimum_required(VERSION 3.13)
project(cuda)

set(CMAKE_CXX_STANDARD 14)
enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cuda main.cu)
