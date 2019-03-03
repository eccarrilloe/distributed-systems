cmake_minimum_required(VERSION 3.13)
project(openmp)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

add_executable(openmp main.cpp)

target_link_libraries(openmp "${OpenMP_CXX_FLAGS}")
target_link_libraries(openmp ${OpenCV_LIBS})
