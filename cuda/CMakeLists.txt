cmake_minimum_required(VERSION 3.13)
project(cuda)

set(CMAKE_CXX_STANDARD 14)
enable_language(CUDA)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

cuda_add_executable(cuda main.cu)
target_link_libraries(cuda ${OpenCV_LIBS} ${CUDA_LIBRARIES})