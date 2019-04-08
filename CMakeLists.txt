set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -lpthread")
cmake_minimum_required(VERSION 2.8.3)
project(cuda_image_filters)
find_package(CUDA REQUIRED)

INCLUDE(FindPkgConfig)
pkg_check_modules(OpenCV REQUIRED opencv)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_62,code=sm_62 --std=c++11)

cuda_add_library(cuda_obj src/bilateral_gpu.cu src/median_gpu.cu)
cuda_add_executable(main src/main.cpp)
target_link_libraries(main ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(main cuda_obj)
