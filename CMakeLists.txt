cmake_minimum_required(VERSION 3.20)

project(convolution CXX)

find_package(CUDA REQUIRED)

set(CUDA_HOST_COMPILER clang++)

set(CMAKE_BUILD_TYPE "release")

list(APPEND CUDA_NVCC_FLAGS "-std=c++17")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

cuda_add_executable(${PROJECT_NAME} 
	src/convolution.cu
)

add_subdirectory(tga)
target_include_directories(${PROJECT_NAME} PRIVATE tga)
target_link_libraries(${PROJECT_NAME} tga-lib)
