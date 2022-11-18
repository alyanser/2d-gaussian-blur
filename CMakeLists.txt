cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD "17")
set(CMAKE_CXX_COMPILER "clang++")

project(convolution)

add_subdirectory(tga)

add_executable(${PROJECT_NAME}
	convolution.cc
)

target_link_libraries(${PROJECT_NAME} PRIVATE
	tga-lib
)

target_include_directories(${PROJECT_NAME} PRIVATE
	tga
)
