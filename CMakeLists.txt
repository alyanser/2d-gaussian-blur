cmake_minimum_required(VERSION 3.20)

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
