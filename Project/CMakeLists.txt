cmake_minimum_required(VERSION 3.20)

project(OpenCVCL)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(OpenCVCL main.cpp)

target_link_libraries(${PROJECT_NAME} OpenCL ${OpenCV_LIBS})
