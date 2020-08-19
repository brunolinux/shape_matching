cmake_minimum_required(VERSION 3.10)
project(shape_matching)

set(CMAKE_CXX_STANDARD 11)

option(USE_MIPP "Whether use MIPP library" ON)
option(USE_CUDA "Whether use CUDA library" OFF)

if (WIN32)
find_package(OpenCV REQUIRED
             PATHS [=[E:\program\lib\libopencv\x64\vc16\lib]=]
             NO_DEFAULT_PATH
             )
else()
find_package(OpenCV 3.2.0 EXACT REQUIRED
             PATHS "/home/bruno/program/source_code/libcv_debug/share"
             NO_DEFAULT_PATH)
endif()


add_subdirectory(src)

add_executable(demo main.cpp)
target_link_libraries(demo shape_matching cuda_icp)

add_subdirectory(test)
add_subdirectory(self_test)
