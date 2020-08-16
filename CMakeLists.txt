cmake_minimum_required(VERSION 3.10)
project(shape_matching)

set(CMAKE_CXX_STANDARD 11)

option(USE_MIPP "Whether use MIPP library" ON)

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


add_library(shape_matching feature.cpp feature_util.cpp
       	 	pyramid.cpp matching.cpp response_map.cpp similarity.cpp)
target_link_libraries(shape_matching PUBLIC ${OpenCV_LIBS})

if (USE_MIPP)
    add_definitions(-DUSE_MIPP)
    SET(CMAKE_CXX_FLAGS "-march=native")
    target_include_directories(shape_matching PUBLIC ${CMAKE_SOURCE_DIR}/mipp)
endif()

add_subdirectory(test)


add_executable(demo main.cpp)
target_link_libraries(demo shape_matching)
