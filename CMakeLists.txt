cmake_minimum_required(VERSION 3.10)
project(shape_matching)

set(CMAKE_CXX_STANDARD 11)

option(USE_MIPP "Whether use MIPP library" ON)

find_package(OpenCV 3.2.0 EXACT REQUIRED
        PATHS "/home/bruno/program/source_code/libcv_debug/share"
        NO_DEFAULT_PATH)

add_executable(shape_matching main.cpp feature.cpp pattern_util.cpp
        pyramid.cpp matching.cpp response_map.cpp)
target_link_libraries(shape_matching ${OpenCV_LIBS})

if (USE_MIPP)
    add_definitions(-DUSE_MIPP)
    SET(CMAKE_CXX_FLAGS "-march=native")
    target_include_directories(shape_matching PUBLIC ${CMAKE_SOURCE_DIR}/mipp)
endif()


add_subdirectory(test)
