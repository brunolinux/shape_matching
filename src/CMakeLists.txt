add_library(shape_matching feature.cpp feature_util.cpp
       	 	pyramid.cpp matching.cpp response_map.cpp similarity.cpp)
target_link_libraries(shape_matching PUBLIC ${OpenCV_LIBS})

if (USE_MIPP)
    add_definitions(-DUSE_MIPP)
    SET(CMAKE_CXX_FLAGS "-march=native")
    target_include_directories(shape_matching PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/mipp)
endif()

add_subdirectory(cuda_icp)
