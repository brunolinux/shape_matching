
set(target_list test_response_nms test_feature test_spread
    test_response test_angle_scale_pair test_linearize
    test_similarity test_readwrite)

foreach(target ${target_list})
    add_executable(${target} ${target}.cpp test_util.cpp)
    target_link_libraries(${target} shape_matching)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/src)
endforeach()

add_executable(test_mipp test_mipp.cpp)
