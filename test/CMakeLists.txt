
set(target_list test_nms test_feature test_spread 
    test_response test_angle_scale_pair test_linearize
    test_similarity)

foreach(target ${target_list})
    add_executable(${target} ${target}.cpp test_util.cpp)
    target_link_libraries(${target} shape_matching)
endforeach()

add_executable(test_mipp test_mipp.cpp)