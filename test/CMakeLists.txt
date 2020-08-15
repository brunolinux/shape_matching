
set(target_list test_nms test_feature test_spread test_response)

set(source "../feature.cpp" "../pattern_util.cpp" "../response_map.cpp" "../pyramid.cpp")

foreach(target ${target_list})
    add_executable(${target} ${target}.cpp ${source})
    target_link_libraries(${target} "${OpenCV_LIBS}")
endforeach()

add_executable(test_mipp test_mipp.cpp)