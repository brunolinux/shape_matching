
SET(icp_cuda_srcs  icp.cu scene/common.cu scene/edge_scene/edge_scene.cu)
SET(icp_srcs  icp.cpp scene/common.cpp scene/edge_scene/edge_scene.cpp)


if(USE_CUDA)
# cuda
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O3 --default-stream per-thread -Xcompiler -fopenmp")
set(icp_cuda_inc ${CUDA_INCLUDE_DIRS})
set(icp_cuda_lib ${CUDA_LIBRARIES})
CUDA_COMPILE(icp_cuda_objs ${icp_cuda_srcs})
endif()

# eigen
find_package(Eigen3 REQUIRED)


# lib & test exe
add_library(cuda_icp
               ${icp_srcs}
               ${icp_cuda_srcs}
               ${icp_cuda_objs} 
)
target_include_directories(cuda_icp PUBLIC ${icp_cuda_inc})
target_link_libraries(cuda_icp PUBLIC ${icp_cuda_lib} ${OpenCV_LIBS} Eigen3::Eigen)
