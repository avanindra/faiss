# configure cuda

find_package(CUDA QUIET REQUIRED)
if(CUDA_FOUND)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    #list(APPEND CUDA_LINKER_LIBS  ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES}) #${CUDA_CUDART_LIBRARY}
else(CUDA_FOUND)
    message(STATUS "Could not locate cuda, disabling cuda support.")
    set(BUILD_WITH_GPU OFF)
    return()
endif(CUDA_FOUND)

# set cuda flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;-std=c++11;-DVERBOSE;-g;-lineinfo;-Xcompiler;-ggdb")
else()
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;-O3;") #-std=c++11; -Xcompiler; -DVERBOSE; -DNDEBUG;-DNDEBU
endif()
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
