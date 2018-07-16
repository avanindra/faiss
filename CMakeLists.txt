cmake_minimum_required(VERSION 2.8.7)

# faiss project
project(faiss C CXX)

option(BUILD_TUTORIAL "Build tutorials" ON)
option(BUILD_TEST "Build tests" ON)
option(BUILD_WITH_GPU "Build faiss with gpu (cuda) support" ON)
option(WITH_MKL "Build with MKL if ON (OpenBLAS if OFF)" OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
include_directories( "C:/Projects/pthread-win32-master"
                     ${CMAKE_SOURCE_DIR})
link_directories("C:/Projects/pthread-win32-master/bin/x64_MSVC2013.Release" )

# OpenMP
find_package(OpenMP REQUIRED)
find_package(OpenCV)
find_package(CUDA)
#find_package(OpenMP)

include_directories(${OpenCV_INCLUDE_DIRS})

# BLAS (MKL os OpenBLAS)
if(WITH_MKL)
    find_package(MKL REQUIRED)
    include_directories(${MKL_INCLUDE_DIRS})
    set(BLAS_LIB ${MKL_LIBRARIES})
else()
    find_package(OpenBLAS REQUIRED)
    include_directories(${OpenBLAS_INCLUDE_DIR})
    set(BLAS_LIB ${OpenBLAS_LIB})
endif()
SET( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT" )
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -m64 -Wall  -msse4 -mpopcnt -fopenmp ")#-Wno-sign-compare -std=c++11 -g -O3 -fPIC 
add_definitions(-DFINTEGER=int)

# specify output bin_path and lib_path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# specify header and cpp files
file(GLOB faiss_cpu_headers ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB faiss_cpu_cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set(faiss_lib faiss)
add_library(${faiss_lib}  ${faiss_cpu_headers} ${faiss_cpu_cpp})
target_link_libraries(${faiss_lib} ${BLAS_LIB} ${OpenCV_LIBS} pthread_dll  ${OpenBLAS_LIB})# ${OpenMP_CXX_FLAGS}

# build gpu lib
if(BUILD_WITH_GPU)
    #include(cmake/Cuda.cmake)
    add_subdirectory(gpu)
endif(BUILD_WITH_GPU)

# build tutorial examples
if(BUILD_TUTORIAL)
    add_subdirectory(tutorial)
endif(BUILD_TUTORIAL)

# build tests
if(BUILD_TEST)
    add_subdirectory(tests)
endif(BUILD_TEST)
# Install libraries
install(TARGETS ${faiss_lib}
        ARCHIVE DESTINATION lib
        )
install(FILES ${faiss_cpu_headers} DESTINATION include/faiss)
