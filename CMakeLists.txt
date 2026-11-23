cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(staker LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

add_compile_options(
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-march=native>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-mtune=native>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-ffast-math>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-fno-math-errno>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-Xcompiler=-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-lineinfo>
)

find_package(fmt REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)
find_package(CUDAToolkit REQUIRED)

foreach(_cuda_lib IN ITEMS cudart cuda_driver)
    if(TARGET CUDA::${_cuda_lib} AND NOT TARGET CUDAToolkit::${_cuda_lib})
        add_library(CUDAToolkit::${_cuda_lib} ALIAS CUDA::${_cuda_lib})
    endif()
endforeach()

find_package(OpenCV REQUIRED)
find_library(CFITSIO_LIB cfitsio REQUIRED)

file(GLOB_RECURSE STAKER_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.cu
)

add_executable(staker main.cpp ${STAKER_SOURCES})

target_include_directories(staker PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/include
)

target_link_directories(staker PRIVATE /usr/local/lib)

target_link_libraries(staker PRIVATE
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
    ${CFITSIO_LIB}
    CUDA::cudart
    fmt::fmt
    TBB::tbb
)

set_target_properties(staker PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
