cmake_minimum_required(VERSION 3.12)

project(cuda_hook CXX CUDA)

include_directories(/usr/local/cuda/include)

add_library(cuda SHARED cuda_hook.gen.cc)
set_target_properties(cuda PROPERTIES
    LIBRARY cuda
    SOVERSION 1
)
