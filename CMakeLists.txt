cmake_minimum_required(VERSION 3.20)
project(micrograd CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # <-- this line is the magic

add_library(micrograd STATIC src/value.cpp)
target_include_directories(micrograd PUBLIC include)

add_executable(neuron examples/neuron.cpp)
target_link_libraries(neuron PRIVATE micrograd)
target_compile_options(neuron PRIVATE -Wall -Wextra -g -O0)
