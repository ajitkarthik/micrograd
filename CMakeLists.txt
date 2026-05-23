cmake_minimum_required(VERSION 3.20)
project(micrograd CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # <-- this line is the magic

add_executable(micrograd src/main.cpp)
target_compile_options(micrograd PRIVATE -Wall -Wextra -g -O0)
