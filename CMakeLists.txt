cmake_minimum_required(VERSION 3.20)
project(micrograd CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # <-- this line is the magic

add_library(micrograd STATIC
    src/value.cpp
    src/neuron.cpp
    src/layer.cpp
    src/mlp.cpp)
target_include_directories(micrograd PUBLIC include)
target_compile_options(micrograd PRIVATE -Wall -Wextra -g -O0 -Wimplicit-fallthrough)

add_executable(microgradtest examples/microgradtest.cpp)
target_link_libraries(microgradtest PRIVATE micrograd)
target_compile_options(microgradtest PRIVATE -Wall -Wextra -g -O0 -Wimplicit-fallthrough)

add_executable(mlptest examples/mlptest.cpp)
target_link_libraries(mlptest PRIVATE micrograd)
target_compile_options(mlptest PRIVATE -Wall -Wextra -g -O0 -Wimplicit-fallthrough)
