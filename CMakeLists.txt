cmake_minimum_required(VERSION 3.10)
project(COCA_Implementation VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native -fopenmp)
elseif(MSVC)
    add_compile_options(/W4 /O2 /openmp)
endif()

# Find required packages
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Include directories
include_directories(src)

# Source files
set(COMMON_HEADERS
    src/coca_model.hpp
    src/io/binary_log.hpp
    src/data/window_maker.hpp
    src/utils/config_parser.hpp
    src/utils/model_io.hpp
)

# Main training executable
add_executable(coca_train 
    apps/coca_train.cpp
    ${COMMON_HEADERS}
)

# Test executable
add_executable(coca_test
    apps/coca_test.cpp
    ${COMMON_HEADERS}
)

# Unit tests executable
add_executable(coca_unit_tests
    tests/test_losses.cpp
    ${COMMON_HEADERS}
)

# Synthetic data generator
add_executable(coca_synthetic
    apps/coca_synthetic.cpp
    ${COMMON_HEADERS}
)

# Link libraries
if(OpenMP_CXX_FOUND)
    target_link_libraries(coca_train OpenMP::OpenMP_CXX)
    target_link_libraries(coca_test OpenMP::OpenMP_CXX)
    target_link_libraries(coca_unit_tests OpenMP::OpenMP_CXX)
    target_link_libraries(coca_synthetic OpenMP::OpenMP_CXX)
endif()

# Installation rules
install(TARGETS coca_train coca_test DESTINATION bin)
install(FILES coca_config.yaml DESTINATION etc/coca)
install(FILES ${COMMON_HEADERS} DESTINATION include/coca)

# Create output directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/models)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/logs)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data)

# Add custom target for tests
enable_testing()
add_test(NAME UnitTests COMMAND coca_unit_tests)
add_test(NAME SyntheticTest COMMAND coca_synthetic --test)

# Print configuration
message(STATUS "COCA Implementation Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
