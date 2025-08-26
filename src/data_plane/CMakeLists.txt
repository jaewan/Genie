cmake_minimum_required(VERSION 3.16)
project(genie_data_plane)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find DPDK
find_package(PkgConfig REQUIRED)
pkg_check_modules(DPDK REQUIRED libdpdk)

# Find CUDA (optional)
find_package(CUDA)
if(CUDA_FOUND)
    add_definitions(-DGENIE_CUDA_SUPPORT)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Find JSON library
find_package(nlohmann_json 3.2.0 REQUIRED)

# Source files
set(SOURCES
    genie_data_plane.cpp
    genie_data_plane_enhanced.cpp
    genie_data_plane_factory.cpp
    genie_zero_copy_rx.cpp
    genie_extbuf_tx.cpp
    genie_dpdk_thread_model.cpp  # Use DPDK native threading instead
)

# Create shared library
add_library(genie_data_plane SHARED ${SOURCES})

# Include directories
target_include_directories(genie_data_plane PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DPDK_INCLUDE_DIRS}
)

# Compile options
target_compile_options(genie_data_plane PRIVATE
    ${DPDK_CFLAGS_OTHER}
    -Wall
    -Wextra
    -O3
    -march=native
    -mtune=native
    -msse4.2
)

# Link libraries
target_link_libraries(genie_data_plane
    ${DPDK_LIBRARIES}
    nlohmann_json::nlohmann_json
    pthread
    numa
    dl
)

if(CUDA_FOUND)
    target_link_libraries(genie_data_plane ${CUDA_LIBRARIES})
endif()

# Link flags
target_link_options(genie_data_plane PRIVATE
    ${DPDK_LDFLAGS_OTHER}
)

# Installation
install(TARGETS genie_data_plane
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Add test executable for DPDK threading
add_executable(test_dpdk_threading test_dpdk_threading.cpp genie_dpdk_thread_model.cpp)
target_link_directories(test_dpdk_threading PRIVATE /opt/dpdk/dpdk-23.11/install/lib/x86_64-linux-gnu)
target_link_libraries(test_dpdk_threading
    ${DPDK_LIBRARIES}
    pthread
    numa
    dl
)
target_compile_options(test_dpdk_threading PRIVATE ${DPDK_CFLAGS})

# Install test executable
install(TARGETS test_dpdk_threading
    RUNTIME DESTINATION bin
)

# Add simple DPDK demo
add_executable(test_dpdk_simple test_dpdk_simple.cpp)
target_link_directories(test_dpdk_simple PRIVATE /opt/dpdk/dpdk-23.11/install/lib/x86_64-linux-gnu)
target_link_libraries(test_dpdk_simple
    ${DPDK_LIBRARIES}
    pthread
)
target_compile_options(test_dpdk_simple PRIVATE ${DPDK_CFLAGS})

install(FILES genie_data_plane.hpp
    DESTINATION include/genie
)

# Test executable (optional)
if(BUILD_TESTS)
    add_executable(test_data_plane test_data_plane.cpp)
    target_link_libraries(test_data_plane genie_data_plane)
endif()

# Print configuration
message(STATUS "DPDK include dirs: ${DPDK_INCLUDE_DIRS}")
message(STATUS "DPDK libraries: ${DPDK_LIBRARIES}")
message(STATUS "CUDA support: ${CUDA_FOUND}")
if(CUDA_FOUND)
    message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
endif()