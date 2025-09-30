cmake_minimum_required(VERSION 3.16)
project(genie_data_plane)
# Optional KCP support
option(GENIE_WITH_KCP "Enable KCP reliable transport" OFF)
set(KCP_LIBRARIES "")
if(GENIE_WITH_KCP)
    message(STATUS "Checking for KCP (ikcp) ...")
    # Assume ikcp headers/libraries are available in system or provided path
    find_path(KCP_INCLUDE_DIRS ikcp.h PATHS /usr/local/include /usr/include $ENV{KCP_INCLUDE_DIR})
    find_library(KCP_LIB ikcp PATHS /usr/local/lib /usr/lib $ENV{KCP_LIB_DIR})
    if(KCP_INCLUDE_DIRS AND KCP_LIB)
        message(STATUS "KCP found: ${KCP_INCLUDE_DIRS} ${KCP_LIB}")
        include_directories(${KCP_INCLUDE_DIRS})
        set(KCP_LIBRARIES ${KCP_LIB})
        add_definitions(-DGENIE_WITH_KCP)
    else()
        message(WARNING "KCP not found; building without KCP linkage (stubs active)")
        set(GENIE_WITH_KCP OFF)
    endif()
endif()

# Optional SPDK support
option(GENIE_WITH_SPDK "Enable SPDK-backed DMA buffers" OFF)
set(SPDK_LIBRARIES "")
if(GENIE_WITH_SPDK)
    message(STATUS "Checking for SPDK ...")
    find_path(SPDK_INCLUDE_DIR spdk/env.h PATHS /usr/local/include /usr/include $ENV{SPDK_INCLUDE_DIR})
    # Try common SPDK env libraries
    find_library(SPDK_ENV_LIB spdk_env_dpdk PATHS /usr/local/lib /usr/lib $ENV{SPDK_LIB_DIR})
    if(NOT SPDK_ENV_LIB)
        find_library(SPDK_ENV_LIB spdk_env PATHS /usr/local/lib /usr/lib $ENV{SPDK_LIB_DIR})
    endif()
    if(SPDK_INCLUDE_DIR AND SPDK_ENV_LIB)
        message(STATUS "SPDK found: ${SPDK_INCLUDE_DIR} ${SPDK_ENV_LIB}")
        include_directories(${SPDK_INCLUDE_DIR})
        list(APPEND SPDK_LIBRARIES ${SPDK_ENV_LIB})
        add_definitions(-DGENIE_WITH_SPDK)
    else()
        message(WARNING "SPDK not found; building without SPDK (fallback to rte_malloc)")
        set(GENIE_WITH_SPDK OFF)
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find DPDK
find_package(PkgConfig REQUIRED)

# Try to find DPDK through pkg-config
pkg_check_modules(DPDK libdpdk)

if(NOT DPDK_FOUND)
    # Try alternative DPDK paths
    set(DPDK_SEARCH_PATHS
        /opt/dpdk/dpdk-23.11/install
        /usr/local
        /usr
        $ENV{DPDK_INSTALL}
        $ENV{RTE_SDK}/install
    )
    
    foreach(path ${DPDK_SEARCH_PATHS})
        if(EXISTS ${path}/lib/x86_64-linux-gnu/pkgconfig/libdpdk.pc)
            set(ENV{PKG_CONFIG_PATH} "${path}/lib/x86_64-linux-gnu/pkgconfig:$ENV{PKG_CONFIG_PATH}")
            pkg_check_modules(DPDK libdpdk)
            if(DPDK_FOUND)
                message(STATUS "Found DPDK at ${path}")
                break()
            endif()
        endif()
    endforeach()
endif()

if(NOT DPDK_FOUND)
    message(WARNING "DPDK not found via pkg-config. Trying manual configuration...")
    
    # Manual DPDK configuration as fallback
    set(DPDK_INCLUDE_DIRS "/opt/dpdk/dpdk-23.11/install/include")
    set(DPDK_LIBRARY_DIRS "/opt/dpdk/dpdk-23.11/install/lib/x86_64-linux-gnu")
    
    if(EXISTS ${DPDK_INCLUDE_DIRS} AND EXISTS ${DPDK_LIBRARY_DIRS})
        set(DPDK_FOUND TRUE)
        set(DPDK_LIBRARIES
            -L${DPDK_LIBRARY_DIRS}
            -Wl,--whole-archive
            -ldpdk
            -Wl,--no-whole-archive
            -lnuma -ldl -lpcap -lm
        )
        message(STATUS "Using manual DPDK configuration")
    else()
        message(FATAL_ERROR "DPDK not found. Please install DPDK or set DPDK_INSTALL environment variable")
    endif()
endif()

# CUDA (optional) - prefer PyTorch's packaged CUDA runtime to avoid version mismatch
set(TORCH_CUDA_LIB_DIR $ENV{TORCH_CUDA_LIB_DIR})
if(TORCH_CUDA_LIB_DIR)
    message(STATUS "Using CUDA runtime from PyTorch: ${TORCH_CUDA_LIB_DIR}")
    add_definitions(-DGENIE_CUDA_SUPPORT)
    # Headers can come from system CUDA; runtime will be resolved from PyTorch's lib dir
    find_path(CUDA_INCLUDE_DIR cuda_runtime.h PATHS /usr/local/cuda/include /usr/include /usr/local/include)
    if(CUDA_INCLUDE_DIR)
        include_directories(${CUDA_INCLUDE_DIR})
    endif()
else()
    find_package(CUDA)
    if(CUDA_FOUND)
        add_definitions(-DGENIE_CUDA_SUPPORT)
        include_directories(${CUDA_INCLUDE_DIRS})
    endif()
endif()

# Find JSON library
find_package(nlohmann_json 3.2.0 REQUIRED)

# Source files
set(SOURCES
    genie_data_plane.cpp
    genie_dpdk_thread_model.cpp
    genie_zero_copy_transport.cpp
    genie_c_api.cpp
    genie_kcp_wrapper.cpp
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

# Add library directories if manually configured
if(DPDK_LIBRARY_DIRS)
    target_link_directories(genie_data_plane PRIVATE ${DPDK_LIBRARY_DIRS})
endif()

# Link libraries
target_link_libraries(genie_data_plane
    ${DPDK_LIBRARIES}
    nlohmann_json::nlohmann_json
    ${KCP_LIBRARIES}
    ${SPDK_LIBRARIES}
    pthread
    numa
    dl
)

if(TORCH_CUDA_LIB_DIR)
    # Link against PyTorch's packaged cudart to match the Python runtime
    target_link_directories(genie_data_plane PRIVATE ${TORCH_CUDA_LIB_DIR})
    target_link_libraries(genie_data_plane cudart)
elseif(CUDA_FOUND)
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

# Removed test_dpdk_threading and test_dpdk_simple demo executables
# These were demos, not actual tests

install(FILES genie_data_plane.hpp
    DESTINATION include/genie
)

# Test executable (optional)
if(BUILD_TESTS)
    add_executable(test_data_plane test_data_plane.cpp)
    # Ensure DPDK library search path is available for the test executable
    target_link_directories(test_data_plane PRIVATE /opt/dpdk/dpdk-23.11/install/lib/x86_64-linux-gnu)
    target_link_libraries(test_data_plane
        genie_data_plane
        ${DPDK_LIBRARIES}
        pthread
        numa
        dl
    )
endif()

# Print configuration
message(STATUS "DPDK include dirs: ${DPDK_INCLUDE_DIRS}")
message(STATUS "DPDK libraries: ${DPDK_LIBRARIES}")
message(STATUS "CUDA support: ${CUDA_FOUND}")
if(CUDA_FOUND)
    message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
endif()