cmake_minimum_required(VERSION 3.16)
project(genie_data_plane)

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
    genie_zero_copy_transport.cpp
    genie_performance_tuning.cpp
    genie_monitoring.cpp
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