/**
 * Test program for Genie Data Plane
 * 
 * Simple test to verify C++ data plane functionality
 */

#include "genie_data_plane.hpp"
#include <iostream>
#include <chrono>
#include <thread>
#include <cstring>

using namespace genie::data_plane;

void print_statistics(const DataPlaneStats& stats) {
    std::cout << "\n=== Data Plane Statistics ===" << std::endl;
    std::cout << "Packets Sent: " << stats.packets_sent.load() << std::endl;
    std::cout << "Packets Received: " << stats.packets_received.load() << std::endl;
    std::cout << "Bytes Sent: " << stats.bytes_sent.load() << std::endl;
    std::cout << "Bytes Received: " << stats.bytes_received.load() << std::endl;
    std::cout << "Packets Dropped: " << stats.packets_dropped.load() << std::endl;
    std::cout << "Retransmissions: " << stats.retransmissions.load() << std::endl;
    std::cout << "Active Transfers: " << stats.active_transfers.load() << std::endl;
    std::cout << "============================\n" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Genie Data Plane Test" << std::endl;
    std::cout << "=====================" << std::endl;
    
    // Create configuration
    DataPlaneConfig config;
    config.eal_args = {
        "test-data-plane",
        "-c", "0x1",  // Use only core 0 for testing
        "-n", "2",    // 2 memory channels
        "--huge-dir", "/mnt/huge",
        "--proc-type", "primary"
    };
    config.local_ip = "192.168.1.100";
    config.local_mac = "aa:bb:cc:dd:ee:01";
    config.data_port = 5556;
    config.mempool_size = 1024;  // Smaller for testing
    
    // Create data plane instance
    std::cout << "Creating data plane..." << std::endl;
    GenieDataPlane data_plane(config);
    
    // Initialize
    std::cout << "Initializing data plane..." << std::endl;
    if (!data_plane.initialize()) {
        std::cerr << "Failed to initialize data plane" << std::endl;
        return 1;
    }
    
    // Start
    std::cout << "Starting data plane..." << std::endl;
    if (!data_plane.start()) {
        std::cerr << "Failed to start data plane" << std::endl;
        return 1;
    }
    
    std::cout << "Data plane started successfully!" << std::endl;
    
    // Configure target node
    std::cout << "Configuring target node..." << std::endl;
    data_plane.set_target_node("test-target", "192.168.1.101", "aa:bb:cc:dd:ee:02");
    
    // Test GPU memory registration
    std::cout << "Testing GPU memory registration..." << std::endl;
    
    // Simulate GPU memory (use regular memory for testing)
    size_t test_size = 4096;
    void* test_memory = malloc(test_size);
    if (!test_memory) {
        std::cerr << "Failed to allocate test memory" << std::endl;
        return 1;
    }
    
    // Fill with test data
    memset(test_memory, 0xAB, test_size);
    
    GPUMemoryHandle gpu_handle;
    if (data_plane.register_gpu_memory(test_memory, test_size, gpu_handle)) {
        std::cout << "GPU memory registered successfully" << std::endl;
        std::cout << "  GPU Ptr: " << gpu_handle.gpu_ptr << std::endl;
        std::cout << "  IOVA: 0x" << std::hex << gpu_handle.iova << std::dec << std::endl;
        std::cout << "  Size: " << gpu_handle.size << std::endl;
        std::cout << "  Registered: " << (gpu_handle.is_registered ? "Yes" : "No (fallback)") << std::endl;
    } else {
        std::cerr << "Failed to register GPU memory" << std::endl;
        free(test_memory);
        return 1;
    }
    
    // Test tensor send (will fail without actual network setup, but tests the path)
    std::cout << "\nTesting tensor send..." << std::endl;
    bool send_result = data_plane.send_tensor(
        "test-transfer-123",
        "test-tensor-456", 
        test_memory,
        test_size,
        "test-target"
    );
    
    if (send_result) {
        std::cout << "Tensor send initiated successfully" << std::endl;
    } else {
        std::cout << "Tensor send failed (expected without network setup)" << std::endl;
    }
    
    // Print initial statistics
    DataPlaneStats stats;
    data_plane.get_statistics(stats);
    print_statistics(stats);
    
    // Run for a short time to test packet processing loops
    std::cout << "Running data plane for 5 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    // Print final statistics
    data_plane.get_statistics(stats);
    print_statistics(stats);
    
    // Test active transfers query
    std::cout << "Active transfers:" << std::endl;
    auto active_transfers = data_plane.get_active_transfers();
    for (const auto& transfer_id : active_transfers) {
        std::cout << "  - " << transfer_id << std::endl;
        
        TransferContext context;
        bool found = false;
        data_plane.get_transfer_status(transfer_id, context, found);
        if (found) {
            std::cout << "    Tensor ID: " << context.tensor_id << std::endl;
            std::cout << "    Target: " << context.target_node << std::endl;
            std::cout << "    Size: " << context.total_size << std::endl;
            std::cout << "    Packets Sent: " << context.packets_sent.load() << std::endl;
        }
    }
    
    // Cleanup
    std::cout << "\nCleaning up..." << std::endl;
    data_plane.unregister_gpu_memory(gpu_handle);
    free(test_memory);
    
    // Stop data plane
    std::cout << "Stopping data plane..." << std::endl;
    data_plane.stop();
    
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}

// Test C interface as well
void test_c_interface() {
    std::cout << "\n=== Testing C Interface ===" << std::endl;
    
    // Create configuration JSON
    const char* config_json = R"({
        "eal_args": ["test-c-interface", "-c", "0x1", "-n", "2", "--huge-dir", "/mnt/huge"],
        "local_ip": "192.168.1.100",
        "local_mac": "aa:bb:cc:dd:ee:01",
        "data_port": 5556,
        "mempool_size": 512
    })";
    
    // Create data plane
    void* data_plane = genie_data_plane_create(config_json);
    if (!data_plane) {
        std::cerr << "Failed to create data plane via C interface" << std::endl;
        return;
    }
    
    std::cout << "Data plane created via C interface" << std::endl;
    
    // Initialize
    if (genie_data_plane_initialize(data_plane) != 0) {
        std::cerr << "Failed to initialize data plane via C interface" << std::endl;
        genie_data_plane_destroy(data_plane);
        return;
    }
    
    std::cout << "Data plane initialized via C interface" << std::endl;
    
    // Start
    if (genie_data_plane_start(data_plane) != 0) {
        std::cerr << "Failed to start data plane via C interface" << std::endl;
        genie_data_plane_destroy(data_plane);
        return;
    }
    
    std::cout << "Data plane started via C interface" << std::endl;
    
    // Configure target node
    genie_set_target_node(data_plane, "c-target", "192.168.1.102", "aa:bb:cc:dd:ee:03");
    
    // Test memory registration
    void* test_mem = malloc(2048);
    uint64_t iova = 0;
    
    if (genie_register_gpu_memory(data_plane, test_mem, 2048, &iova) == 0) {
        std::cout << "Memory registered via C interface, IOVA: 0x" << std::hex << iova << std::dec << std::endl;
    }
    
    // Get statistics
    char stats_buffer[1024];
    genie_get_statistics(data_plane, stats_buffer, sizeof(stats_buffer));
    std::cout << "Statistics: " << stats_buffer << std::endl;
    
    // Cleanup
    genie_unregister_gpu_memory(data_plane, test_mem);
    free(test_mem);
    
    genie_data_plane_stop(data_plane);
    genie_data_plane_destroy(data_plane);
    
    std::cout << "C interface test completed" << std::endl;
}
