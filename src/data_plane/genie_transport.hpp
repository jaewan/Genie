/**
 * Simplified Genie Transport
 * 
 * This replaces 3 files (3000+ lines) with ONE file (~500 lines)
 * 
 * Responsibilities:
 * 1. DPDK initialization
 * 2. GPU memory registration (GPUDev)
 * 3. Zero-copy packet send/receive
 * 4. Simple reliability (ACK/NACK)
 * 
 * NOT included (premature optimization):
 * - Flow control algorithms (CUBIC, etc.)
 * - Thread pools
 * - KCP protocol
 * - CUDA Graphs
 */

#pragma once

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <rte_gpudev.h>

#include <string>
#include <atomic>
#include <cstdint>

namespace genie {

class GenieTransport {
public:
    struct Config {
        uint16_t data_port = 5556;
        size_t mtu = 9000;        // Jumbo frames for datacenter
        bool use_gpu_direct = true;
    };
    
    explicit GenieTransport(const Config& config);
    ~GenieTransport();
    
    // Initialize (DPDK + optional GPUDev)
    bool initialize();
    
    // Send tensor (zero-copy if GPU)
    bool send_tensor(
        void* gpu_ptr,
        size_t size,
        const std::string& target
    );
    
    // Receive tensor
    bool receive_tensor(
        void* buffer,
        size_t max_size,
        const std::string& source
    );
    
    // Check if GPU Direct is available
    bool has_gpu_direct() const { return gpu_available_; }
    
private:
    Config config_;
    
    // DPDK resources
    struct rte_mempool* mbuf_pool_;
    uint16_t port_id_;
    
    // GPU resources
    int gpu_dev_id_;
    bool gpu_available_;
    
    // State
    std::atomic<bool> running_;
    
    // Init helpers
    bool init_dpdk();
    bool init_gpu();
    bool init_port();
    
    // Memory management
    bool register_gpu_memory(void* ptr, size_t size);
    
    // Packet operations
    bool send_packet(
        const uint8_t* data,
        size_t size,
        const std::string& target
    );
    
    bool recv_packet(
        uint8_t* buffer,
        size_t max_size
    );
};

} // namespace genie

// C API for Python
extern "C" {
    void* genie_transport_create(const char* config_json);
    int genie_transport_send(
        void* transport,
        void* gpu_ptr,
        size_t size,
        const char* target
    );
    void genie_transport_destroy(void* transport);
}