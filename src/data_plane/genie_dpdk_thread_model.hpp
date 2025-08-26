/**
 * DPDK Native Thread Model using lcore and service cores
 * 
 * This implementation uses DPDK's built-in threading model which is
 * the standard approach in DPDK applications.
 */

#pragma once

#include <rte_lcore.h>
#include <rte_launch.h>
#include <rte_service.h>
#include <rte_service_component.h>
#include <rte_cycles.h>
#include <rte_ring.h>
#include <rte_mbuf.h>
#include <rte_ethdev.h>
#include <atomic>
#include <functional>
#include <vector>
#include <memory>

namespace genie {
namespace data_plane {

// DPDK uses plain C function pointers for lcores
// using lcore_function_t = int (*)(void*);

/**
 * Per-lcore statistics
 */
struct LcoreStats {
    uint64_t packets_processed = 0;
    uint64_t bytes_processed = 0;
    uint64_t bursts = 0;
    uint64_t idle_cycles = 0;
    uint64_t busy_cycles = 0;
};

/**
 * Lcore configuration
 */
struct LcoreConfig {
    unsigned lcore_id;
    enum LcoreRole {
        RX_LCORE,
        TX_LCORE,
        WORKER_LCORE,
        SERVICE_LCORE
    } role;
    uint16_t port_id;
    uint16_t queue_id;
    struct rte_ring* rx_ring;
    struct rte_ring* tx_ring;
    void* user_data;
};

/**
 * DPDK Native Thread Manager
 * 
 * Uses DPDK's lcore management for threading
 */
class DPDKThreadManager {
public:
    DPDKThreadManager();
    ~DPDKThreadManager();
    
    // Initialize with available lcores
    bool initialize();
    
    // Assign lcore to specific role
    bool assign_lcore(unsigned lcore_id, LcoreConfig::LcoreRole role,
                      uint16_t port_id = 0, uint16_t queue_id = 0);
    
    // Launch all configured lcores
    bool launch_all();
    
    // Wait for all lcores to finish
    void wait_all();
    
    // Stop all lcores
    void stop_all();
    
    // Get statistics
    void print_stats() const;
    
    // Static lcore functions (DPDK requires C-style functions)
    static int rx_lcore_main(void* arg);
    static int tx_lcore_main(void* arg);
    static int worker_lcore_main(void* arg);
    
private:
    std::vector<LcoreConfig> lcore_configs_;
    std::vector<LcoreStats> lcore_stats_;
    std::atomic<bool> force_quit_{false};
    
    // Inter-lcore communication rings
    struct rte_ring* rx_to_worker_ring_;
    struct rte_ring* worker_to_tx_ring_;
    
public:
    // Port configuration constants
    static constexpr uint16_t RX_RING_SIZE = 1024;
    static constexpr uint16_t TX_RING_SIZE = 1024;
    static constexpr uint16_t NUM_MBUFS = 8191;
    static constexpr uint16_t MBUF_CACHE_SIZE = 250;
    static constexpr uint16_t BURST_SIZE = 32;

private:
    
    bool create_rings();
    void destroy_rings();
};

/**
 * DPDK Service Core Implementation
 * 
 * Uses DPDK's service core framework for background tasks
 */
class ServiceCoreManager {
public:
    ServiceCoreManager();
    ~ServiceCoreManager();
    
    // Register a service
    bool register_service(const char* name,
                         rte_service_func service_func,
                         void* user_data);
    
    // Map service to lcore
    bool map_service_to_lcore(uint32_t service_id, uint32_t lcore_id);
    
    // Start service cores
    bool start_service_cores();
    
    // Stop service cores
    void stop_service_cores();
    
    // Example service functions
    static int32_t stats_service(void* args);
    static int32_t timeout_service(void* args);
    
private:
    std::vector<uint32_t> service_ids_;
    std::vector<uint32_t> service_lcores_;
};

/**
 * Simplified DPDK Application using native threading
 */
class DPDKApplication {
public:
    DPDKApplication();
    ~DPDKApplication();
    
    // Initialize DPDK EAL
    bool init_eal(int argc, char* argv[]);
    
    // Initialize ports
    bool init_ports(uint16_t nb_ports);
    
    // Setup lcore assignments
    bool setup_lcores();
    
    // Run application
    int run();
    
    // Cleanup
    void cleanup();
    
private:
    std::unique_ptr<DPDKThreadManager> thread_manager_;
    std::unique_ptr<ServiceCoreManager> service_manager_;
    struct rte_mempool* mbuf_pool_;
    uint16_t nb_ports_;
    
    // Port initialization
    bool port_init(uint16_t port, struct rte_mempool* mbuf_pool);
    
    // Default port configuration
    static const struct rte_eth_conf port_conf_default;
};

/**
 * Helper functions for DPDK thread management
 */
class DPDKThreadHelpers {
public:
    // Get optimal lcore for role
    static int get_lcore_for_role(LcoreConfig::LcoreRole role);
    
    // Check if lcore is available
    static bool is_lcore_available(unsigned lcore_id);
    
    // Get NUMA node for lcore
    static int get_numa_node(unsigned lcore_id);
    
    // Print lcore layout
    static void print_lcore_layout();
};

} // namespace data_plane
} // namespace genie
