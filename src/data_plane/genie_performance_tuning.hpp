/**
 * Performance Tuning Module for Genie Data Plane
 * 
 * Implements:
 * - NUMA-aware memory allocation and thread placement
 * - CPU cache prefetching strategies
 * - Hardware offload capabilities (checksum, TSO, etc.)
 * - Memory pool optimization
 * - CPU frequency scaling control
 */

#pragma once

#include <rte_ethdev.h>
#include <rte_malloc.h>
#include <rte_mempool.h>
#include <rte_prefetch.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <numa.h>
#include <vector>
#include <memory>
#include <atomic>
#include <map>

namespace genie {
namespace data_plane {

/**
 * NUMA-aware memory allocator
 */
class NUMAAllocator {
public:
    NUMAAllocator();
    ~NUMAAllocator();
    
    // Initialize NUMA subsystem
    bool initialize();
    
    // Allocate memory on specific NUMA node
    void* allocate(size_t size, int numa_node = -1);
    void* allocate_huge(size_t size, int numa_node = -1);
    
    // Free NUMA-allocated memory
    void free(void* ptr);
    
    // Get optimal NUMA node for a given CPU core
    int get_numa_node_for_core(unsigned core_id) const;
    
    // Get NUMA distance between nodes
    int get_numa_distance(int node1, int node2) const;
    
    // Print NUMA topology
    void print_topology() const;
    
private:
    bool initialized_;
    int num_nodes_;
    std::vector<int> cpu_to_node_;
    std::vector<std::vector<int>> distance_matrix_;
    
    void discover_topology();
};

/**
 * Prefetch strategies for different workloads
 */
class PrefetchManager {
public:
    enum Strategy {
        NONE = 0,
        LINEAR,      // Sequential access pattern
        STRIDED,     // Regular stride pattern
        RANDOM,      // Random access pattern
        ADAPTIVE     // Auto-detect and adapt
    };
    
    PrefetchManager();
    
    // Set prefetch strategy
    void set_strategy(Strategy strategy, int stride = 64);
    
    // Prefetch for packet processing
    static inline void prefetch_packet(struct rte_mbuf* mbuf) {
        rte_prefetch0(mbuf);
        rte_prefetch0(rte_pktmbuf_mtod(mbuf, void*));
    }
    
    // Prefetch for batch processing
    static inline void prefetch_batch(struct rte_mbuf** mbufs, uint16_t nb_pkts) {
        // Prefetch first wave
        for (uint16_t i = 0; i < 4 && i < nb_pkts; i++) {
            rte_prefetch0(mbufs[i]);
        }
        
        // Process with prefetch ahead
        for (uint16_t i = 0; i < nb_pkts; i++) {
            // Prefetch next packet
            if (i + 4 < nb_pkts) {
                rte_prefetch0(mbufs[i + 4]);
                rte_prefetch0(rte_pktmbuf_mtod(mbufs[i + 4], void*));
            }
            
            // Current packet already prefetched
            // Process it here...
        }
    }
    
    // Prefetch GPU memory
    static inline void prefetch_gpu_memory(void* gpu_ptr, size_t size) {
        const size_t cache_line = 64;
        char* ptr = static_cast<char*>(gpu_ptr);
        
        for (size_t i = 0; i < size; i += cache_line) {
            rte_prefetch0(ptr + i);
        }
    }
    
    // Adaptive prefetch based on access pattern
    void record_access(void* addr);
    void* predict_next_access();
    
private:
    Strategy strategy_;
    int stride_;
    
    // For adaptive strategy
    std::vector<void*> access_history_;
    size_t history_index_;
    static constexpr size_t HISTORY_SIZE = 1024;
    
    void analyze_pattern();
};

/**
 * Hardware offload configuration
 */
class HardwareOffloadManager {
public:
    struct OffloadCapabilities {
        // RX offloads
        bool rx_checksum_ipv4;
        bool rx_checksum_tcp;
        bool rx_checksum_udp;
        bool rx_vlan_strip;
        bool rx_lro;  // Large Receive Offload
        
        // TX offloads
        bool tx_checksum_ipv4;
        bool tx_checksum_tcp;
        bool tx_checksum_udp;
        bool tx_vlan_insert;
        bool tx_tso;  // TCP Segmentation Offload
        bool tx_ufo;  // UDP Fragmentation Offload
        
        // Advanced offloads
        bool rss;      // Receive Side Scaling
        bool fdir;     // Flow Director
        bool vmdq;     // Virtual Machine Device Queue
    };
    
    HardwareOffloadManager();
    
    // Detect available offloads for a port
    bool detect_capabilities(uint16_t port_id);
    
    // Configure offloads for a port
    bool configure_offloads(uint16_t port_id, const OffloadCapabilities& requested);
    
    // Enable specific offloads
    bool enable_checksum_offload(uint16_t port_id);
    bool enable_tso(uint16_t port_id, uint16_t mss = 1460);
    bool enable_rss(uint16_t port_id, uint16_t nb_queues);
    
    // Get current offload status
    OffloadCapabilities get_enabled_offloads(uint16_t port_id) const;
    
    // Print offload capabilities
    void print_capabilities(uint16_t port_id) const;
    
private:
    std::map<uint16_t, OffloadCapabilities> port_capabilities_;
    std::map<uint16_t, OffloadCapabilities> enabled_offloads_;
    
    uint64_t capabilities_to_dpdk_flags(const OffloadCapabilities& caps, bool is_rx) const;
};

/**
 * Memory pool optimizer
 */
class MemoryPoolOptimizer {
public:
    struct PoolConfig {
        size_t element_size;
        size_t num_elements;
        size_t cache_size;
        int numa_node;
        bool use_huge_pages;
    };
    
    MemoryPoolOptimizer();
    ~MemoryPoolOptimizer();
    
    // Create optimized memory pool
    struct rte_mempool* create_pool(const char* name, const PoolConfig& config);
    
    // Optimize existing pool
    bool optimize_pool(struct rte_mempool* pool);
    
    // Monitor pool usage
    struct PoolStats {
        size_t available;
        size_t in_use;
        size_t cache_count;
        double utilization;
        uint64_t get_success;
        uint64_t get_fail;
    };
    
    PoolStats get_stats(struct rte_mempool* pool) const;
    
    // Auto-tune pool size based on usage
    bool auto_tune(struct rte_mempool* pool, double target_utilization = 0.8);
    
private:
    std::map<struct rte_mempool*, PoolConfig> pool_configs_;
    std::map<struct rte_mempool*, PoolStats> pool_history_;
    
    size_t calculate_optimal_size(const PoolStats& stats, double target_util) const;
};

/**
 * CPU frequency governor
 */
class CPUFrequencyManager {
public:
    enum Governor {
        PERFORMANCE,    // Maximum frequency
        POWERSAVE,      // Minimum frequency
        ONDEMAND,       // Dynamic based on load
        USERSPACE       // Manual control
    };
    
    CPUFrequencyManager();
    
    // Set governor for specific cores
    bool set_governor(unsigned core_id, Governor gov);
    bool set_governor_all(Governor gov);
    
    // Manual frequency control (requires USERSPACE governor)
    bool set_frequency(unsigned core_id, uint64_t freq_khz);
    
    // Get current frequency
    uint64_t get_frequency(unsigned core_id) const;
    
    // Turbo boost control
    bool enable_turbo_boost(unsigned core_id);
    bool disable_turbo_boost(unsigned core_id);
    
    // C-state control (for latency optimization)
    bool disable_c_states(unsigned core_id);
    
private:
    std::string governor_to_string(Governor gov) const;
    bool write_sysfs(const std::string& path, const std::string& value);
    std::string read_sysfs(const std::string& path) const;
};

/**
 * Main performance tuning coordinator
 */
class PerformanceTuner {
public:
    struct TuningConfig {
        // NUMA settings
        bool numa_aware = true;
        bool bind_threads = true;
        
        // Prefetch settings
        PrefetchManager::Strategy prefetch_strategy = PrefetchManager::ADAPTIVE;
        
        // Hardware offloads
        bool enable_offloads = true;
        bool enable_tso = true;
        bool enable_checksum = true;
        bool enable_rss = true;
        
        // Memory optimization
        bool optimize_pools = true;
        double pool_target_utilization = 0.8;
        
        // CPU settings
        CPUFrequencyManager::Governor cpu_governor = CPUFrequencyManager::PERFORMANCE;
        bool disable_cpu_idle = true;
        bool enable_turbo = true;
    };
    
    PerformanceTuner();
    ~PerformanceTuner();
    
    // Initialize with configuration
    bool initialize(const TuningConfig& config);
    
    // Apply all tunings
    bool apply_tunings(uint16_t port_id);
    
    // Runtime optimization
    void optimize_runtime();
    
    // Get current performance metrics
    struct PerformanceMetrics {
        double packets_per_second;
        double bytes_per_second;
        double average_latency_ns;
        double cpu_utilization;
        double cache_hit_rate;
        uint64_t cache_misses;
        uint64_t branch_mispredicts;
    };
    
    PerformanceMetrics get_metrics() const;
    
    // Print performance report
    void print_report() const;
    
private:
    TuningConfig config_;
    std::unique_ptr<NUMAAllocator> numa_allocator_;
    std::unique_ptr<PrefetchManager> prefetch_manager_;
    std::unique_ptr<HardwareOffloadManager> offload_manager_;
    std::unique_ptr<MemoryPoolOptimizer> pool_optimizer_;
    std::unique_ptr<CPUFrequencyManager> cpu_manager_;
    
    // Performance counters
    std::atomic<uint64_t> packet_count_{0};
    std::atomic<uint64_t> byte_count_{0};
    std::atomic<uint64_t> cycle_count_{0};
    
    void setup_numa_bindings();
    void configure_hardware_offloads(uint16_t port_id);
    void optimize_memory_pools();
    void tune_cpu_settings();
};

/**
 * Helper functions for performance optimization
 */
namespace PerfHelpers {
    // Force inline for hot path functions
    #define ALWAYS_INLINE __attribute__((always_inline)) inline
    
    // Likely/unlikely branch hints
    #define likely(x) __builtin_expect(!!(x), 1)
    #define unlikely(x) __builtin_expect(!!(x), 0)
    
    // Cache line alignment
    #define CACHE_LINE_SIZE 64
    #define __cache_aligned __attribute__((aligned(CACHE_LINE_SIZE)))
    
    // Prefetch distance tuning
    static constexpr int PREFETCH_OFFSET = 3;
    
    // Batch size optimization
    static constexpr uint16_t OPTIMAL_BATCH_SIZE = 32;
    
    // CPU pause for spinlock optimization
    ALWAYS_INLINE void cpu_pause() {
        rte_pause();
    }
    
    // Memory barrier
    ALWAYS_INLINE void memory_barrier() {
        rte_mb();
    }
    
    // Read TSC with serialization
    ALWAYS_INLINE uint64_t rdtsc_precise() {
        rte_mb();
        return rte_rdtsc_precise();
    }
}

} // namespace data_plane
} // namespace genie














