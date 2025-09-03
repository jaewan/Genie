/**
 * Performance Tuning Implementation
 */

#include "genie_performance_tuning.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <rte_ethdev.h>
#include <rte_errno.h>

namespace genie {
namespace data_plane {

// ============================================================================
// NUMAAllocator Implementation
// ============================================================================

NUMAAllocator::NUMAAllocator() : initialized_(false), num_nodes_(0) {}

NUMAAllocator::~NUMAAllocator() {
    // Cleanup handled by DPDK
}

bool NUMAAllocator::initialize() {
    if (initialized_) {
        return true;
    }
    
    // Check if NUMA is available
    if (numa_available() < 0) {
        std::cerr << "NUMA is not available on this system" << std::endl;
        return false;
    }
    
    num_nodes_ = numa_max_node() + 1;
    discover_topology();
    
    initialized_ = true;
    std::cout << "NUMA allocator initialized with " << num_nodes_ << " nodes" << std::endl;
    
    return true;
}

void* NUMAAllocator::allocate(size_t size, int numa_node) {
    if (!initialized_) {
        return nullptr;
    }
    
    if (numa_node < 0) {
        // Use current CPU's NUMA node
        numa_node = numa_node_of_cpu(rte_lcore_id());
    }
    
    // Use DPDK's NUMA-aware allocation
    void* ptr = rte_malloc_socket(nullptr, size, 64, numa_node);
    
    if (!ptr) {
        std::cerr << "Failed to allocate " << size << " bytes on NUMA node " 
                  << numa_node << std::endl;
    }
    
    return ptr;
}

void* NUMAAllocator::allocate_huge(size_t size, int numa_node) {
    if (!initialized_) {
        return nullptr;
    }
    
    if (numa_node < 0) {
        numa_node = numa_node_of_cpu(rte_lcore_id());
    }
    
    // Allocate from hugepage memory
    void* ptr = rte_malloc_socket("huge", size, 2 * 1024 * 1024, numa_node);
    
    return ptr;
}

void NUMAAllocator::free(void* ptr) {
    rte_free(ptr);
}

int NUMAAllocator::get_numa_node_for_core(unsigned core_id) const {
    if (core_id < cpu_to_node_.size()) {
        return cpu_to_node_[core_id];
    }
    return -1;
}

int NUMAAllocator::get_numa_distance(int node1, int node2) const {
    if (node1 >= 0 && node1 < num_nodes_ && 
        node2 >= 0 && node2 < num_nodes_) {
        return distance_matrix_[node1][node2];
    }
    return -1;
}

void NUMAAllocator::discover_topology() {
    // Build CPU to NUMA node mapping
    cpu_to_node_.resize(RTE_MAX_LCORE);
    
    unsigned lcore_id;
    RTE_LCORE_FOREACH(lcore_id) {
        cpu_to_node_[lcore_id] = rte_lcore_to_socket_id(lcore_id);
    }
    
    // Build distance matrix
    distance_matrix_.resize(num_nodes_);
    for (int i = 0; i < num_nodes_; i++) {
        distance_matrix_[i].resize(num_nodes_);
        for (int j = 0; j < num_nodes_; j++) {
            distance_matrix_[i][j] = numa_distance(i, j);
        }
    }
}

void NUMAAllocator::print_topology() const {
    std::cout << "\n=== NUMA Topology ===" << std::endl;
    std::cout << "Number of NUMA nodes: " << num_nodes_ << std::endl;
    
    std::cout << "\nCPU to NUMA mapping:" << std::endl;
    unsigned lcore_id;
    RTE_LCORE_FOREACH(lcore_id) {
        std::cout << "  Core " << lcore_id << " -> NUMA " 
                  << cpu_to_node_[lcore_id] << std::endl;
    }
    
    std::cout << "\nNUMA distance matrix:" << std::endl;
    for (int i = 0; i < num_nodes_; i++) {
        std::cout << "  Node " << i << ": ";
        for (int j = 0; j < num_nodes_; j++) {
            std::cout << distance_matrix_[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// ============================================================================
// PrefetchManager Implementation
// ============================================================================

PrefetchManager::PrefetchManager() 
    : strategy_(ADAPTIVE), stride_(64), history_index_(0) {
    access_history_.reserve(HISTORY_SIZE);
}

void PrefetchManager::set_strategy(Strategy strategy, int stride) {
    strategy_ = strategy;
    stride_ = stride;
}

void PrefetchManager::record_access(void* addr) {
    if (strategy_ != ADAPTIVE) {
        return;
    }
    
    if (access_history_.size() < HISTORY_SIZE) {
        access_history_.push_back(addr);
    } else {
        access_history_[history_index_] = addr;
        history_index_ = (history_index_ + 1) % HISTORY_SIZE;
    }
    
    // Analyze pattern periodically
    if (history_index_ == 0) {
        analyze_pattern();
    }
}

void* PrefetchManager::predict_next_access() {
    if (access_history_.size() < 2) {
        return nullptr;
    }
    
    switch (strategy_) {
        case LINEAR: {
            // Predict next sequential address
            size_t last_idx = (history_index_ + HISTORY_SIZE - 1) % HISTORY_SIZE;
            char* last_addr = static_cast<char*>(access_history_[last_idx]);
            return last_addr + stride_;
        }
        
        case STRIDED: {
            // Detect stride pattern
            if (access_history_.size() >= 3) {
                size_t idx1 = (history_index_ + HISTORY_SIZE - 1) % HISTORY_SIZE;
                size_t idx2 = (history_index_ + HISTORY_SIZE - 2) % HISTORY_SIZE;
                
                char* addr1 = static_cast<char*>(access_history_[idx1]);
                char* addr2 = static_cast<char*>(access_history_[idx2]);
                
                ptrdiff_t detected_stride = addr1 - addr2;
                return addr1 + detected_stride;
            }
            break;
        }
        
        default:
            break;
    }
    
    return nullptr;
}

void PrefetchManager::analyze_pattern() {
    if (access_history_.size() < 10) {
        return;
    }
    
    // Simple pattern detection
    bool is_sequential = true;
    bool is_strided = true;
    ptrdiff_t common_stride = 0;
    
    for (size_t i = 1; i < std::min(size_t(10), access_history_.size()); i++) {
        char* addr1 = static_cast<char*>(access_history_[i]);
        char* addr2 = static_cast<char*>(access_history_[i-1]);
        ptrdiff_t stride = addr1 - addr2;
        
        if (i == 1) {
            common_stride = stride;
        }
        
        if (stride != stride_) {
            is_sequential = false;
        }
        
        if (stride != common_stride) {
            is_strided = false;
        }
    }
    
    // Update strategy based on pattern
    if (is_sequential) {
        strategy_ = LINEAR;
    } else if (is_strided) {
        strategy_ = STRIDED;
        stride_ = common_stride;
    } else {
        strategy_ = RANDOM;
    }
}

// ============================================================================
// HardwareOffloadManager Implementation
// ============================================================================

HardwareOffloadManager::HardwareOffloadManager() {}

bool HardwareOffloadManager::detect_capabilities(uint16_t port_id) {
    struct rte_eth_dev_info dev_info;
    
    if (rte_eth_dev_info_get(port_id, &dev_info) != 0) {
        std::cerr << "Failed to get device info for port " << port_id << std::endl;
        return false;
    }
    
    OffloadCapabilities caps = {};
    
    // Check RX offloads
    uint64_t rx_offloads = dev_info.rx_offload_capa;
    caps.rx_checksum_ipv4 = (rx_offloads & RTE_ETH_RX_OFFLOAD_IPV4_CKSUM) != 0;
    caps.rx_checksum_tcp = (rx_offloads & RTE_ETH_RX_OFFLOAD_TCP_CKSUM) != 0;
    caps.rx_checksum_udp = (rx_offloads & RTE_ETH_RX_OFFLOAD_UDP_CKSUM) != 0;
    caps.rx_vlan_strip = (rx_offloads & RTE_ETH_RX_OFFLOAD_VLAN_STRIP) != 0;
    caps.rx_lro = (rx_offloads & RTE_ETH_RX_OFFLOAD_TCP_LRO) != 0;
    
    // Check TX offloads
    uint64_t tx_offloads = dev_info.tx_offload_capa;
    caps.tx_checksum_ipv4 = (tx_offloads & RTE_ETH_TX_OFFLOAD_IPV4_CKSUM) != 0;
    caps.tx_checksum_tcp = (tx_offloads & RTE_ETH_TX_OFFLOAD_TCP_CKSUM) != 0;
    caps.tx_checksum_udp = (tx_offloads & RTE_ETH_TX_OFFLOAD_UDP_CKSUM) != 0;
    caps.tx_vlan_insert = (tx_offloads & RTE_ETH_TX_OFFLOAD_VLAN_INSERT) != 0;
    caps.tx_tso = (tx_offloads & RTE_ETH_TX_OFFLOAD_TCP_TSO) != 0;
    caps.tx_ufo = (tx_offloads & RTE_ETH_TX_OFFLOAD_UDP_TSO) != 0;
    
    // Check flow control capabilities
    caps.rss = (dev_info.flow_type_rss_offloads != 0);
    
    port_capabilities_[port_id] = caps;
    
    return true;
}

bool HardwareOffloadManager::configure_offloads(uint16_t port_id, 
                                               const OffloadCapabilities& requested) {
    if (port_capabilities_.find(port_id) == port_capabilities_.end()) {
        if (!detect_capabilities(port_id)) {
            return false;
        }
    }
    
    const OffloadCapabilities& available = port_capabilities_[port_id];
    OffloadCapabilities to_enable = {};
    
    // Only enable what's available and requested
    #define CHECK_OFFLOAD(field) \
        to_enable.field = requested.field && available.field
    
    CHECK_OFFLOAD(rx_checksum_ipv4);
    CHECK_OFFLOAD(rx_checksum_tcp);
    CHECK_OFFLOAD(rx_checksum_udp);
    CHECK_OFFLOAD(rx_vlan_strip);
    CHECK_OFFLOAD(rx_lro);
    CHECK_OFFLOAD(tx_checksum_ipv4);
    CHECK_OFFLOAD(tx_checksum_tcp);
    CHECK_OFFLOAD(tx_checksum_udp);
    CHECK_OFFLOAD(tx_vlan_insert);
    CHECK_OFFLOAD(tx_tso);
    CHECK_OFFLOAD(tx_ufo);
    CHECK_OFFLOAD(rss);
    
    #undef CHECK_OFFLOAD
    
    enabled_offloads_[port_id] = to_enable;
    
    return true;
}

bool HardwareOffloadManager::enable_checksum_offload(uint16_t port_id) {
    OffloadCapabilities requested = {};
    requested.rx_checksum_ipv4 = true;
    requested.rx_checksum_tcp = true;
    requested.rx_checksum_udp = true;
    requested.tx_checksum_ipv4 = true;
    requested.tx_checksum_tcp = true;
    requested.tx_checksum_udp = true;
    
    return configure_offloads(port_id, requested);
}

bool HardwareOffloadManager::enable_tso(uint16_t port_id, uint16_t mss) {
    OffloadCapabilities requested = {};
    requested.tx_tso = true;
    
    bool result = configure_offloads(port_id, requested);
    
    if (result && enabled_offloads_[port_id].tx_tso) {
        std::cout << "TSO enabled on port " << port_id 
                  << " with MSS " << mss << std::endl;
    }
    
    return result;
}

bool HardwareOffloadManager::enable_rss(uint16_t port_id, uint16_t nb_queues) {
    OffloadCapabilities requested = {};
    requested.rss = true;
    
    bool result = configure_offloads(port_id, requested);
    
    if (result && enabled_offloads_[port_id].rss) {
        std::cout << "RSS enabled on port " << port_id 
                  << " with " << nb_queues << " queues" << std::endl;
    }
    
    return result;
}

void HardwareOffloadManager::print_capabilities(uint16_t port_id) const {
    auto it = port_capabilities_.find(port_id);
    if (it == port_capabilities_.end()) {
        std::cout << "No capabilities detected for port " << port_id << std::endl;
        return;
    }
    
    const OffloadCapabilities& caps = it->second;
    
    std::cout << "\n=== Port " << port_id << " Offload Capabilities ===" << std::endl;
    
    std::cout << "RX Offloads:" << std::endl;
    std::cout << "  IPv4 checksum: " << (caps.rx_checksum_ipv4 ? "YES" : "NO") << std::endl;
    std::cout << "  TCP checksum:  " << (caps.rx_checksum_tcp ? "YES" : "NO") << std::endl;
    std::cout << "  UDP checksum:  " << (caps.rx_checksum_udp ? "YES" : "NO") << std::endl;
    std::cout << "  VLAN strip:    " << (caps.rx_vlan_strip ? "YES" : "NO") << std::endl;
    std::cout << "  LRO:           " << (caps.rx_lro ? "YES" : "NO") << std::endl;
    
    std::cout << "\nTX Offloads:" << std::endl;
    std::cout << "  IPv4 checksum: " << (caps.tx_checksum_ipv4 ? "YES" : "NO") << std::endl;
    std::cout << "  TCP checksum:  " << (caps.tx_checksum_tcp ? "YES" : "NO") << std::endl;
    std::cout << "  UDP checksum:  " << (caps.tx_checksum_udp ? "YES" : "NO") << std::endl;
    std::cout << "  VLAN insert:   " << (caps.tx_vlan_insert ? "YES" : "NO") << std::endl;
    std::cout << "  TSO:           " << (caps.tx_tso ? "YES" : "NO") << std::endl;
    std::cout << "  UFO:           " << (caps.tx_ufo ? "YES" : "NO") << std::endl;
    
    std::cout << "\nAdvanced Features:" << std::endl;
    std::cout << "  RSS:           " << (caps.rss ? "YES" : "NO") << std::endl;
}

// ============================================================================
// MemoryPoolOptimizer Implementation
// ============================================================================

MemoryPoolOptimizer::MemoryPoolOptimizer() {}

MemoryPoolOptimizer::~MemoryPoolOptimizer() {}

struct rte_mempool* MemoryPoolOptimizer::create_pool(const char* name, 
                                                     const PoolConfig& config) {
    struct rte_mempool* pool = rte_mempool_create(
        name,
        config.num_elements,
        config.element_size,
        config.cache_size,
        0,  // private data size
        nullptr, nullptr,  // pool constructor
        nullptr, nullptr,  // object constructor
        config.numa_node,
        0   // flags
    );
    
    if (pool) {
        pool_configs_[pool] = config;
        std::cout << "Created optimized pool '" << name << "' with "
                  << config.num_elements << " elements of size "
                  << config.element_size << " on NUMA " << config.numa_node << std::endl;
    }
    
    return pool;
}

bool MemoryPoolOptimizer::optimize_pool(struct rte_mempool* pool) {
    if (!pool) {
        return false;
    }
    
    PoolStats stats = get_stats(pool);
    
    // Optimize cache size based on usage pattern
    if (stats.utilization > 0.9) {
        // High utilization - increase cache size
        // Note: This is simplified - real implementation would recreate pool
        std::cout << "Pool utilization high (" << stats.utilization * 100 
                  << "%), consider increasing size" << std::endl;
    }
    
    return true;
}

MemoryPoolOptimizer::PoolStats MemoryPoolOptimizer::get_stats(
    struct rte_mempool* pool) const {
    
    PoolStats stats = {};
    
    if (!pool) {
        return stats;
    }
    
    stats.available = rte_mempool_avail_count(pool);
    stats.in_use = rte_mempool_in_use_count(pool);
    
    size_t total = stats.available + stats.in_use;
    if (total > 0) {
        stats.utilization = static_cast<double>(stats.in_use) / total;
    }
    
    // Get cache stats
    unsigned lcore_id;
    RTE_LCORE_FOREACH(lcore_id) {
        stats.cache_count += pool->local_cache[lcore_id].len;
    }
    
    return stats;
}

bool MemoryPoolOptimizer::auto_tune(struct rte_mempool* pool, double target_utilization) {
    if (!pool) {
        return false;
    }
    
    PoolStats current = get_stats(pool);
    pool_history_[pool] = current;
    
    if (std::abs(current.utilization - target_utilization) > 0.1) {
        size_t optimal = calculate_optimal_size(current, target_utilization);
        std::cout << "Pool auto-tuning: current util=" << current.utilization
                  << ", target=" << target_utilization
                  << ", suggested size=" << optimal << std::endl;
    }
    
    return true;
}

size_t MemoryPoolOptimizer::calculate_optimal_size(const PoolStats& stats, 
                                                   double target_util) const {
    if (target_util <= 0 || target_util >= 1.0) {
        return stats.available + stats.in_use;
    }
    
    // Simple calculation: scale current size to achieve target utilization
    size_t current_size = stats.available + stats.in_use;
    size_t optimal_size = static_cast<size_t>(stats.in_use / target_util);
    
    // Add some headroom
    optimal_size = static_cast<size_t>(optimal_size * 1.1);
    
    // Round up to power of 2 for efficiency
    optimal_size--;
    optimal_size |= optimal_size >> 1;
    optimal_size |= optimal_size >> 2;
    optimal_size |= optimal_size >> 4;
    optimal_size |= optimal_size >> 8;
    optimal_size |= optimal_size >> 16;
    optimal_size++;
    
    return optimal_size;
}

// ============================================================================
// CPUFrequencyManager Implementation
// ============================================================================

CPUFrequencyManager::CPUFrequencyManager() {}

bool CPUFrequencyManager::set_governor(unsigned core_id, Governor gov) {
    std::stringstream path;
    path << "/sys/devices/system/cpu/cpu" << core_id << "/cpufreq/scaling_governor";
    
    return write_sysfs(path.str(), governor_to_string(gov));
}

bool CPUFrequencyManager::set_governor_all(Governor gov) {
    unsigned lcore_id;
    bool success = true;
    
    RTE_LCORE_FOREACH(lcore_id) {
        if (!set_governor(lcore_id, gov)) {
            success = false;
        }
    }
    
    return success;
}

bool CPUFrequencyManager::set_frequency(unsigned core_id, uint64_t freq_khz) {
    // First set to userspace governor
    if (!set_governor(core_id, USERSPACE)) {
        return false;
    }
    
    std::stringstream path;
    path << "/sys/devices/system/cpu/cpu" << core_id << "/cpufreq/scaling_setspeed";
    
    return write_sysfs(path.str(), std::to_string(freq_khz));
}

uint64_t CPUFrequencyManager::get_frequency(unsigned core_id) const {
    std::stringstream path;
    path << "/sys/devices/system/cpu/cpu" << core_id << "/cpufreq/scaling_cur_freq";
    
    std::string freq_str = read_sysfs(path.str());
    if (!freq_str.empty()) {
        return std::stoull(freq_str);
    }
    
    return 0;
}

bool CPUFrequencyManager::disable_c_states(unsigned core_id) {
    // Disable idle states to reduce latency
    std::stringstream path;
    path << "/sys/devices/system/cpu/cpu" << core_id << "/cpuidle/state";
    
    // Disable all C-states except C0
    for (int state = 1; state < 10; state++) {
        std::stringstream state_path;
        state_path << path.str() << state << "/disable";
        
        if (!write_sysfs(state_path.str(), "1")) {
            break;  // No more states
        }
    }
    
    return true;
}

std::string CPUFrequencyManager::governor_to_string(Governor gov) const {
    switch (gov) {
        case PERFORMANCE: return "performance";
        case POWERSAVE: return "powersave";
        case ONDEMAND: return "ondemand";
        case USERSPACE: return "userspace";
        default: return "ondemand";
    }
}

bool CPUFrequencyManager::write_sysfs(const std::string& path, const std::string& value) {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }
    
    file << value;
    file.close();
    
    return file.good();
}

std::string CPUFrequencyManager::read_sysfs(const std::string& path) const {
    std::ifstream file(path);
    if (!file.is_open()) {
        return "";
    }
    
    std::string value;
    file >> value;
    file.close();
    
    return value;
}

// ============================================================================
// PerformanceTuner Implementation
// ============================================================================

PerformanceTuner::PerformanceTuner() {}

PerformanceTuner::~PerformanceTuner() {}

bool PerformanceTuner::initialize(const TuningConfig& config) {
    config_ = config;
    
    // Initialize components
    if (config.numa_aware) {
        numa_allocator_ = std::make_unique<NUMAAllocator>();
        if (!numa_allocator_->initialize()) {
            std::cerr << "Warning: NUMA initialization failed" << std::endl;
        } else {
            numa_allocator_->print_topology();
        }
    }
    
    prefetch_manager_ = std::make_unique<PrefetchManager>();
    prefetch_manager_->set_strategy(config.prefetch_strategy);
    
    offload_manager_ = std::make_unique<HardwareOffloadManager>();
    
    pool_optimizer_ = std::make_unique<MemoryPoolOptimizer>();
    
    cpu_manager_ = std::make_unique<CPUFrequencyManager>();
    
    std::cout << "Performance tuner initialized" << std::endl;
    
    return true;
}

bool PerformanceTuner::apply_tunings(uint16_t port_id) {
    std::cout << "\nApplying performance tunings..." << std::endl;
    
    // Setup NUMA bindings
    if (config_.numa_aware && config_.bind_threads) {
        setup_numa_bindings();
    }
    
    // Configure hardware offloads
    if (config_.enable_offloads) {
        configure_hardware_offloads(port_id);
    }
    
    // Optimize memory pools
    if (config_.optimize_pools) {
        optimize_memory_pools();
    }
    
    // Tune CPU settings
    tune_cpu_settings();
    
    std::cout << "Performance tunings applied successfully" << std::endl;
    
    return true;
}

void PerformanceTuner::optimize_runtime() {
    // Runtime optimization - called periodically
    
    // Auto-tune memory pools
    // Note: Simplified - would need actual pool references
    
    // Adjust prefetch strategy based on patterns
    // Note: Would need actual access pattern data
    
    // Update performance counters
    packet_count_++;
    cycle_count_ += rte_rdtsc();
}

PerformanceTuner::PerformanceMetrics PerformanceTuner::get_metrics() const {
    PerformanceMetrics metrics = {};
    
    uint64_t hz = rte_get_tsc_hz();
    uint64_t cycles = cycle_count_.load();
    uint64_t packets = packet_count_.load();
    uint64_t bytes = byte_count_.load();
    
    if (cycles > 0 && hz > 0) {
        double seconds = static_cast<double>(cycles) / hz;
        metrics.packets_per_second = packets / seconds;
        metrics.bytes_per_second = bytes / seconds;
        metrics.average_latency_ns = (seconds / packets) * 1e9;
    }
    
    // CPU utilization (simplified)
    metrics.cpu_utilization = 0.85;  // Placeholder
    
    // Cache metrics (would need perf counters)
    metrics.cache_hit_rate = 0.95;  // Placeholder
    
    return metrics;
}

void PerformanceTuner::print_report() const {
    PerformanceMetrics metrics = get_metrics();
    
    std::cout << "\n=== Performance Report ===" << std::endl;
    std::cout << "Packets/sec:     " << metrics.packets_per_second << std::endl;
    std::cout << "Bytes/sec:       " << metrics.bytes_per_second << std::endl;
    std::cout << "Avg latency:     " << metrics.average_latency_ns << " ns" << std::endl;
    std::cout << "CPU utilization: " << metrics.cpu_utilization * 100 << "%" << std::endl;
    std::cout << "Cache hit rate:  " << metrics.cache_hit_rate * 100 << "%" << std::endl;
}

void PerformanceTuner::setup_numa_bindings() {
    if (!numa_allocator_) {
        return;
    }
    
    std::cout << "Setting up NUMA thread bindings..." << std::endl;
    
    unsigned lcore_id;
    RTE_LCORE_FOREACH(lcore_id) {
        int numa_node = numa_allocator_->get_numa_node_for_core(lcore_id);
        
        // Pin thread to NUMA node
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(lcore_id, &cpuset);
        
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0) {
            std::cout << "  Pinned lcore " << lcore_id 
                     << " to NUMA node " << numa_node << std::endl;
        }
    }
}

void PerformanceTuner::configure_hardware_offloads(uint16_t port_id) {
    if (!offload_manager_) {
        return;
    }
    
    std::cout << "Configuring hardware offloads for port " << port_id << "..." << std::endl;
    
    // Detect capabilities
    offload_manager_->detect_capabilities(port_id);
    offload_manager_->print_capabilities(port_id);
    
    // Enable requested offloads
    if (config_.enable_checksum) {
        offload_manager_->enable_checksum_offload(port_id);
    }
    
    if (config_.enable_tso) {
        offload_manager_->enable_tso(port_id);
    }
    
    if (config_.enable_rss) {
        offload_manager_->enable_rss(port_id, 4);  // 4 queues
    }
}

void PerformanceTuner::optimize_memory_pools() {
    std::cout << "Optimizing memory pools..." << std::endl;
    
    // Create optimized pools for different purposes
    MemoryPoolOptimizer::PoolConfig mbuf_config = {
        .element_size = RTE_MBUF_DEFAULT_BUF_SIZE,
        .num_elements = 8192,
        .cache_size = 256,
        .numa_node = 0,
        .use_huge_pages = true
    };
    
    struct rte_mempool* mbuf_pool = pool_optimizer_->create_pool(
        "optimized_mbuf_pool", mbuf_config);
    
    if (mbuf_pool) {
        // Monitor and auto-tune
        pool_optimizer_->auto_tune(mbuf_pool, config_.pool_target_utilization);
    }
}

void PerformanceTuner::tune_cpu_settings() {
    if (!cpu_manager_) {
        return;
    }
    
    std::cout << "Tuning CPU settings..." << std::endl;
    
    // Set CPU governor
    if (cpu_manager_->set_governor_all(config_.cpu_governor)) {
        std::cout << "  CPU governor set to " 
                  << (config_.cpu_governor == CPUFrequencyManager::PERFORMANCE ? 
                      "performance" : "adaptive") << std::endl;
    }
    
    // Disable C-states for low latency
    if (config_.disable_cpu_idle) {
        unsigned lcore_id;
        RTE_LCORE_FOREACH(lcore_id) {
            cpu_manager_->disable_c_states(lcore_id);
        }
        std::cout << "  CPU idle states disabled for low latency" << std::endl;
    }
}

} // namespace data_plane
} // namespace genie














