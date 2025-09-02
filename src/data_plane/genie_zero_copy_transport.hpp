/**
 * Zero-Copy Transport Implementation
 * 
 * Provides true zero-copy data transfer between GPU memory and NIC
 * using DPDK external buffers and GPUDev for direct DMA
 */

#pragma once

#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <rte_ethdev.h>
#include <rte_malloc.h>
#include <rte_ring.h>
#include <rte_gpudev.h>

#ifdef GENIE_CUDA_SUPPORT
#include <cuda_runtime.h>
#else
// CUDA types for compatibility when CUDA is not available
typedef void* cudaStream_t;
typedef int cudaError_t;
#define cudaSuccess 0
#endif

#include <memory>
#include <vector>
#include <map>
#include <queue>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>

#include "genie_data_plane.hpp"
#include "genie_monitoring.hpp"

namespace genie {
namespace data_plane {

/**
 * GPU buffer descriptor for zero-copy
 */
struct GPUBuffer {
    void* gpu_ptr;           // GPU memory pointer
    uint64_t iova;          // IO virtual address for DMA
    size_t size;            // Buffer size
    int gpu_id;             // GPU device ID
    cudaStream_t stream;    // CUDA stream for async ops
    
    // For external buffer attachment
    struct rte_mbuf_ext_shared_info* shinfo;
    int refcount;  // Changed from atomic to regular int
    
    GPUBuffer() : gpu_ptr(nullptr), iova(0), size(0), gpu_id(0), 
                  stream(nullptr), shinfo(nullptr), refcount(1) {}
    
    // Copy constructor
    GPUBuffer(const GPUBuffer& other) 
        : gpu_ptr(other.gpu_ptr), iova(other.iova), size(other.size),
          gpu_id(other.gpu_id), stream(other.stream), shinfo(other.shinfo),
          refcount(other.refcount) {}
    
    // Assignment operator
    GPUBuffer& operator=(const GPUBuffer& other) {
        if (this != &other) {
            gpu_ptr = other.gpu_ptr;
            iova = other.iova;
            size = other.size;
            gpu_id = other.gpu_id;
            stream = other.stream;
            shinfo = other.shinfo;
            refcount = other.refcount;
        }
        return *this;
    }
    
    bool is_valid() const { return gpu_ptr != nullptr && iova != 0; }
};

/**
 * Transfer descriptor
 */
struct TransferDescriptor {
    std::string transfer_id;
    GPUBuffer gpu_buffer;
    void* cpu_staging;      // Fallback CPU buffer if needed
    size_t total_size;
    size_t bytes_sent;
    size_t bytes_received;
    
    // Fragment tracking
    uint32_t total_fragments;
    std::vector<bool> fragments_received;
    std::vector<struct rte_mbuf*> fragment_mbufs;
    
    // Timing
    uint64_t start_tsc;
    uint64_t end_tsc;
    
    // Completion
    std::promise<bool> completion_promise;
    std::atomic<bool> completed{false};
    std::atomic<bool> failed{false};
    
    TransferDescriptor() : cpu_staging(nullptr), total_size(0), 
                          bytes_sent(0), bytes_received(0),
                          total_fragments(0), start_tsc(0), end_tsc(0) {}
};

/**
 * Zero-copy transport manager
 */
class ZeroCopyTransport {
public:
    struct Config {
        uint16_t port_id = 0;
        uint16_t tx_queue = 0;
        uint16_t rx_queue = 0;
        size_t mtu = 8192;
        size_t burst_size = 32;
        bool use_gpu_direct = true;
        bool use_external_buffers = true;
        size_t gpu_buffer_pool_size = 1024 * 1024 * 1024; // 1GB
        int gpu_id = 0;
    };
    
    ZeroCopyTransport(const Config& config);
    ~ZeroCopyTransport();
    
    // Initialize transport
    bool initialize();
    void shutdown();
    
    // Send operations
    std::future<bool> send_gpu_buffer(const GPUBuffer& buffer, 
                                      const std::string& transfer_id,
                                      uint32_t dest_ip, uint16_t dest_port);
    
    std::future<bool> send_tensor_zero_copy(void* gpu_ptr, size_t size,
                                           const std::string& transfer_id,
                                           uint32_t dest_ip, uint16_t dest_port);
    
    // Receive operations
    bool prepare_receive(const std::string& transfer_id, size_t size);
    std::future<GPUBuffer> receive_to_gpu(const std::string& transfer_id);
    
    // Polling functions (called from DPDK threads)
    void poll_tx();
    void poll_rx();
    void process_completions();
    
    // Statistics
    struct Stats {
        uint64_t transfers_sent;
        uint64_t transfers_received;
        uint64_t bytes_sent;
        uint64_t bytes_received;
        uint64_t zero_copy_sends;
        uint64_t fallback_sends;
        uint64_t dma_errors;
        double avg_throughput_gbps;
        double avg_latency_us;
    };
    
    Stats get_stats() const;
    void print_stats() const;

    // Introspection helpers
    bool gpu_direct_enabled() const { return config_.use_gpu_direct; }
    size_t gpu_buffer_count() const { return gpu_buffer_pool_.size(); }
    
private:
    Config config_;
    
    // DPDK resources
    struct rte_mempool* mbuf_pool_;
    struct rte_mempool* extbuf_pool_;  // For external buffer info
    struct rte_ring* tx_ring_;
    struct rte_ring* rx_ring_;
    struct rte_ring* completion_ring_;
    
    // GPU resources
    int gpu_dev_id_;  // DPDK GPU device ID
    cudaStream_t cuda_stream_;
    std::vector<GPUBuffer> gpu_buffer_pool_;
    std::queue<GPUBuffer*> available_gpu_buffers_;
    std::mutex gpu_buffer_mutex_;
    
    // Transfer tracking
    std::map<std::string, std::unique_ptr<TransferDescriptor>> active_transfers_;
    std::mutex transfer_mutex_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    Stats stats_;
    
    // Helper functions
    bool init_dpdk_resources();
    bool init_gpu_resources();
    bool register_gpu_memory(GPUBuffer& buffer);
    void unregister_gpu_memory(GPUBuffer& buffer);
    
    // Packet operations
    struct rte_mbuf* create_packet_with_gpu_memory(const GPUBuffer& buffer,
                                                   size_t offset, size_t length,
                                                   uint32_t seq_num, uint32_t frag_id,
                                                   uint32_t total_frags,
                                                   const std::string& transfer_id);
    
    bool attach_external_gpu_buffer(struct rte_mbuf* mbuf,
                                   const GPUBuffer& buffer,
                                   size_t offset, size_t length);
    
    void fragment_and_send(const GPUBuffer& buffer,
                          const std::string& transfer_id,
                          uint32_t dest_ip, uint16_t dest_port);
    
    bool reassemble_fragments(TransferDescriptor& transfer);
    
    // Completion handling
    void complete_transfer(const std::string& transfer_id, bool success);
    
    // Fallback path
    bool send_with_cpu_staging(void* gpu_ptr, size_t size,
                              const std::string& transfer_id,
                              uint32_t dest_ip, uint16_t dest_port);
    
    // External buffer callbacks
    static void ext_buf_free_cb(void* addr, void* opaque);
};

/**
 * External buffer manager for zero-copy
 */
class ExternalBufferManager {
public:
    ExternalBufferManager(size_t pool_size = 1024);
    ~ExternalBufferManager();
    
    // Get shared info for external buffer
    struct rte_mbuf_ext_shared_info* alloc_shinfo();
    void free_shinfo(struct rte_mbuf_ext_shared_info* shinfo);
    
    // Reference counting
    void add_ref(struct rte_mbuf_ext_shared_info* shinfo);
    void release_ref(struct rte_mbuf_ext_shared_info* shinfo);
    
private:
    struct rte_mempool* shinfo_pool_;
    std::mutex mutex_;
    
    static void free_cb(void* addr, void* opaque);
};

/**
 * DMA engine for GPU-NIC transfers
 */
class GPUNICDMAEngine {
public:
    GPUNICDMAEngine(int gpu_id, uint16_t port_id);
    ~GPUNICDMAEngine();
    
    // Initialize DMA engine
    bool initialize();
    
    // DMA operations
    bool dma_gpu_to_nic(const GPUBuffer& gpu_buffer,
                       struct rte_mbuf** mbufs, uint16_t nb_mbufs);
    
    bool dma_nic_to_gpu(struct rte_mbuf** mbufs, uint16_t nb_mbufs,
                       GPUBuffer& gpu_buffer, size_t offset);
    
    // Check if GPU Direct is available
    bool is_gpu_direct_available() const { return gpu_direct_available_; }
    
    // Wait for DMA completion
    bool wait_dma_completion(uint64_t timeout_cycles = 0);
    
private:
    int gpu_id_;
    uint16_t port_id_;
    int gpu_dev_id_;  // DPDK GPU device ID
    
    bool gpu_direct_available_;
    cudaStream_t dma_stream_;
    
    // DMA descriptor rings
    struct dma_descriptor {
        void* src;
        void* dst;
        size_t size;
        bool completed;
    };
    
    std::vector<dma_descriptor> pending_dma_;
    std::mutex dma_mutex_;
    
    bool setup_gpu_direct();
    bool perform_dma(void* src, void* dst, size_t size, bool gpu_to_nic);
};

/**
 * Reliability manager for ACK/NACK handling
 */
class ReliabilityManager {
public:
    struct Config {
        uint32_t timeout_ms = 100;
        uint32_t max_retries = 3;
        double initial_rtt_ms = 10.0;
        bool enable_sack = true;  // Selective ACK
    };
    
    ReliabilityManager(const Config& config);
    ~ReliabilityManager();
    
    // Track sent packets
    void track_packet(struct rte_mbuf* mbuf, uint32_t seq_num,
                     const std::string& transfer_id);
    
    // Process ACK/NACK
    void process_ack(uint32_t ack_num, const std::string& transfer_id);
    void process_nack(uint32_t seq_num, const std::string& transfer_id);
    void process_sack(const std::vector<uint32_t>& sack_blocks,
                     const std::string& transfer_id);
    
    // Check for timeouts and retransmit
    std::vector<struct rte_mbuf*> check_timeouts();
    
    // RTT estimation
    void update_rtt(double sample_rtt_ms);
    double get_rto() const { return rto_ms_; }
    
    // Statistics
    struct Stats {
        uint64_t packets_sent;
        uint64_t packets_acked;
        uint64_t packets_nacked;
        uint64_t retransmissions;
        uint64_t timeouts;
        double avg_rtt_ms;
        double min_rtt_ms;
        double max_rtt_ms;
    };
    
    Stats get_stats() const;
    
private:
    Config config_;
    
    // Packet tracking
    struct PacketInfo {
        struct rte_mbuf* mbuf;
        uint32_t seq_num;
        std::string transfer_id;
        uint64_t send_time_tsc;
        uint32_t retry_count;
        bool acked;
    };
    
    std::map<uint32_t, PacketInfo> unacked_packets_;
    std::mutex packet_mutex_;
    
    // RTT estimation (using TCP's algorithm)
    double srtt_ms_;  // Smoothed RTT
    double rttvar_ms_;  // RTT variance
    double rto_ms_;  // Retransmission timeout
    
    // Statistics
    mutable std::mutex stats_mutex_;
    Stats stats_;
    
    void calculate_rto();
    bool should_retransmit(const PacketInfo& pkt) const;
};

/**
 * Flow control manager
 */
class FlowControlManager {
public:
    enum CongestionState {
        SLOW_START,
        CONGESTION_AVOIDANCE,
        FAST_RECOVERY,
        TIMEOUT
    };
    
    struct Config {
        uint32_t initial_cwnd = 10;
        uint32_t max_cwnd = 10000;
        uint32_t ssthresh = 64;
        bool enable_cubic = true;  // Use CUBIC instead of Reno
    };
    
    FlowControlManager(const Config& config);
    ~FlowControlManager();
    
    // Window management
    bool can_send() const;
    void on_packet_sent(uint32_t seq_num);
    void on_ack_received(uint32_t ack_num);
    void on_loss_detected();
    void on_timeout();
    
    // Get current window size
    uint32_t get_cwnd() const { return cwnd_; }
    uint32_t get_in_flight() const { return in_flight_; }
    
    // Rate limiting
    void set_rate_limit(double gbps);
    bool check_rate_limit() const;
    
    // Statistics
    struct Stats {
        uint32_t current_cwnd;
        uint32_t current_ssthresh;
        uint32_t in_flight;
        CongestionState state;
        uint64_t total_sent;
        uint64_t total_acked;
        double throughput_gbps;
    };
    
    Stats get_stats() const;
    
private:
    Config config_;
    
    // Congestion control state
    std::atomic<uint32_t> cwnd_;
    std::atomic<uint32_t> ssthresh_;
    std::atomic<uint32_t> in_flight_;
    std::atomic<CongestionState> state_;
    
    // CUBIC parameters
    double cubic_beta_ = 0.7;
    double cubic_c_ = 0.4;
    uint32_t w_max_;  // Window size before last reduction
    uint64_t epoch_start_;  // Time of last reduction
    
    // Rate limiting
    std::atomic<bool> rate_limited_{false};
    double rate_limit_gbps_{0};
    uint64_t last_send_time_{0};
    uint64_t tokens_{0};
    
    // Statistics
    mutable std::mutex stats_mutex_;
    uint64_t total_sent_{0};
    uint64_t total_acked_{0};
    
    void increase_window();
    void decrease_window();
    uint32_t cubic_window(uint64_t t) const;
};

} // namespace data_plane
} // namespace genie

