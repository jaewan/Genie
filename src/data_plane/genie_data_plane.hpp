/**
 * Genie Zero-Copy Data Plane
 * 
 * High-performance C++ implementation of the DPDK-based data plane for zero-copy
 * tensor transfers between GPU nodes.
 * 
 * Architecture:
 * - Pure C++ for maximum performance
 * - DPDK for userspace networking
 * - GPUDev for GPU memory registration
 * - Custom reliable transport protocol
 * - Thread-safe packet processing
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

// DPDK headers
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <rte_ring.h>
#include <rte_lcore.h>
#include <rte_gpudev.h>

// CUDA headers (for GPU memory)
#ifdef GENIE_CUDA_SUPPORT
#include <cuda_runtime.h>
#include <cuda.h>
#endif

namespace genie {
namespace data_plane {

// Forward declarations
class PacketProcessor;
class GPUMemoryManager;
class ReliabilityManager;
class FlowController;
class KCPWrapper;

/**
 * Configuration for the data plane
 */
struct DataPlaneConfig {
    // DPDK configuration
    std::vector<std::string> eal_args;
    uint16_t port_id = 0;
    uint16_t queue_id = 0;
    uint16_t rx_queues = 1;
    uint16_t tx_queues = 1;
    bool enable_rss = false;
    uint32_t mempool_size = 8192;
    uint16_t rx_ring_size = 1024;
    uint16_t tx_ring_size = 1024;
    // NIC offloads (configurable; applied when supported by device)
    bool rx_offload_checksum = false;
    bool rx_offload_lro = false;
    bool tx_offload_ipv4_cksum = false;
    bool tx_offload_udp_cksum = false;
    bool tx_offload_tso = false;
    
    // GPU configuration
    int gpu_device_id = 0;
    bool enable_gpudev = true;
    
    // Network configuration
    std::string local_ip = "192.168.1.100";
    std::string local_mac = "aa:bb:cc:dd:ee:01";
    uint16_t data_port = 5556;
    
    // Performance tuning
    uint32_t burst_size = 32;
    uint32_t poll_interval_us = 100;
    bool enable_batching = true;
    uint16_t mtu = 9000;  // Jumbo frames for cloud
    
    // Reliability
    uint32_t ack_timeout_ms = 100;
    uint32_t max_retries = 3;
    uint32_t window_size = 64;
    
    // Phase 3: CUDA Graphs integration
    bool enable_cuda_graphs = false;
    
    // Enhanced features
    bool use_dpdk_libs = true;  // Use rte_ip_frag, rte_reorder, rte_hash
    uint32_t max_connections = 4096;
    
    // Thread pool configuration
    bool use_thread_pool = true;       // Use high-performance thread pool
    uint32_t rx_threads = 1;           // Number of RX threads
    uint32_t tx_threads = 1;           // Number of TX threads
    uint32_t worker_threads = 2;       // Number of worker threads
    bool thread_busy_wait = true;      // Use busy-wait vs hybrid mode
    uint64_t thread_sleep_ns = 100;    // Sleep time in hybrid mode
    uint32_t fragment_timeout_ms = 2000;
};

/**
 * Packet header structures (matching Python protocol.py)
 */
#pragma pack(push, 1)

struct EthernetHeader {
    uint8_t dst_mac[6];
    uint8_t src_mac[6];
    uint16_t ethertype;
    
    EthernetHeader() : ethertype(0x0800) {} // IPv4
};

struct IPv4Header {
    uint8_t version_ihl;
    uint8_t tos;
    uint16_t total_length;
    uint16_t identification;
    uint16_t flags_fragment;
    uint8_t ttl;
    uint8_t protocol;
    uint16_t checksum;
    uint32_t src_ip;
    uint32_t dst_ip;
    
    IPv4Header() : version_ihl(0x45), ttl(64), protocol(17) {} // UDP
};

struct UDPHeader {
    uint16_t src_port;
    uint16_t dst_port;
    uint16_t length;
    uint16_t checksum;
};

struct GeniePacketHeader {
    uint32_t magic;           // 0x47454E49 "GENI"
    uint8_t version;          // Protocol version
    uint8_t flags;            // Packet flags
    uint8_t type;             // Packet type
    uint8_t reserved;         // Reserved byte
    uint8_t tensor_id[16];    // Tensor UUID
    uint32_t seq_num;         // Sequence number
    uint16_t frag_id;         // Fragment ID
    uint16_t frag_count;      // Total fragments
    uint32_t offset;          // Data offset
    uint32_t length;          // Data length
    uint32_t total_size;      // Total tensor size
    uint32_t checksum;        // Data checksum
    uint64_t timestamp_ns;    // Timestamp
    // Semantic metadata (compact): dtype code + shape dims + phase
    uint8_t dtype_code;       // 0=unknown,1=f32,2=f16,3=i64,4=i32,5=u8
    uint8_t phase;            // 0=prefill,1=decode,2=fusion,3=other
    uint8_t shape_rank;       // number of dims (0..4)
    uint16_t shape_dims[4];   // up to 4 dims (pack larger dims as truncated)
};

struct GeniePacket {
    EthernetHeader eth;
    IPv4Header ip;
    UDPHeader udp;
    GeniePacketHeader app;
};

#pragma pack(pop)

// Packet types
enum class PacketType : uint8_t {
    DATA = 0,
    ACK = 1,
    NACK = 2,
    HEARTBEAT = 3,
    CONTROL = 4
};

// Packet flags
enum class PacketFlags : uint8_t {
    NONE = 0x00,
    FRAGMENTED = 0x01,
    LAST_FRAGMENT = 0x02,
    RETRANSMIT = 0x04,
    COMPRESSED = 0x08,
    ENCRYPTED = 0x10
};

/**
 * GPU memory handle for DMA operations
 */
struct GPUMemoryHandle {
    void* gpu_ptr;           // GPU memory pointer
    uint64_t iova;           // IOVA address for DMA
    size_t size;             // Memory size
    int gpu_id;              // GPU device ID
    uint32_t ref_count;      // Reference count
    bool is_registered;      // GPUDev registration status
    
    GPUMemoryHandle() : gpu_ptr(nullptr), iova(0), size(0), gpu_id(0), 
                       ref_count(0), is_registered(false) {}
    
    bool is_valid() const { return gpu_ptr != nullptr && iova != 0; }
};

/**
 * Transfer context for tracking active transfers
 */
struct TransferContext {
    std::string transfer_id;
    std::string tensor_id;
    std::string source_node;
    std::string target_node;
    
    // Memory management
    GPUMemoryHandle gpu_handle;
    void* cpu_buffer;        // Fallback CPU buffer
    size_t total_size;
    
    // Fragmentation
    uint32_t total_fragments;
    std::vector<bool> received_fragments;
    std::vector<uint8_t*> fragment_data;
    
    // Statistics
    std::atomic<uint64_t> bytes_transferred{0};
    std::atomic<uint32_t> packets_sent{0};
    std::atomic<uint32_t> packets_received{0};
    std::chrono::high_resolution_clock::time_point start_time;
    
    // State
    std::atomic<bool> is_complete{false};
    std::atomic<bool> has_error{false};
    std::string error_message;
    // Semantic metadata for header (default unknown)
    uint8_t meta_dtype_code{0};
    uint8_t meta_phase{3};
    uint8_t meta_shape_rank{0};
    uint16_t meta_shape_dims[4]{};
    
    TransferContext() : cpu_buffer(nullptr), total_size(0), total_fragments(0) {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    ~TransferContext() {
        if (cpu_buffer) {
            free(cpu_buffer);
        }
    }
};

/**
 * Statistics for monitoring performance
 */
struct DataPlaneStats {
    std::atomic<uint64_t> packets_sent{0};
    std::atomic<uint64_t> packets_received{0};
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> bytes_received{0};
    std::atomic<uint64_t> packets_dropped{0};
    std::atomic<uint64_t> retransmissions{0};
    std::atomic<uint64_t> transfers_completed{0};
    std::atomic<uint64_t> transfers_failed{0};
    std::atomic<uint32_t> active_transfers{0};
    
    // Performance metrics
    std::atomic<uint64_t> avg_latency_ns{0};
    std::atomic<uint64_t> peak_bandwidth_bps{0};
    std::atomic<double> packet_loss_rate{0.0};
};

/**
 * Main data plane class
 */
class GenieDataPlane {
public:
    explicit GenieDataPlane(const DataPlaneConfig& config);
    virtual ~GenieDataPlane();
    // Phase 2.1: runtime reliability mode scaffolding (no behavior change yet)
    enum class ReliabilityMode { CUSTOM = 0, KCP = 1 };
    void set_reliability_mode(ReliabilityMode mode) { reliability_mode_ = mode; }
    ReliabilityMode reliability_mode() const { return reliability_mode_; }
    
    // Lifecycle management
    virtual bool initialize();
    bool start();
    void stop();
    virtual void shutdown();
    
    // Thread management
    bool initialize_thread_pool(bool use_thread_pool = true);
    void start_thread_pool();
    void stop_thread_pool();
    void print_thread_statistics() const;
    
    // Transfer operations
    bool send_tensor(const std::string& transfer_id,
                    const std::string& tensor_id,
                    void* gpu_ptr,
                    size_t size,
                    const std::string& target_node);
    
    bool receive_tensor(const std::string& transfer_id,
                       const std::string& tensor_id,
                       void* gpu_ptr,
                       size_t size,
                       const std::string& source_node);
    
    // Semantic metadata setter
    bool set_transfer_metadata(const std::string& transfer_id,
                               uint8_t dtype_code,
                               uint8_t phase,
                               const uint16_t* shape_dims,
                               uint8_t shape_rank);
    
    // Memory management
    bool register_gpu_memory(void* gpu_ptr, size_t size, GPUMemoryHandle& handle);
    void unregister_gpu_memory(const GPUMemoryHandle& handle);
    
    // Status and monitoring
    void get_statistics(DataPlaneStats& stats) const;
    std::vector<std::string> get_active_transfers() const;
    void get_transfer_status(const std::string& transfer_id, TransferContext& context, bool& found) const;
    
    // Configuration
    void set_target_node(const std::string& node_id, const std::string& ip, const std::string& mac);
    void remove_target_node(const std::string& node_id);
    
    // Phase 3: Configuration methods for C API
    void configure_queues(uint16_t rx_queues, uint16_t tx_queues, bool enable_rss);
    void enable_offloads(bool rx_checksum, bool rx_lro, bool tx_ipv4_cksum, bool tx_udp_cksum, bool tx_tso);
    void enable_cuda_graphs(bool enable);
    
private:
    // DPDK initialization
    bool init_dpdk();
    bool init_port();
    bool init_mempool();
    
    // GPU initialization
    bool init_gpu();
    
    // Packet processing
    void rx_loop();
    void tx_loop();
    virtual void process_rx_packets(struct rte_mbuf** pkts, uint16_t nb_pkts);
    void process_tx_queue();
    
    // Packet building and parsing
    struct rte_mbuf* build_packet(const std::string& target_node,
                                 const std::string& tensor_id,
                                 uint32_t seq_num,
                                 const uint8_t* payload,
                                 uint32_t payload_size,
                                 PacketType type = PacketType::DATA,
                                 uint16_t frag_id = 0,
                                 uint16_t frag_count = 1,
                                 uint32_t offset = 0,
                                 PacketFlags flags = PacketFlags::NONE);
    
    bool parse_packet(struct rte_mbuf* pkt, GeniePacket& parsed_pkt, uint8_t*& payload, uint32_t& payload_size);
    
    // Fragment management
    virtual bool fragment_and_send(const std::string& transfer_id,
                                  const std::string& tensor_id,
                                  const uint8_t* data,
                                  size_t size,
                                  const std::string& target_node);
    
    bool handle_fragment(const GeniePacket& pkt, const uint8_t* payload, uint32_t payload_size);
    
protected:
    // Reliability - made protected for derived classes
    void send_ack(const std::string& target_node, uint32_t seq_num);
    void send_nack(const std::string& target_node, uint32_t seq_num);
    void handle_ack(uint32_t seq_num);
    void handle_nack(uint32_t seq_num);
    void check_timeouts();
    void send_ack_for_fragment(const GeniePacket& pkt, uint32_t seq_num);
    
private:
    
    // Utilities
    uint32_t calculate_checksum(const uint8_t* data, size_t size);
    void update_statistics();
    int dpdk_send_raw(const char* buf, int len);
    
protected:  // Allow derived classes to access these
    // Configuration
    DataPlaneConfig config_;
    
    // Packet processors
    std::unique_ptr<PacketProcessor> packet_processor_;
    std::unique_ptr<ReliabilityManager> reliability_mgr_;
    std::unique_ptr<FlowController> flow_controller_;
    
    // DPDK resources
    struct rte_mempool* mempool_;
    struct rte_ring* tx_ring_;
    
private:
    uint16_t port_id_;
    uint16_t queue_id_;
    
    // GPU resources
    int gpu_device_id_;
    bool gpudev_available_;
    
    // Threading - Using DPDK native lcore management
    std::atomic<bool> running_{false};
    std::unique_ptr<class DPDKThreadManager> thread_manager_;
    std::thread rx_thread_;  // Legacy fallback
    std::thread tx_thread_;  // Legacy fallback
    std::thread timeout_thread_;
    
    // Transfer management
    mutable std::mutex transfers_mutex_;
    std::unordered_map<std::string, std::unique_ptr<TransferContext>> active_transfers_;
    
    // Network topology
    mutable std::mutex nodes_mutex_;
    std::unordered_map<std::string, std::pair<std::string, std::string>> target_nodes_; // node_id -> (ip, mac)
    
    // Packet queues
    std::queue<struct rte_mbuf*> tx_queue_;
    std::mutex tx_queue_mutex_;
    std::condition_variable tx_queue_cv_;
    
    // Reliability tracking
    mutable std::mutex reliability_mutex_;
    std::unordered_map<uint32_t, std::chrono::high_resolution_clock::time_point> pending_acks_;
    std::unordered_map<uint32_t, struct rte_mbuf*> pending_retx_;
    std::atomic<uint32_t> next_seq_num_{1};
    
    // Statistics
    mutable DataPlaneStats stats_;
    
    // GPU memory management
    std::unique_ptr<GPUMemoryManager> gpu_memory_mgr_;

    // Runtime reliability selection (scaffold)
    ReliabilityMode reliability_mode_ = ReliabilityMode::CUSTOM;
    std::unique_ptr<KCPWrapper> kcp_;
};

/**
 * C interface for Python bindings
 */
extern "C" {
    // Data plane lifecycle
    void* genie_data_plane_create(const char* config_json);
    int genie_data_plane_initialize(void* data_plane);
    int genie_data_plane_start(void* data_plane);
    void genie_data_plane_stop(void* data_plane);
    void genie_data_plane_destroy(void* data_plane);
    
    // Transfer operations
    int genie_send_tensor(void* data_plane,
                         const char* transfer_id,
                         const char* tensor_id,
                         void* gpu_ptr,
                         size_t size,
                         const char* target_node);
    
    int genie_receive_tensor(void* data_plane,
                            const char* transfer_id,
                            const char* tensor_id,
                            void* gpu_ptr,
                            size_t size,
                            const char* source_node);
    
    // Memory management
    int genie_register_gpu_memory(void* data_plane, void* gpu_ptr, size_t size, uint64_t* iova);
    void genie_unregister_gpu_memory(void* data_plane, void* gpu_ptr);
    
    // Status and monitoring
    void genie_get_statistics(void* data_plane, char* stats_json, size_t buffer_size);
    int genie_get_transfer_status(void* data_plane, const char* transfer_id, char* status_json, size_t buffer_size);
    
    // Configuration
    void genie_set_target_node(void* data_plane, const char* node_id, const char* ip, const char* mac);
    void genie_remove_target_node(void* data_plane, const char* node_id);
    
    // Semantic metadata (optional): set dtype/phase/shape for a transfer
    int genie_set_transfer_metadata(void* data_plane,
                                   const char* transfer_id,
                                   uint8_t dtype_code,
                                   uint8_t phase,
                                   uint8_t shape_rank,
                                   const uint16_t* shape_dims,
                                   size_t dims_len);
    
    // Phase 3: Multi-queue & RSS configuration
    int genie_configure_queues(void* data_plane, uint16_t rx_queues, uint16_t tx_queues, int enable_rss);
    
    // Phase 3: NIC offloads configuration  
    int genie_enable_offloads(void* data_plane, int rx_checksum, int rx_lro, int tx_ipv4_cksum, int tx_udp_cksum, int tx_tso);
    
    // Phase 3: CUDA Graphs integration
    int genie_enable_cuda_graphs(void* data_plane, int enable);
}

} // namespace data_plane
} // namespace genie
