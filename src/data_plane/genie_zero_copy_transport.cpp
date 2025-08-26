/**
 * Zero-Copy Transport Implementation
 */

#include "genie_zero_copy_transport.hpp"
#include <iostream>
#include <cstring>
#include <chrono>
#include <cmath>
#include <rte_cycles.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>

#ifdef GENIE_CUDA_SUPPORT
// CUDA functions are available
#else
// Stub implementations for CUDA functions when CUDA is not available
inline cudaError_t cudaGetDeviceCount(int* count) { *count = 0; return 1; }
inline cudaError_t cudaSetDevice(int) { return 1; }
inline cudaError_t cudaStreamCreate(cudaStream_t*) { return 1; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return 0; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return 0; }
inline cudaError_t cudaMalloc(void**, size_t) { return 1; }
inline cudaError_t cudaFree(void*) { return 0; }
inline cudaError_t cudaMemcpy(void*, const void*, size_t, int) { return 1; }
inline cudaError_t cudaMemcpyAsync(void*, const void*, size_t, int, cudaStream_t) { return 1; }
inline cudaError_t cudaHostRegister(void*, size_t, unsigned int) { return 1; }
inline const char* cudaGetErrorString(cudaError_t) { return "CUDA not available"; }
enum { 
    cudaMemcpyDeviceToHost = 0,
    cudaMemcpyDeviceToDevice = 1,
    cudaHostRegisterDefault = 0
};
#endif

namespace genie {
namespace data_plane {

// ============================================================================
// ZeroCopyTransport Implementation
// ============================================================================

ZeroCopyTransport::ZeroCopyTransport(const Config& config)
    : config_(config), mbuf_pool_(nullptr), extbuf_pool_(nullptr),
      tx_ring_(nullptr), rx_ring_(nullptr), completion_ring_(nullptr),
      gpu_dev_id_(-1), cuda_stream_(nullptr) {
    
    memset(&stats_, 0, sizeof(stats_));
}

ZeroCopyTransport::~ZeroCopyTransport() {
    shutdown();
}

bool ZeroCopyTransport::initialize() {
    std::cout << "Initializing Zero-Copy Transport..." << std::endl;
    
    // Initialize DPDK resources
    if (!init_dpdk_resources()) {
        std::cerr << "Failed to initialize DPDK resources" << std::endl;
        return false;
    }
    
    // Initialize GPU resources
    if (config_.use_gpu_direct) {
        if (!init_gpu_resources()) {
            std::cerr << "Failed to initialize GPU resources, falling back to CPU staging" << std::endl;
            config_.use_gpu_direct = false;
        }
    }
    
    std::cout << "Zero-Copy Transport initialized successfully" << std::endl;
    std::cout << "  GPU Direct: " << (config_.use_gpu_direct ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  External Buffers: " << (config_.use_external_buffers ? "Enabled" : "Disabled") << std::endl;
    
    return true;
}

void ZeroCopyTransport::shutdown() {
    // Clean up active transfers
    {
        std::lock_guard<std::mutex> lock(transfer_mutex_);
        for (auto& [id, transfer] : active_transfers_) {
            transfer->failed = true;
            transfer->completion_promise.set_value(false);
        }
        active_transfers_.clear();
    }
    
    // Clean up GPU resources
    if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }
    
    // Clean up GPU buffers
    for (auto& buffer : gpu_buffer_pool_) {
        if (buffer.gpu_ptr) {
            cudaFree(buffer.gpu_ptr);
        }
    }
    gpu_buffer_pool_.clear();
    
    // Clean up DPDK resources
    if (tx_ring_) {
        rte_ring_free(tx_ring_);
        tx_ring_ = nullptr;
    }
    
    if (rx_ring_) {
        rte_ring_free(rx_ring_);
        rx_ring_ = nullptr;
    }
    
    if (completion_ring_) {
        rte_ring_free(completion_ring_);
        completion_ring_ = nullptr;
    }
    
    if (mbuf_pool_) {
        rte_mempool_free(mbuf_pool_);
        mbuf_pool_ = nullptr;
    }
    
    if (extbuf_pool_) {
        rte_mempool_free(extbuf_pool_);
        extbuf_pool_ = nullptr;
    }
}

bool ZeroCopyTransport::init_dpdk_resources() {
    // Create mbuf pool
    mbuf_pool_ = rte_pktmbuf_pool_create(
        "zero_copy_mbuf_pool",
        8192,  // Number of mbufs
        256,   // Cache size
        0,     // Private data size
        RTE_MBUF_DEFAULT_BUF_SIZE,
        rte_socket_id()
    );
    
    if (!mbuf_pool_) {
        std::cerr << "Failed to create mbuf pool" << std::endl;
        return false;
    }
    
    // Create pool for external buffer shared info
    extbuf_pool_ = rte_mempool_create(
        "extbuf_shinfo_pool",
        1024,  // Number of elements
        sizeof(struct rte_mbuf_ext_shared_info),
        32,    // Cache size
        0,     // Private data size
        nullptr, nullptr,  // No init functions
        nullptr, nullptr,
        rte_socket_id(),
        0      // Flags
    );
    
    if (!extbuf_pool_) {
        std::cerr << "Failed to create external buffer pool" << std::endl;
        return false;
    }
    
    // Create rings for packet queues
    tx_ring_ = rte_ring_create(
        "zero_copy_tx_ring",
        1024,
        rte_socket_id(),
        RING_F_MP_HTS_ENQ | RING_F_MC_HTS_DEQ
    );
    
    rx_ring_ = rte_ring_create(
        "zero_copy_rx_ring",
        1024,
        rte_socket_id(),
        RING_F_MP_HTS_ENQ | RING_F_MC_HTS_DEQ
    );
    
    completion_ring_ = rte_ring_create(
        "zero_copy_completion_ring",
        256,
        rte_socket_id(),
        RING_F_MP_HTS_ENQ | RING_F_MC_HTS_DEQ
    );
    
    if (!tx_ring_ || !rx_ring_ || !completion_ring_) {
        std::cerr << "Failed to create rings" << std::endl;
        return false;
    }
    
    return true;
}

bool ZeroCopyTransport::init_gpu_resources() {
    // Check CUDA availability
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices available" << std::endl;
        return false;
    }
    
    // Set GPU device
    err = cudaSetDevice(config_.gpu_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Create CUDA stream for async operations
    err = cudaStreamCreate(&cuda_stream_);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Initialize DPDK GPU device
    if (rte_gpu_count_avail() > 0) {
        gpu_dev_id_ = rte_gpu_find_next(0, RTE_GPU_ID_ANY);
        if (gpu_dev_id_ >= 0) {
            std::cout << "Found DPDK GPU device: " << gpu_dev_id_ << std::endl;
        }
    }
    
    // Pre-allocate GPU buffer pool
    size_t buffer_size = 64 * 1024 * 1024;  // 64MB buffers
    size_t num_buffers = config_.gpu_buffer_pool_size / buffer_size;
    
    gpu_buffer_pool_.reserve(num_buffers);
    
    for (size_t i = 0; i < num_buffers; i++) {
        GPUBuffer buffer;
        buffer.size = buffer_size;
        buffer.gpu_id = config_.gpu_id;
        buffer.stream = cuda_stream_;
        
        // Allocate GPU memory
        err = cudaMalloc(&buffer.gpu_ptr, buffer.size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate GPU buffer: " << cudaGetErrorString(err) << std::endl;
            break;
        }
        
        // Register for DMA
        if (!register_gpu_memory(buffer)) {
            cudaFree(buffer.gpu_ptr);
            continue;
        }
        
        gpu_buffer_pool_.push_back(buffer);
        available_gpu_buffers_.push(&gpu_buffer_pool_.back());
    }
    
    std::cout << "Allocated " << gpu_buffer_pool_.size() << " GPU buffers" << std::endl;
    
    return !gpu_buffer_pool_.empty();
}

bool ZeroCopyTransport::register_gpu_memory(GPUBuffer& buffer) {
    if (gpu_dev_id_ < 0) {
        // Fallback: Get IOVA through host registration
        cudaHostRegister(buffer.gpu_ptr, buffer.size, cudaHostRegisterDefault);
        buffer.iova = (uint64_t)buffer.gpu_ptr;  // Simplified - would need proper IOVA
        return true;
    }
    
    // Register with DPDK GPU device
    int ret = rte_gpu_mem_register(gpu_dev_id_, buffer.size, buffer.gpu_ptr);
    if (ret < 0) {
        std::cerr << "Failed to register GPU memory with DPDK" << std::endl;
        return false;
    }
    
    // Get IOVA for DMA
    // Note: This is simplified - actual implementation would query the IOVA
    buffer.iova = (uint64_t)buffer.gpu_ptr;
    
    return true;
}

std::future<bool> ZeroCopyTransport::send_tensor_zero_copy(
    void* gpu_ptr, size_t size,
    const std::string& transfer_id,
    uint32_t dest_ip, uint16_t dest_port) {
    
    // Create transfer descriptor
    auto transfer = std::make_unique<TransferDescriptor>();
    transfer->transfer_id = transfer_id;
    transfer->total_size = size;
    transfer->start_tsc = rte_rdtsc();
    
    // Get or create GPU buffer
    GPUBuffer gpu_buffer;
    
    if (config_.use_gpu_direct) {
        // Try to get buffer from pool
        {
            std::lock_guard<std::mutex> lock(gpu_buffer_mutex_);
            if (!available_gpu_buffers_.empty()) {
                gpu_buffer = *available_gpu_buffers_.front();
                available_gpu_buffers_.pop();
            }
        }
        
        if (!gpu_buffer.is_valid()) {
            // Allocate new buffer
            gpu_buffer.size = size;
            gpu_buffer.gpu_id = config_.gpu_id;
            gpu_buffer.stream = cuda_stream_;
            
            cudaError_t err = cudaMalloc(&gpu_buffer.gpu_ptr, size);
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(err) << std::endl;
                // Fall back to CPU staging
                auto promise = std::make_shared<std::promise<bool>>();
                bool result = send_with_cpu_staging(gpu_ptr, size, transfer_id, dest_ip, dest_port);
                promise->set_value(result);
                return promise->get_future();
            }
            
            if (!register_gpu_memory(gpu_buffer)) {
                cudaFree(gpu_buffer.gpu_ptr);
                auto promise = std::make_shared<std::promise<bool>>();
                bool result = send_with_cpu_staging(gpu_ptr, size, transfer_id, dest_ip, dest_port);
                promise->set_value(result);
                return promise->get_future();
            }
        }
        
        // Copy data to buffer
        cudaError_t err = cudaMemcpyAsync(gpu_buffer.gpu_ptr, gpu_ptr, size,
                                         cudaMemcpyDeviceToDevice, cuda_stream_);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy GPU data: " << cudaGetErrorString(err) << std::endl;
            transfer->completion_promise.set_value(false);
            return transfer->completion_promise.get_future();
        }
        
        // Wait for copy to complete
        cudaStreamSynchronize(cuda_stream_);
    } else {
        // Use CPU staging
        auto promise = std::make_shared<std::promise<bool>>();
        bool result = send_with_cpu_staging(gpu_ptr, size, transfer_id, dest_ip, dest_port);
        promise->set_value(result);
        return promise->get_future();
    }
    
    transfer->gpu_buffer = gpu_buffer;
    
    // Get future before moving transfer
    auto future = transfer->completion_promise.get_future();
    
    // Store transfer
    {
        std::lock_guard<std::mutex> lock(transfer_mutex_);
        active_transfers_[transfer_id] = std::move(transfer);
    }
    
    // Fragment and send
    fragment_and_send(gpu_buffer, transfer_id, dest_ip, dest_port);
    
    // Update stats
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.transfers_sent++;
        stats_.zero_copy_sends++;
    }
    
    return future;
}

void ZeroCopyTransport::fragment_and_send(const GPUBuffer& buffer,
                                         const std::string& transfer_id,
                                         uint32_t dest_ip, uint16_t dest_port) {
    
    size_t payload_size = config_.mtu - 64;  // Reserve space for headers
    uint32_t total_fragments = (buffer.size + payload_size - 1) / payload_size;
    
    for (uint32_t frag_id = 0; frag_id < total_fragments; frag_id++) {
        size_t offset = frag_id * payload_size;
        size_t length = std::min(payload_size, buffer.size - offset);
        
        // Create packet with external GPU buffer
        struct rte_mbuf* mbuf = create_packet_with_gpu_memory(
            buffer, offset, length,
            frag_id,  // Use frag_id as seq_num for simplicity
            frag_id, total_fragments,
            transfer_id
        );
        
        if (!mbuf) {
            std::cerr << "Failed to create packet for fragment " << frag_id << std::endl;
            continue;
        }
        
        // Add L2/L3/L4 headers
        struct rte_ether_hdr* eth_hdr = rte_pktmbuf_mtod(mbuf, struct rte_ether_hdr*);
        struct rte_ipv4_hdr* ip_hdr = (struct rte_ipv4_hdr*)(eth_hdr + 1);
        struct rte_udp_hdr* udp_hdr = (struct rte_udp_hdr*)(ip_hdr + 1);
        
        // Fill Ethernet header
        memset(eth_hdr->dst_addr.addr_bytes, 0xFF, RTE_ETHER_ADDR_LEN);  // Broadcast for now
        memset(eth_hdr->src_addr.addr_bytes, 0x00, RTE_ETHER_ADDR_LEN);
        eth_hdr->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);
        
        // Fill IP header
        ip_hdr->version_ihl = (4 << 4) | 5;
        ip_hdr->type_of_service = 0;
        ip_hdr->total_length = rte_cpu_to_be_16(sizeof(*ip_hdr) + sizeof(*udp_hdr) + length);
        ip_hdr->packet_id = 0;
        ip_hdr->fragment_offset = 0;
        ip_hdr->time_to_live = 64;
        ip_hdr->next_proto_id = IPPROTO_UDP;
        ip_hdr->hdr_checksum = 0;
        ip_hdr->src_addr = 0;  // Would set actual source IP
        ip_hdr->dst_addr = dest_ip;
        
        // Fill UDP header
        udp_hdr->src_port = rte_cpu_to_be_16(12345);  // Would use actual port
        udp_hdr->dst_port = rte_cpu_to_be_16(dest_port);
        udp_hdr->dgram_len = rte_cpu_to_be_16(sizeof(*udp_hdr) + length);
        udp_hdr->dgram_cksum = 0;
        
        // Queue for transmission
        if (rte_ring_enqueue(tx_ring_, mbuf) != 0) {
            std::cerr << "TX ring full, dropping packet" << std::endl;
            rte_pktmbuf_free(mbuf);
        }
    }
}

struct rte_mbuf* ZeroCopyTransport::create_packet_with_gpu_memory(
    const GPUBuffer& buffer,
    size_t offset, size_t length,
    uint32_t seq_num, uint32_t frag_id,
    uint32_t total_frags,
    const std::string& transfer_id) {
    
    // Allocate mbuf for headers only
    struct rte_mbuf* mbuf = rte_pktmbuf_alloc(mbuf_pool_);
    if (!mbuf) {
        return nullptr;
    }
    
    // Reserve space for headers
    size_t header_size = sizeof(struct rte_ether_hdr) + 
                        sizeof(struct rte_ipv4_hdr) + 
                        sizeof(struct rte_udp_hdr) + 
                        64;  // Genie header
    
    char* pkt_data = rte_pktmbuf_append(mbuf, header_size);
    if (!pkt_data) {
        rte_pktmbuf_free(mbuf);
        return nullptr;
    }
    
    // Attach GPU memory as external buffer if using zero-copy
    if (config_.use_external_buffers && buffer.is_valid()) {
        if (!attach_external_gpu_buffer(mbuf, buffer, offset, length)) {
            // Fall back to copying
            char* payload = rte_pktmbuf_append(mbuf, length);
            if (payload) {
                // Copy from GPU to packet
                cudaMemcpy(payload, (char*)buffer.gpu_ptr + offset, length,
                          cudaMemcpyDeviceToHost);
                
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.fallback_sends++;
                }
            }
        }
    }
    
    return mbuf;
}

bool ZeroCopyTransport::attach_external_gpu_buffer(struct rte_mbuf* mbuf,
                                                  const GPUBuffer& buffer,
                                                  size_t offset, size_t length) {
    
    // Get shared info structure from pool
    struct rte_mbuf_ext_shared_info* shinfo = nullptr;
    
    if (rte_mempool_get(extbuf_pool_, (void**)&shinfo) != 0) {
        return false;
    }
    
    // Initialize shared info
    shinfo->free_cb = ext_buf_free_cb;
    shinfo->fcb_opaque = (void*)this;
    rte_mbuf_ext_refcnt_set(shinfo, 1);
    
    // Attach external buffer
    rte_pktmbuf_attach_extbuf(
        mbuf,
        (char*)buffer.gpu_ptr + offset,  // Virtual address
        buffer.iova + offset,             // Physical address for DMA
        length,
        shinfo
    );
    
    // Update mbuf fields
    mbuf->data_off = 0;
    mbuf->data_len = length;
    mbuf->pkt_len = length;
    
    return true;
}

void ZeroCopyTransport::ext_buf_free_cb(void* addr, void* opaque) {
    // Called when mbuf with external buffer is freed
    // Return GPU buffer to pool if needed
    ZeroCopyTransport* transport = static_cast<ZeroCopyTransport*>(opaque);
    
    // Find and return buffer to pool
    // Note: Simplified - would need proper tracking
}

bool ZeroCopyTransport::prepare_receive(const std::string& transfer_id, size_t size) {
    // Create transfer descriptor for receiving
    auto transfer = std::make_unique<TransferDescriptor>();
    transfer->transfer_id = transfer_id;
    transfer->total_size = size;
    transfer->start_tsc = rte_rdtsc();
    
    // Allocate GPU buffer for receiving
    if (config_.use_gpu_direct) {
        GPUBuffer gpu_buffer;
        gpu_buffer.size = size;
        gpu_buffer.gpu_id = config_.gpu_id;
        gpu_buffer.stream = cuda_stream_;
        
        cudaError_t err = cudaMalloc(&gpu_buffer.gpu_ptr, size);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate GPU receive buffer" << std::endl;
            return false;
        }
        
        if (!register_gpu_memory(gpu_buffer)) {
            cudaFree(gpu_buffer.gpu_ptr);
            return false;
        }
        
        transfer->gpu_buffer = gpu_buffer;
    } else {
        // Allocate CPU staging buffer
        transfer->cpu_staging = rte_malloc("rx_staging", size, 64);
        if (!transfer->cpu_staging) {
            return false;
        }
    }
    
    // Calculate number of fragments
    size_t payload_size = config_.mtu - 64;
    transfer->total_fragments = (size + payload_size - 1) / payload_size;
    transfer->fragments_received.resize(transfer->total_fragments, false);
    transfer->fragment_mbufs.resize(transfer->total_fragments, nullptr);
    
    // Store transfer
    {
        std::lock_guard<std::mutex> lock(transfer_mutex_);
        active_transfers_[transfer_id] = std::move(transfer);
    }
    
    return true;
}

void ZeroCopyTransport::poll_tx() {
    struct rte_mbuf* mbufs[config_.burst_size];
    
    // Dequeue packets from TX ring
    unsigned nb_tx = rte_ring_dequeue_burst(tx_ring_, (void**)mbufs,
                                           config_.burst_size, nullptr);
    
    if (nb_tx > 0) {
        // Transmit packets
        uint16_t nb_sent = rte_eth_tx_burst(config_.port_id, config_.tx_queue,
                                           mbufs, nb_tx);
        
        // Update statistics
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            for (uint16_t i = 0; i < nb_sent; i++) {
                stats_.bytes_sent += mbufs[i]->pkt_len;
            }
        }
        
        // Free unsent packets
        for (uint16_t i = nb_sent; i < nb_tx; i++) {
            rte_pktmbuf_free(mbufs[i]);
        }
    }
}

void ZeroCopyTransport::poll_rx() {
    struct rte_mbuf* mbufs[config_.burst_size];
    
    // Receive packets
    uint16_t nb_rx = rte_eth_rx_burst(config_.port_id, config_.rx_queue,
                                     mbufs, config_.burst_size);
    
    for (uint16_t i = 0; i < nb_rx; i++) {
        // Parse packet header to get transfer ID and fragment info
        // Note: Simplified - would need actual protocol parsing
        
        // For now, just free the packets
        rte_pktmbuf_free(mbufs[i]);
    }
    
    // Update statistics
    if (nb_rx > 0) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.bytes_received += nb_rx * config_.mtu;  // Approximate
    }
}

void ZeroCopyTransport::process_completions() {
    // Check for completed transfers
    std::vector<std::string> completed_transfers;
    
    {
        std::lock_guard<std::mutex> lock(transfer_mutex_);
        
        for (auto& [id, transfer] : active_transfers_) {
            if (reassemble_fragments(*transfer)) {
                completed_transfers.push_back(id);
                transfer->end_tsc = rte_rdtsc();
                transfer->completed = true;
                transfer->completion_promise.set_value(true);
            }
        }
    }
    
    // Clean up completed transfers
    for (const auto& id : completed_transfers) {
        complete_transfer(id, true);
    }
}

bool ZeroCopyTransport::reassemble_fragments(TransferDescriptor& transfer) {
    // Check if all fragments received
    for (bool received : transfer.fragments_received) {
        if (!received) {
            return false;
        }
    }
    
    // All fragments received
    return true;
}

void ZeroCopyTransport::complete_transfer(const std::string& transfer_id, bool success) {
    std::lock_guard<std::mutex> lock(transfer_mutex_);
    
    auto it = active_transfers_.find(transfer_id);
    if (it == active_transfers_.end()) {
        return;
    }
    
    auto& transfer = it->second;
    
    // Calculate statistics
    if (success) {
        uint64_t duration_cycles = transfer->end_tsc - transfer->start_tsc;
        double duration_s = static_cast<double>(duration_cycles) / rte_get_tsc_hz();
        double throughput_gbps = (transfer->total_size * 8.0) / (duration_s * 1e9);
        
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.transfers_received++;
            stats_.avg_throughput_gbps = 
                (stats_.avg_throughput_gbps * (stats_.transfers_received - 1) + throughput_gbps) /
                stats_.transfers_received;
        }
    }
    
    // Return GPU buffer to pool if needed
    if (transfer->gpu_buffer.is_valid()) {
        std::lock_guard<std::mutex> lock(gpu_buffer_mutex_);
        available_gpu_buffers_.push(&transfer->gpu_buffer);
    }
    
    // Clean up CPU staging buffer
    if (transfer->cpu_staging) {
        rte_free(transfer->cpu_staging);
    }
    
    // Remove from active transfers
    active_transfers_.erase(it);
}

bool ZeroCopyTransport::send_with_cpu_staging(void* gpu_ptr, size_t size,
                                             const std::string& transfer_id,
                                             uint32_t dest_ip, uint16_t dest_port) {
    
    // Allocate CPU staging buffer
    void* cpu_buffer = rte_malloc("tx_staging", size, 64);
    if (!cpu_buffer) {
        return false;
    }
    
    // Copy from GPU to CPU
    cudaError_t err = cudaMemcpy(cpu_buffer, gpu_ptr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        rte_free(cpu_buffer);
        return false;
    }
    
    // Send using CPU buffer
    // Note: Simplified - would fragment and send
    
    rte_free(cpu_buffer);
    
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.fallback_sends++;
    }
    
    return true;
}

ZeroCopyTransport::Stats ZeroCopyTransport::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void ZeroCopyTransport::print_stats() const {
    Stats s = get_stats();
    
    std::cout << "\n=== Zero-Copy Transport Statistics ===" << std::endl;
    std::cout << "Transfers sent:     " << s.transfers_sent << std::endl;
    std::cout << "Transfers received: " << s.transfers_received << std::endl;
    std::cout << "Bytes sent:         " << s.bytes_sent << std::endl;
    std::cout << "Bytes received:     " << s.bytes_received << std::endl;
    std::cout << "Zero-copy sends:    " << s.zero_copy_sends << std::endl;
    std::cout << "Fallback sends:     " << s.fallback_sends << std::endl;
    std::cout << "DMA errors:         " << s.dma_errors << std::endl;
    std::cout << "Avg throughput:     " << s.avg_throughput_gbps << " Gbps" << std::endl;
    std::cout << "Avg latency:        " << s.avg_latency_us << " µs" << std::endl;
}

// ============================================================================
// ReliabilityManager Implementation
// ============================================================================

ReliabilityManager::ReliabilityManager(const Config& config)
    : config_(config),
      srtt_ms_(config.initial_rtt_ms),
      rttvar_ms_(config.initial_rtt_ms / 2),
      rto_ms_(config.initial_rtt_ms * 3) {
    
    memset(&stats_, 0, sizeof(stats_));
    stats_.min_rtt_ms = 999999;
}

ReliabilityManager::~ReliabilityManager() {}

void ReliabilityManager::track_packet(struct rte_mbuf* mbuf, uint32_t seq_num,
                                     const std::string& transfer_id) {
    std::lock_guard<std::mutex> lock(packet_mutex_);
    
    PacketInfo info;
    info.mbuf = mbuf;
    info.seq_num = seq_num;
    info.transfer_id = transfer_id;
    info.send_time_tsc = rte_rdtsc();
    info.retry_count = 0;
    info.acked = false;
    
    unacked_packets_[seq_num] = info;
    
    stats_.packets_sent++;
}

void ReliabilityManager::process_ack(uint32_t ack_num, const std::string& transfer_id) {
    std::lock_guard<std::mutex> lock(packet_mutex_);
    
    auto it = unacked_packets_.find(ack_num);
    if (it != unacked_packets_.end()) {
        // Calculate RTT
        uint64_t now_tsc = rte_rdtsc();
        uint64_t rtt_cycles = now_tsc - it->second.send_time_tsc;
        double rtt_ms = (rtt_cycles * 1000.0) / rte_get_tsc_hz();
        
        update_rtt(rtt_ms);
        
        // Mark as ACKed
        it->second.acked = true;
        
        // Free mbuf if not needed for retransmission
        if (it->second.mbuf) {
            rte_pktmbuf_free(it->second.mbuf);
        }
        
        unacked_packets_.erase(it);
        stats_.packets_acked++;
    }
}

void ReliabilityManager::process_nack(uint32_t seq_num, const std::string& transfer_id) {
    std::lock_guard<std::mutex> lock(packet_mutex_);
    
    auto it = unacked_packets_.find(seq_num);
    if (it != unacked_packets_.end()) {
        // Immediate retransmission
        if (it->second.mbuf && it->second.retry_count < config_.max_retries) {
            // Clone packet for retransmission
            struct rte_mbuf* clone = rte_pktmbuf_clone(it->second.mbuf, 
                                                       it->second.mbuf->pool);
            if (clone) {
                // Queue for retransmission
                // Note: Would enqueue to TX ring
                it->second.retry_count++;
                stats_.retransmissions++;
            }
        }
        
        stats_.packets_nacked++;
    }
}

std::vector<struct rte_mbuf*> ReliabilityManager::check_timeouts() {
    std::vector<struct rte_mbuf*> retransmit_packets;
    uint64_t now_tsc = rte_rdtsc();
    uint64_t timeout_cycles = (rto_ms_ * rte_get_tsc_hz()) / 1000;
    
    std::lock_guard<std::mutex> lock(packet_mutex_);
    
    for (auto& [seq_num, info] : unacked_packets_) {
        if (!info.acked && (now_tsc - info.send_time_tsc) > timeout_cycles) {
            if (info.retry_count < config_.max_retries) {
                // Clone packet for retransmission
                if (info.mbuf) {
                    struct rte_mbuf* clone = rte_pktmbuf_clone(info.mbuf,
                                                               info.mbuf->pool);
                    if (clone) {
                        retransmit_packets.push_back(clone);
                        info.retry_count++;
                        info.send_time_tsc = now_tsc;  // Reset timer
                        stats_.retransmissions++;
                    }
                }
                
                stats_.timeouts++;
            }
        }
    }
    
    return retransmit_packets;
}

void ReliabilityManager::update_rtt(double sample_rtt_ms) {
    // TCP's RTT estimation algorithm (RFC 6298)
    const double alpha = 0.125;
    const double beta = 0.25;
    
    double rttvar = (1 - beta) * rttvar_ms_ + beta * std::abs(srtt_ms_ - sample_rtt_ms);
    double srtt = (1 - alpha) * srtt_ms_ + alpha * sample_rtt_ms;
    
    srtt_ms_ = srtt;
    rttvar_ms_ = rttvar;
    
    // Calculate RTO
    rto_ms_ = srtt_ms_ + 4 * rttvar_ms_;
    
    // Enforce minimum and maximum
    rto_ms_ = std::max(config_.timeout_ms / 10.0, rto_ms_);
    rto_ms_ = std::min(config_.timeout_ms * 10.0, rto_ms_);
    
    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.avg_rtt_ms = srtt_ms_;
        stats_.min_rtt_ms = std::min(stats_.min_rtt_ms, sample_rtt_ms);
        stats_.max_rtt_ms = std::max(stats_.max_rtt_ms, sample_rtt_ms);
    }
}

// ============================================================================
// FlowControlManager Implementation
// ============================================================================

FlowControlManager::FlowControlManager(const Config& config)
    : config_(config),
      cwnd_(config.initial_cwnd),
      ssthresh_(config.ssthresh),
      in_flight_(0),
      state_(SLOW_START),
      w_max_(0),
      epoch_start_(0) {}

FlowControlManager::~FlowControlManager() {}

bool FlowControlManager::can_send() const {
    // Check congestion window
    if (in_flight_ >= cwnd_) {
        return false;
    }
    
    // Check rate limit if enabled
    if (rate_limited_ && !check_rate_limit()) {
        return false;
    }
    
    return true;
}

void FlowControlManager::on_packet_sent(uint32_t seq_num) {
    in_flight_++;
    total_sent_++;
    
    // Update rate limiter
    if (rate_limited_) {
        last_send_time_ = rte_rdtsc();
    }
}

void FlowControlManager::on_ack_received(uint32_t ack_num) {
    if (in_flight_ > 0) {
        in_flight_--;
    }
    
    total_acked_++;
    
    // Update congestion window based on state
    switch (state_.load()) {
        case SLOW_START:
            // Exponential growth
            cwnd_++;
            if (cwnd_ >= ssthresh_) {
                state_ = CONGESTION_AVOIDANCE;
            }
            break;
            
        case CONGESTION_AVOIDANCE:
            // Linear growth (increase by 1 per RTT)
            if (config_.enable_cubic) {
                // CUBIC algorithm
                uint64_t t = rte_rdtsc() - epoch_start_;
                uint32_t cubic_cwnd = cubic_window(t);
                cwnd_ = cubic_cwnd;
            } else {
                // Reno: increase by 1/cwnd per ACK
                static uint32_t ack_count = 0;
                ack_count++;
                if (ack_count >= cwnd_) {
                    cwnd_++;
                    ack_count = 0;
                }
            }
            break;
            
        case FAST_RECOVERY:
            // Stay in fast recovery until all lost packets are recovered
            break;
            
        case TIMEOUT:
            // Reset to slow start
            state_ = SLOW_START;
            cwnd_ = config_.initial_cwnd;
            break;
    }
    
    // Enforce maximum window
    if (cwnd_.load() > config_.max_cwnd) {
        cwnd_.store(config_.max_cwnd);
    }
}

void FlowControlManager::on_loss_detected() {
    // Save current window size
    w_max_ = cwnd_.load();
    epoch_start_ = rte_rdtsc();
    
    // Multiplicative decrease
    if (config_.enable_cubic) {
        cwnd_.store(static_cast<uint32_t>(cwnd_.load() * cubic_beta_));
    } else {
        cwnd_.store(cwnd_.load() / 2);
    }
    
    ssthresh_.store(cwnd_.load());
    state_ = FAST_RECOVERY;
}

void FlowControlManager::on_timeout() {
    // Severe congestion - reset to minimum
    cwnd_.store(config_.initial_cwnd);
    ssthresh_.store(cwnd_.load() * 2);
    state_ = TIMEOUT;
    epoch_start_ = rte_rdtsc();
}

uint32_t FlowControlManager::cubic_window(uint64_t t) const {
    // CUBIC window function: W(t) = C*(t-K)^3 + W_max
    // where K = cubic_root(W_max * beta / C)
    
    double t_sec = static_cast<double>(t) / rte_get_tsc_hz();
    double K = std::cbrt(w_max_ * (1 - cubic_beta_) / cubic_c_);
    
    double w_cubic = cubic_c_ * std::pow(t_sec - K, 3) + w_max_;
    
    return static_cast<uint32_t>(w_cubic);
}

bool FlowControlManager::check_rate_limit() const {
    if (rate_limit_gbps_ <= 0) {
        return true;
    }
    
    // Token bucket algorithm
    uint64_t now = rte_rdtsc();
    uint64_t elapsed = now - last_send_time_;
    double elapsed_sec = static_cast<double>(elapsed) / rte_get_tsc_hz();
    
    // Calculate tokens accumulated
    double bits_allowed = rate_limit_gbps_ * 1e9 * elapsed_sec;
    uint64_t new_tokens = static_cast<uint64_t>(bits_allowed / 8);  // Convert to bytes
    
    // Simple check - would need proper token bucket implementation
    return new_tokens > config_.initial_cwnd * 1500;  // MTU size
}

FlowControlManager::Stats FlowControlManager::get_stats() const {
    Stats stats;
    stats.current_cwnd = cwnd_;
    stats.current_ssthresh = ssthresh_;
    stats.in_flight = in_flight_;
    stats.state = state_;
    stats.total_sent = total_sent_;
    stats.total_acked = total_acked_;
    
    // Calculate throughput
    if (total_acked_ > 0) {
        // Simplified - would need actual time tracking
        stats.throughput_gbps = 0;  // Placeholder
    }
    
    return stats;
}

} // namespace data_plane
} // namespace genie

