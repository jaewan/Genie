/**
 * Genie Zero-Copy Data Plane Implementation
 * 
 * Primary transport implementation using DPDK + GPUDev for zero-copy tensor
 * transfers in commodity cloud environments. This is the DEFAULT transport,
 * not a fallback - it works without special hardware (no RNICs required).
 * 
 * Architecture:
 * - DPDK for userspace networking (works with standard NICs)
 * - GPUDev for GPU memory registration and DMA
 * - Custom reliable protocol over UDP (software reliability)
 * - Optimized for cloud environments (AWS, GCP, Azure)
 */

#include "genie_data_plane.hpp"
#include "genie_dpdk_thread_model.hpp"  // Use DPDK native threading
#include "genie_kcp_wrapper.hpp"
#include <iostream>
#include <cstring>
#include <cstdlib>  // for aligned_alloc
#include <chrono>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <map>      // for std::multimap

// JSON library for configuration parsing
#include <nlohmann/json.hpp>
using json = nlohmann::json;

// Network byte order conversion
#include <arpa/inet.h>

namespace genie {
namespace data_plane {

// Constants
static constexpr uint32_t GENIE_MAGIC = 0x47454E49; // "GENI"
static constexpr uint8_t GENIE_VERSION = 1;
static constexpr uint16_t DEFAULT_MTU = 1500;
static constexpr uint32_t MAX_PAYLOAD_SIZE = DEFAULT_MTU - sizeof(GeniePacket);

/**
 * Enhanced GPU memory manager with GPUDev optimization
 * 
 * This is the PRIMARY method for GPU memory management in DPDK transport.
 * Optimized for cloud environments with standard GPUs (V100, A100, H100).
 */
class GPUMemoryManager {
public:
    GPUMemoryManager(int device_id) : device_id_(device_id), gpudev_available_(false) {
        init_gpudev();
        init_memory_pools();
    }
    
    ~GPUMemoryManager() {
        cleanup_memory_pools();
    }
    
    bool register_memory(void* gpu_ptr, size_t size, GPUMemoryHandle& handle) {
        handle.gpu_ptr = gpu_ptr;
        handle.size = size;
        handle.gpu_id = device_id_;
        handle.ref_count = 1;
        
        // Try GPUDev registration first (primary path)
        if (gpudev_available_) {
            // Register with GPUDev for DMA
            int ret = rte_gpu_mem_register(device_id_, size, gpu_ptr);
            if (ret == 0) {
                // Get IOVA for DMA operations
                // Note: In DPDK 23.11+, use rte_gpu_mem_get_iova if available
                handle.iova = get_gpu_iova(gpu_ptr, size);
                handle.is_registered = true;
                
                // Cache registration for reuse
                registration_cache_[gpu_ptr] = handle;
                
                std::cout << "GPU memory registered: ptr=" << gpu_ptr 
                          << ", size=" << size << ", iova=0x" << std::hex 
                          << handle.iova << std::dec << std::endl;
                return true;
            } else {
                std::cerr << "GPUDev registration failed: " << rte_strerror(-ret) << std::endl;
            }
        }
        
        // Fallback: CPU-staged transfers (still better than TCP)
        if (!handle.is_registered) {
            // Allocate pinned CPU buffer as staging area
            void* cpu_staging = allocate_pinned_memory(size);
            if (cpu_staging) {
                handle.iova = reinterpret_cast<uint64_t>(cpu_staging);
                handle.is_registered = false;  // Mark as staged
                staging_buffers_[gpu_ptr] = cpu_staging;
                
                std::cout << "Using CPU staging buffer for GPU memory" << std::endl;
                return true;
            }
        }
        
        return false;
    }
    
    void unregister_memory(const GPUMemoryHandle& handle) {
        if (gpudev_available_ && handle.is_registered) {
            rte_gpu_mem_unregister(handle.gpu_id, handle.gpu_ptr);
            registration_cache_.erase(handle.gpu_ptr);
        }
        
        // Clean up staging buffer if used
        auto it = staging_buffers_.find(handle.gpu_ptr);
        if (it != staging_buffers_.end()) {
            free_pinned_memory(it->second, handle.size);
            staging_buffers_.erase(it);
        }
    }
    
    bool is_available() const { return gpudev_available_; }
    
    // Check if a memory region is already registered
    bool is_registered(void* gpu_ptr) const {
        return registration_cache_.find(gpu_ptr) != registration_cache_.end();
    }
    
    // Pre-allocate GPU memory pool for receives
    void* allocate_gpu_buffer(size_t size) {
        if (!gpudev_available_) {
            return nullptr;
        }
        
        // Try to get from pool first
        auto it = gpu_memory_pool_.lower_bound(size);
        if (it != gpu_memory_pool_.end()) {
            void* buffer = it->second;
            gpu_memory_pool_.erase(it);
            return buffer;
        }
        
        // Allocate new GPU buffer
        void* buffer = nullptr;
#ifdef GENIE_CUDA_SUPPORT
        cudaError_t err = cudaMalloc(&buffer, size);
        if (err != cudaSuccess) {
            std::cerr << "GPU allocation failed: " << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }
#else
        // Use GPUDev allocation if available
        buffer = rte_gpu_mem_alloc(device_id_, size, 0);
#endif
        return buffer;
    }
    
    void free_gpu_buffer(void* buffer, size_t size) {
        if (!buffer) return;
        
        // Return to pool for reuse (up to max pool size)
        if (gpu_memory_pool_.size() < MAX_POOL_BUFFERS) {
            gpu_memory_pool_.insert({size, buffer});
        } else {
#ifdef GENIE_CUDA_SUPPORT
            cudaFree(buffer);
#else
            rte_gpu_mem_free(device_id_, buffer);
#endif
        }
    }
    
private:
    void init_gpudev() {
        // Initialize GPUDev library
        int ret = rte_gpu_init(16);  // Max 16 GPUs
        if (ret < 0) {
            std::cerr << "Failed to initialize GPUDev: " << rte_strerror(-ret) << std::endl;
            return;
        }
        
        // Check available GPUs
        int gpu_count = rte_gpu_count_avail();
        if (gpu_count > 0 && device_id_ < gpu_count) {
            // Get GPU info
            struct rte_gpu_info gpu_info;
            ret = rte_gpu_info_get(device_id_, &gpu_info);
            if (ret == 0) {
                gpudev_available_ = true;
                std::cout << "GPUDev initialized: " << gpu_count << " GPUs available" << std::endl;
                std::cout << "GPU " << device_id_ << ": " << gpu_info.name 
                          << ", Memory: " << gpu_info.total_memory / (1024*1024*1024) << " GB" << std::endl;
            }
        }
        
        if (!gpudev_available_) {
            std::cout << "GPUDev not available, using CPU staging (still better than TCP)" << std::endl;
        }
    }
    
    void init_memory_pools() {
        // Pre-allocate common buffer sizes
        const size_t pool_sizes[] = {
            1 * 1024 * 1024,      // 1 MB
            10 * 1024 * 1024,     // 10 MB
            100 * 1024 * 1024,    // 100 MB
            1024 * 1024 * 1024    // 1 GB
        };
        
        for (size_t size : pool_sizes) {
            void* buffer = allocate_gpu_buffer(size);
            if (buffer) {
                gpu_memory_pool_.insert({size, buffer});
            }
        }
    }
    
    void cleanup_memory_pools() {
        for (auto& [size, buffer] : gpu_memory_pool_) {
#ifdef GENIE_CUDA_SUPPORT
            cudaFree(buffer);
#else
            if (gpudev_available_) {
                rte_gpu_mem_free(device_id_, buffer);
            }
#endif
        }
        gpu_memory_pool_.clear();
    }
    
    uint64_t get_gpu_iova(void* gpu_ptr, size_t size) {
        // In production, this would call rte_gpu_mem_get_iova or similar
        // For now, we need to handle different GPU/driver combinations
        
#ifdef GENIE_CUDA_SUPPORT
        // Try to get physical address via CUDA
        CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(gpu_ptr);
        // Note: Requires CUDA driver API for physical address query
        // This is simplified - production would need proper CUDA driver calls
        return static_cast<uint64_t>(dptr);
#else
        // Fallback: use virtual address (requires IOMMU in passthrough mode)
        return reinterpret_cast<uint64_t>(gpu_ptr);
#endif
    }
    
    void* allocate_pinned_memory(size_t size) {
        void* ptr = nullptr;
#ifdef GENIE_CUDA_SUPPORT
        cudaError_t err = cudaMallocHost(&ptr, size);
        if (err != cudaSuccess) {
            ptr = nullptr;
        }
#else
        // Use hugepage-backed memory for better DMA performance
        ptr = aligned_alloc(2 * 1024 * 1024, size);  // 2MB alignment for hugepages
#endif
        return ptr;
    }
    
    void free_pinned_memory(void* ptr, size_t size) {
#ifdef GENIE_CUDA_SUPPORT
        cudaFreeHost(ptr);
#else
        free(ptr);
#endif
    }
    
    static constexpr size_t MAX_POOL_BUFFERS = 16;
    
    int device_id_;
    bool gpudev_available_;
    
    // Registration cache for reuse
    std::unordered_map<void*, GPUMemoryHandle> registration_cache_;
    
    // CPU staging buffers for fallback path
    std::unordered_map<void*, void*> staging_buffers_;
    
    // Pre-allocated GPU memory pool
    std::multimap<size_t, void*> gpu_memory_pool_;
};

/**
 * Packet processor for building and parsing packets
 */
class PacketProcessor {
public:
    PacketProcessor(const std::string& local_ip, const std::string& local_mac, uint16_t data_port)
        : local_ip_(inet_addr(local_ip.c_str())), data_port_(htons(data_port)) {
        parse_mac_address(local_mac, local_mac_);
    }
    
    struct rte_mbuf* build_data_packet(struct rte_mempool* pool,
                                      const std::string& target_ip,
                                      const std::string& target_mac,
                                      const std::string& tensor_id,
                                      uint32_t seq_num,
                                      const uint8_t* payload,
                                      uint32_t payload_size,
                                      uint16_t frag_id = 0,
                                      uint16_t frag_count = 1,
                                      uint32_t offset = 0,
                                      PacketFlags flags = PacketFlags::NONE) {
        
        // Allocate mbuf
        struct rte_mbuf* pkt = rte_pktmbuf_alloc(pool);
        if (!pkt) {
            return nullptr;
        }
        
        // Calculate total packet size
        uint32_t total_size = sizeof(GeniePacket) + payload_size;
        
        // Ensure we have enough space
        if (total_size > rte_pktmbuf_tailroom(pkt)) {
            rte_pktmbuf_free(pkt);
            return nullptr;
        }
        
        // Get packet buffer
        GeniePacket* genie_pkt = rte_pktmbuf_mtod(pkt, GeniePacket*);
        
        // Build Ethernet header
        parse_mac_address(target_mac, genie_pkt->eth.dst_mac);
        std::memcpy(genie_pkt->eth.src_mac, local_mac_, 6);
        genie_pkt->eth.ethertype = htons(0x0800); // IPv4
        
        // Build IPv4 header
        genie_pkt->ip.version_ihl = 0x45; // IPv4, 20 byte header
        genie_pkt->ip.tos = 0;
        genie_pkt->ip.total_length = htons(sizeof(IPv4Header) + sizeof(UDPHeader) + sizeof(GeniePacketHeader) + payload_size);
        genie_pkt->ip.identification = htons(seq_num & 0xFFFF);
        genie_pkt->ip.flags_fragment = 0;
        genie_pkt->ip.ttl = 64;
        genie_pkt->ip.protocol = 17; // UDP
        genie_pkt->ip.checksum = 0; // Will be calculated later
        genie_pkt->ip.src_ip = local_ip_;
        genie_pkt->ip.dst_ip = inet_addr(target_ip.c_str());
        
        // Calculate IP checksum
        genie_pkt->ip.checksum = calculate_ip_checksum(&genie_pkt->ip);
        
        // Build UDP header
        genie_pkt->udp.src_port = data_port_;
        genie_pkt->udp.dst_port = data_port_;
        genie_pkt->udp.length = htons(sizeof(UDPHeader) + sizeof(GeniePacketHeader) + payload_size);
        genie_pkt->udp.checksum = 0; // Optional for IPv4
        
        // Build Genie application header
        genie_pkt->app.magic = htonl(GENIE_MAGIC);
        genie_pkt->app.version = GENIE_VERSION;
        genie_pkt->app.flags = static_cast<uint8_t>(flags);
        genie_pkt->app.type = static_cast<uint8_t>(PacketType::DATA);
        genie_pkt->app.reserved = 0;
        
        // Copy tensor ID (first 16 bytes of UUID string)
        std::memset(genie_pkt->app.tensor_id, 0, 16);
        std::strncpy(reinterpret_cast<char*>(genie_pkt->app.tensor_id), tensor_id.c_str(), 15);
        
        genie_pkt->app.seq_num = htonl(seq_num);
        genie_pkt->app.frag_id = htons(frag_id);
        genie_pkt->app.frag_count = htons(frag_count);
        genie_pkt->app.offset = htonl(offset);
        genie_pkt->app.length = htonl(payload_size);
        genie_pkt->app.total_size = htonl(offset + payload_size); // Will be updated for fragments
        genie_pkt->app.timestamp_ns = htobe64(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count());
        
        // Calculate payload checksum
        genie_pkt->app.checksum = htonl(calculate_checksum(payload, payload_size));
        
        // Populate semantic metadata (best-effort defaults)
        genie_pkt->app.dtype_code = 0;  // unknown; Python can set via metadata later
        genie_pkt->app.phase = 3;       // other
        genie_pkt->app.shape_rank = 0;
        std::memset(genie_pkt->app.shape_dims, 0, sizeof(genie_pkt->app.shape_dims));
        
        // Copy payload
        if (payload_size > 0) {
            uint8_t* payload_ptr = reinterpret_cast<uint8_t*>(genie_pkt + 1);
            std::memcpy(payload_ptr, payload, payload_size);
        }
        
        // Set packet length
        pkt->data_len = total_size;
        pkt->pkt_len = total_size;
        
        return pkt;
    }
    
    bool parse_packet(struct rte_mbuf* pkt, GeniePacket& parsed_pkt, uint8_t*& payload, uint32_t& payload_size) {
        // Check minimum packet size
        if (pkt->data_len < sizeof(GeniePacket)) {
            return false;
        }
        
        // Get packet data
        GeniePacket* genie_pkt = rte_pktmbuf_mtod(pkt, GeniePacket*);
        
        // Copy headers
        std::memcpy(&parsed_pkt, genie_pkt, sizeof(GeniePacket));
        
        // Validate magic number
        if (ntohl(parsed_pkt.app.magic) != GENIE_MAGIC) {
            return false;
        }
        
        // Validate version
        if (parsed_pkt.app.version != GENIE_VERSION) {
            return false;
        }
        
        // Extract payload
        payload_size = ntohl(parsed_pkt.app.length);
        if (payload_size > 0) {
            if (pkt->data_len < sizeof(GeniePacket) + payload_size) {
                return false;
            }
            payload = reinterpret_cast<uint8_t*>(genie_pkt + 1);
        } else {
            payload = nullptr;
        }
        
        return true;
    }
    
    struct rte_mbuf* build_ack_packet(struct rte_mempool* pool,
                                      const std::string& target_ip,
                                      const std::string& target_mac,
                                      uint32_t seq_num) {
        // Build a minimal ACK packet
        struct rte_mbuf* pkt = rte_pktmbuf_alloc(pool);
        if (!pkt) return nullptr;
        
        uint32_t total_size = sizeof(GeniePacket);
        if (total_size > rte_pktmbuf_tailroom(pkt)) {
            rte_pktmbuf_free(pkt);
            return nullptr;
        }
        
        GeniePacket* genie_pkt = rte_pktmbuf_mtod(pkt, GeniePacket*);
        
        // Build headers (similar to data packet but with ACK type)
        parse_mac_address(target_mac.empty() ? "ff:ff:ff:ff:ff:ff" : target_mac, genie_pkt->eth.dst_mac);
        std::memcpy(genie_pkt->eth.src_mac, local_mac_, 6);
        genie_pkt->eth.ethertype = htons(0x0800);
        
        genie_pkt->ip.version_ihl = 0x45;
        genie_pkt->ip.tos = 0;
        genie_pkt->ip.total_length = htons(sizeof(IPv4Header) + sizeof(UDPHeader) + sizeof(GeniePacketHeader));
        genie_pkt->ip.identification = htons(seq_num & 0xFFFF);
        genie_pkt->ip.flags_fragment = 0;
        genie_pkt->ip.ttl = 64;
        genie_pkt->ip.protocol = 17;
        genie_pkt->ip.checksum = 0;
        genie_pkt->ip.src_ip = local_ip_;
        genie_pkt->ip.dst_ip = inet_addr(target_ip.c_str());
        genie_pkt->ip.checksum = calculate_ip_checksum(&genie_pkt->ip);
        
        genie_pkt->udp.src_port = data_port_;
        genie_pkt->udp.dst_port = data_port_;
        genie_pkt->udp.length = htons(sizeof(UDPHeader) + sizeof(GeniePacketHeader));
        genie_pkt->udp.checksum = 0;
        
        // ACK-specific header
        genie_pkt->app.magic = htonl(GENIE_MAGIC);
        genie_pkt->app.version = GENIE_VERSION;
        genie_pkt->app.flags = 0;
        genie_pkt->app.type = static_cast<uint8_t>(PacketType::ACK);  // Set ACK type
        genie_pkt->app.reserved = 0;
        std::memset(genie_pkt->app.tensor_id, 0, 16);
        genie_pkt->app.seq_num = htonl(seq_num);  // ACK for this sequence number
        genie_pkt->app.frag_id = 0;
        genie_pkt->app.frag_count = 0;
        genie_pkt->app.offset = 0;
        genie_pkt->app.length = 0;
        genie_pkt->app.total_size = 0;
        genie_pkt->app.checksum = 0;
        genie_pkt->app.timestamp_ns = htobe64(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count());
        // Initialize semantic fields for ACK path
        genie_pkt->app.dtype_code = 0;
        genie_pkt->app.phase = 3;
        genie_pkt->app.shape_rank = 0;
        std::memset(genie_pkt->app.shape_dims, 0, sizeof(genie_pkt->app.shape_dims));
        
        pkt->data_len = total_size;
        pkt->pkt_len = total_size;
        
        return pkt;
    }
    
    struct rte_mbuf* build_nack_packet(struct rte_mempool* pool,
                                       const std::string& target_ip,
                                       const std::string& target_mac,
                                       uint32_t seq_num,
                                       uint16_t missing_frag_id) {
        // Similar to ACK but with NACK type and fragment info
        struct rte_mbuf* pkt = build_ack_packet(pool, target_ip, target_mac, seq_num);
        if (pkt) {
            GeniePacket* genie_pkt = rte_pktmbuf_mtod(pkt, GeniePacket*);
            genie_pkt->app.type = static_cast<uint8_t>(PacketType::NACK);
            genie_pkt->app.frag_id = htons(missing_frag_id);  // Indicate which fragment is missing
        }
        return pkt;
    }
    
private:
    void parse_mac_address(const std::string& mac_str, uint8_t mac[6]) {
        std::sscanf(mac_str.c_str(), "%hhx:%hhx:%hhx:%hhx:%hhx:%hhx",
                   &mac[0], &mac[1], &mac[2], &mac[3], &mac[4], &mac[5]);
    }
    
    uint16_t calculate_ip_checksum(const IPv4Header* ip_hdr) {
        uint32_t sum = 0;
        const uint16_t* ptr = reinterpret_cast<const uint16_t*>(ip_hdr);
        
        // Sum all 16-bit words in header (excluding checksum field)
        for (int i = 0; i < 10; i++) {
            if (i != 5) { // Skip checksum field
                sum += ntohs(ptr[i]);
            }
        }
        
        // Add carry
        while (sum >> 16) {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        
        return htons(~sum);
    }
    
    uint32_t calculate_checksum(const uint8_t* data, size_t size) {
        uint32_t checksum = 0;
        for (size_t i = 0; i < size; i++) {
            checksum += data[i];
        }
        return checksum;
    }
    
    uint32_t local_ip_;
    uint8_t local_mac_[6];
    uint16_t data_port_;
};

/**
 * Reliability manager for ACK/NACK handling
 */
class ReliabilityManager {
public:
    ReliabilityManager(uint32_t timeout_ms, uint32_t max_retries)
        : timeout_ms_(timeout_ms), max_retries_(max_retries), next_seq_num_(1) {}
    
    uint32_t get_next_sequence() {
        return next_seq_num_.fetch_add(1);
    }
    
    void add_pending_packet(uint32_t seq_num, struct rte_mbuf* pkt) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_packets_[seq_num] = {pkt, std::chrono::steady_clock::now(), 0};
    }
    
    bool handle_ack(uint32_t seq_num) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pending_packets_.find(seq_num);
        if (it != pending_packets_.end()) {
            rte_pktmbuf_free(it->second.pkt);
            pending_packets_.erase(it);
            return true;
        }
        return false;
    }
    
    std::vector<struct rte_mbuf*> check_timeouts() {
        std::vector<struct rte_mbuf*> retransmit_list;
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto now = std::chrono::steady_clock::now();
        auto timeout = std::chrono::milliseconds(timeout_ms_);
        
        for (auto it = pending_packets_.begin(); it != pending_packets_.end();) {
            if (now - it->second.timestamp > timeout) {
                if (it->second.retry_count < max_retries_) {
                    // Clone packet for retransmission
                    struct rte_mbuf* clone = rte_pktmbuf_clone(it->second.pkt, it->second.pkt->pool);
                    if (clone) {
                        retransmit_list.push_back(clone);
                        it->second.retry_count++;
                        it->second.timestamp = now;
                    }
                    ++it;
                } else {
                    // Max retries exceeded
                    rte_pktmbuf_free(it->second.pkt);
                    it = pending_packets_.erase(it);
                }
            } else {
                ++it;
            }
        }
        
        return retransmit_list;
    }
    
private:
    struct PendingPacket {
        struct rte_mbuf* pkt;
        std::chrono::steady_clock::time_point timestamp;
        uint32_t retry_count;
    };
    
    std::mutex mutex_;
    std::unordered_map<uint32_t, PendingPacket> pending_packets_;
    uint32_t timeout_ms_;
    uint32_t max_retries_;
    std::atomic<uint32_t> next_seq_num_;
};

/**
 * Flow controller for congestion control with dynamic window adjustment
 */
class FlowController {
public:
    FlowController(uint32_t initial_window) 
        : cwnd_(initial_window), ssthresh_(64), in_flight_(0), 
          rtt_min_(UINT64_MAX), rtt_smooth_(0), rtt_var_(0),
          dup_ack_count_(0), last_ack_time_(std::chrono::steady_clock::now()) {
        // Start with slow start
        state_ = CongestionState::SLOW_START;
    }
    
    bool can_send() {
        return in_flight_.load() < cwnd_.load();
    }
    
    void packet_sent() {
        in_flight_.fetch_add(1);
    }
    
    void packet_acked(uint64_t rtt_ns = 0) {
        if (in_flight_.load() > 0) {
            in_flight_.fetch_sub(1);
        }
        
        // Update RTT estimates if provided
        if (rtt_ns > 0) {
            update_rtt(rtt_ns);
        }
        
        // Update congestion window based on state
        auto current_cwnd = cwnd_.load();
        auto current_ssthresh = ssthresh_.load();
        
        switch (state_) {
            case CongestionState::SLOW_START:
                // Exponential growth
                cwnd_.store(std::min(current_cwnd * 2, MAX_WINDOW_SIZE));
                if (current_cwnd >= current_ssthresh) {
                    state_ = CongestionState::CONGESTION_AVOIDANCE;
                }
                break;
                
            case CongestionState::CONGESTION_AVOIDANCE:
                // Linear growth (additive increase)
                if (current_cwnd < MAX_WINDOW_SIZE) {
                    cwnd_.fetch_add(1);
                }
                break;
                
            case CongestionState::FAST_RECOVERY:
                // Stay in fast recovery until all lost packets are recovered
                break;
        }
        
        dup_ack_count_ = 0;
        last_ack_time_ = std::chrono::steady_clock::now();
    }
    
    void packet_lost() {
        // Multiplicative decrease on loss
        auto current_cwnd = cwnd_.load();
        ssthresh_.store(std::max(current_cwnd / 2, MIN_WINDOW_SIZE));
        
        if (state_ == CongestionState::SLOW_START) {
            cwnd_.store(MIN_WINDOW_SIZE);
            state_ = CongestionState::SLOW_START;
        } else {
            // Enter fast recovery
            cwnd_.store(ssthresh_.load() + 3);  // 3 for the duplicate ACKs
            state_ = CongestionState::FAST_RECOVERY;
        }
    }
    
    void duplicate_ack() {
        dup_ack_count_++;
        
        if (dup_ack_count_ == 3) {
            // Fast retransmit triggered
            packet_lost();
        } else if (dup_ack_count_ > 3 && state_ == CongestionState::FAST_RECOVERY) {
            // Inflate window for each additional duplicate ACK
            cwnd_.fetch_add(1);
        }
    }
    
    void timeout() {
        // Severe congestion - reset to slow start
        cwnd_.store(MIN_WINDOW_SIZE);
        ssthresh_.store(std::max(cwnd_.load() / 2, MIN_WINDOW_SIZE));
        state_ = CongestionState::SLOW_START;
        dup_ack_count_ = 0;
    }
    
    uint32_t get_window_size() const {
        return cwnd_.load();
    }
    
    uint64_t get_rtt_estimate() const {
        return rtt_smooth_.load();
    }
    
private:
    enum class CongestionState {
        SLOW_START,
        CONGESTION_AVOIDANCE,
        FAST_RECOVERY
    };
    
    void update_rtt(uint64_t measured_rtt) {
        // Update minimum RTT
        uint64_t min_rtt = rtt_min_.load();
        while (measured_rtt < min_rtt && !rtt_min_.compare_exchange_weak(min_rtt, measured_rtt));
        
        // Update smoothed RTT (exponential moving average)
        uint64_t smooth = rtt_smooth_.load();
        uint64_t var = rtt_var_.load();
        
        if (smooth == 0) {
            // First measurement
            rtt_smooth_.store(measured_rtt);
            rtt_var_.store(measured_rtt / 2);
        } else {
            // EWMA with alpha = 1/8, beta = 1/4 (RFC 6298)
            int64_t err = measured_rtt - smooth;
            var = (3 * var + std::abs(err)) / 4;
            smooth = (7 * smooth + measured_rtt) / 8;
            
            rtt_smooth_.store(smooth);
            rtt_var_.store(var);
        }
    }
    
    static constexpr uint32_t MIN_WINDOW_SIZE = 2;
    static constexpr uint32_t MAX_WINDOW_SIZE = 256;
    
    std::atomic<uint32_t> cwnd_;           // Congestion window
    std::atomic<uint32_t> ssthresh_;       // Slow start threshold
    std::atomic<uint32_t> in_flight_;      // Packets in flight
    std::atomic<uint64_t> rtt_min_;        // Minimum RTT observed
    std::atomic<uint64_t> rtt_smooth_;     // Smoothed RTT estimate
    std::atomic<uint64_t> rtt_var_;        // RTT variance
    std::atomic<uint32_t> dup_ack_count_;  // Duplicate ACK counter
    std::chrono::steady_clock::time_point last_ack_time_;
    CongestionState state_;
};

/**
 * GenieDataPlane implementation
 */
GenieDataPlane::GenieDataPlane(const DataPlaneConfig& config)
    : config_(config), mempool_(nullptr), port_id_(config.port_id), queue_id_(config.queue_id),
      gpu_device_id_(config.gpu_device_id), gpudev_available_(false), tx_ring_(nullptr) {
}

GenieDataPlane::~GenieDataPlane() {
    shutdown();
}

bool GenieDataPlane::initialize() {
    try {
        std::cout << "Initializing Genie data plane..." << std::endl;
        
        // Initialize DPDK
        if (!init_dpdk()) {
            std::cerr << "Failed to initialize DPDK" << std::endl;
            return false;
        }
        
        // Initialize port
        if (!init_port()) {
            std::cerr << "Failed to initialize port" << std::endl;
            return false;
        }
        
        // Initialize mempool
        if (!init_mempool()) {
            std::cerr << "Failed to initialize mempool" << std::endl;
            return false;
        }
        
        // Initialize GPU
        if (!init_gpu()) {
            std::cerr << "Failed to initialize GPU" << std::endl;
            return false;
        }
        
        // Initialize components
        gpu_memory_mgr_ = std::make_unique<GPUMemoryManager>(gpu_device_id_);
        packet_processor_ = std::make_unique<PacketProcessor>(config_.local_ip, config_.local_mac, config_.data_port);
        reliability_mgr_ = std::make_unique<ReliabilityManager>(config_.ack_timeout_ms, config_.max_retries);
        flow_controller_ = std::make_unique<FlowController>(config_.window_size);
        
        // Initialize KCP wrapper if reliability mode is KCP
        if (reliability_mode() == GenieDataPlane::ReliabilityMode::KCP) {
            // Output function bridges to DPDK send
            auto output_fn = [this](const char* buf, int len) -> int {
                return this->dpdk_send_raw(buf, len);
            };
            kcp_ = std::make_unique<KCPWrapper>();
            kcp_->initialize(0x1u, output_fn);
        }

        std::cout << "Data plane initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during initialization: " << e.what() << std::endl;
        return false;
    }
}
int GenieDataPlane::dpdk_send_raw(const char* buf, int len) {
    if (len <= 0 || !buf) return -1;
    // Allocate mbuf
    struct rte_mbuf* pkt = rte_pktmbuf_alloc(mempool_);
    if (!pkt) return -1;
    // Ensure buffer fits
    if (rte_pktmbuf_tailroom(pkt) < static_cast<uint16_t>(len)) {
        rte_pktmbuf_free(pkt);
        return -1;
    }
    // Copy data into mbuf data area
    char* data_ptr = rte_pktmbuf_append(pkt, static_cast<uint16_t>(len));
    if (!data_ptr) {
        rte_pktmbuf_free(pkt);
        return -1;
    }
    std::memcpy(data_ptr, buf, len);
    // Transmit immediately on queue
    uint16_t sent = rte_eth_tx_burst(port_id_, queue_id_, &pkt, 1);
    if (sent == 0) {
        rte_pktmbuf_free(pkt);
        return -1;
    }
    stats_.packets_sent.fetch_add(1);
    stats_.bytes_sent.fetch_add(static_cast<uint64_t>(len));
    return len;
}

bool GenieDataPlane::init_dpdk() {
    // Convert EAL args to C-style
    std::vector<char*> eal_argv;
    std::vector<std::string> eal_args_storage;
    
    for (const auto& arg : config_.eal_args) {
        eal_args_storage.push_back(arg);
        eal_argv.push_back(const_cast<char*>(eal_args_storage.back().c_str()));
    }
    
    int argc = static_cast<int>(eal_argv.size());
    char** argv = eal_argv.data();
    
    // Initialize EAL
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        std::cerr << "EAL initialization failed: " << rte_strerror(-ret) << std::endl;
        return false;
    }
    
    std::cout << "DPDK EAL initialized with " << rte_lcore_count() << " cores" << std::endl;
    return true;
}

bool GenieDataPlane::init_port() {
    // Check if port is available
    if (!rte_eth_dev_is_valid_port(port_id_)) {
        std::cerr << "Port " << port_id_ << " is not available" << std::endl;
        return false;
    }
    
    // Get port info
    struct rte_eth_dev_info dev_info;
    int ret = rte_eth_dev_info_get(port_id_, &dev_info);
    if (ret != 0) {
        std::cerr << "Failed to get port info: " << rte_strerror(-ret) << std::endl;
        return false;
    }
    
    // Configure port
    struct rte_eth_conf port_conf = {};
    port_conf.rxmode.mtu = DEFAULT_MTU;
    if (config_.enable_rss) {
        port_conf.rxmode.mq_mode = RTE_ETH_MQ_RX_RSS;
        port_conf.rx_adv_conf.rss_conf.rss_key = nullptr;
        port_conf.rx_adv_conf.rss_conf.rss_hf = RTE_ETH_RSS_IP | RTE_ETH_RSS_UDP;
    }
    // Optional NIC offloads
    if (config_.rx_offload_checksum) {
        port_conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_CHECKSUM;
    }
    if (config_.rx_offload_lro) {
        port_conf.rxmode.offloads |= RTE_ETH_RX_OFFLOAD_TCP_LRO;
        port_conf.rxmode.mtu = std::max<uint16_t>(port_conf.rxmode.mtu, 9000);
    }
    if (config_.tx_offload_ipv4_cksum) {
        port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_IPV4_CKSUM;
    }
    if (config_.tx_offload_udp_cksum) {
        port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_UDP_CKSUM;
    }
    if (config_.tx_offload_tso) {
        port_conf.txmode.offloads |= RTE_ETH_TX_OFFLOAD_TCP_TSO;
    }
    
    ret = rte_eth_dev_configure(port_id_, config_.rx_queues, config_.tx_queues, &port_conf);
    if (ret < 0) {
        std::cerr << "Failed to configure port: " << rte_strerror(-ret) << std::endl;
        return false;
    }
    
    std::cout << "Port " << port_id_ << " configured successfully" << std::endl;
    return true;
}

bool GenieDataPlane::init_mempool() {
    // Create mempool
    mempool_ = rte_pktmbuf_pool_create(
        "genie_pool",
        config_.mempool_size,
        256,  // Cache size
        0,    // Private data size
        RTE_MBUF_DEFAULT_BUF_SIZE,
        rte_socket_id()
    );
    
    if (!mempool_) {
        std::cerr << "Failed to create mempool" << std::endl;
        return false;
    }
    
    std::cout << "Mempool created with " << config_.mempool_size << " mbufs" << std::endl;
    return true;
}

bool GenieDataPlane::init_gpu() {
    // Check GPU availability
    int gpu_count = rte_gpu_count_avail();
    if (gpu_count > 0 && gpu_device_id_ < gpu_count) {
        gpudev_available_ = true;
        std::cout << "GPUDev available with " << gpu_count << " GPUs" << std::endl;
    } else {
        std::cout << "GPUDev not available, using fallback mode" << std::endl;
    }
    
    return true;
}

bool GenieDataPlane::start() {
    if (running_.load()) {
        return true;
    }
    
    try {
        // Start port
        int ret = rte_eth_dev_start(port_id_);
        if (ret < 0) {
            std::cerr << "Failed to start port: " << rte_strerror(-ret) << std::endl;
            return false;
        }
        
        // Setup RX queues
        for (uint16_t q = 0; q < config_.rx_queues; q++) {
            ret = rte_eth_rx_queue_setup(port_id_, q, config_.rx_ring_size,
                                        rte_eth_dev_socket_id(port_id_), nullptr, mempool_);
            if (ret < 0) {
                std::cerr << "Failed to setup RX queue " << q << ": " << rte_strerror(-ret) << std::endl;
                return false;
            }
        }
        
        // Setup TX queues
        for (uint16_t q = 0; q < config_.tx_queues; q++) {
            ret = rte_eth_tx_queue_setup(port_id_, q, config_.tx_ring_size,
                                        rte_eth_dev_socket_id(port_id_), nullptr);
            if (ret < 0) {
                std::cerr << "Failed to setup TX queue " << q << ": " << rte_strerror(-ret) << std::endl;
                return false;
            }
        }
        
        // Create TX ring for inter-thread communication
        tx_ring_ = rte_ring_create("genie_tx_ring", 1024, rte_socket_id(), RING_F_SP_ENQ | RING_F_SC_DEQ);
        if (!tx_ring_) {
            std::cerr << "Failed to create TX ring" << std::endl;
            return false;
        }
        
        running_.store(true);
        
        // Check if DPDK threading should be used
        if (config_.use_thread_pool && rte_lcore_count() > 1) {
            // Use DPDK native lcore management if we have worker lcores
            if (!initialize_thread_pool(true)) {
                // Fall back to legacy threads if DPDK threading fails
                std::cerr << "DPDK threading initialization failed, using legacy threads" << std::endl;
                config_.use_thread_pool = false;
            } else {
                start_thread_pool();
            }
        }
        
        if (!config_.use_thread_pool) {
            // Start legacy worker threads as fallback
            rx_thread_ = std::thread(&GenieDataPlane::rx_loop, this);
            tx_thread_ = std::thread(&GenieDataPlane::tx_loop, this);
            timeout_thread_ = std::thread([this]() {
                while (running_.load()) {
                    check_timeouts();
                    std::this_thread::sleep_for(std::chrono::milliseconds(config_.ack_timeout_ms / 2));
                }
            });
            // Start KCP update loop when KCP mode is active
            if (reliability_mode() == GenieDataPlane::ReliabilityMode::KCP && kcp_) {
                std::thread([this]() {
                    const int interval_ms = 10;
                    std::vector<char> recv_buf(64 * 1024);
                    while (running_.load()) {
                        kcp_->update(0);
                        // Poll KCP for received payloads and hand to reassembly path
                        int n = 0;
                        do {
                            n = kcp_->recv(recv_buf.data(), static_cast<int>(recv_buf.size()));
                            if (n > 0) {
                                // Expect KCP messages to carry GeniePacketHeader + payload
                                if (static_cast<size_t>(n) > sizeof(GeniePacketHeader)) {
                                    GeniePacketHeader header{};
                                    std::memcpy(&header, recv_buf.data(), sizeof(GeniePacketHeader));
                                    const uint8_t* payload_ptr = reinterpret_cast<const uint8_t*>(recv_buf.data() + sizeof(GeniePacketHeader));
                                    const uint32_t payload_len = static_cast<uint32_t>(n - static_cast<int>(sizeof(GeniePacketHeader)));

                                    // Construct a minimal GeniePacket wrapper to reuse handle_fragment()
                                    GeniePacket pkt{};
                                    pkt.app = header;
                                    handle_fragment(pkt, payload_ptr, payload_len);
                                    stats_.bytes_received.fetch_add(static_cast<uint64_t>(payload_len));
                                } else {
                                    // Not enough data to contain a header; drop
                                }
                            }
                        } while (n > 0);
                        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
                    }
                }).detach();
            }
        }
        
        std::cout << "Data plane started successfully"
                  << (config_.use_thread_pool ? " with DPDK native threading" : " with legacy threads")
                  << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during start: " << e.what() << std::endl;
        return false;
    }
}

void GenieDataPlane::stop() {
    if (!running_.load()) {
        return;
    }
    
    running_.store(false);
    
    // Stop DPDK threads if active
    if (thread_manager_) {
        stop_thread_pool();
    } else {
        // Wait for legacy threads to finish
        if (rx_thread_.joinable()) {
            rx_thread_.join();
        }
        if (tx_thread_.joinable()) {
            tx_thread_.join();
        }
        if (timeout_thread_.joinable()) {
            timeout_thread_.join();
        }
    }
    
    // Stop port
    rte_eth_dev_stop(port_id_);
    
    std::cout << "Data plane stopped" << std::endl;
}

void GenieDataPlane::shutdown() {
    stop();
    // Close KCP if active
    if (kcp_) {
        kcp_->close();
        kcp_.reset();
    }
    
    // Cleanup resources
    if (tx_ring_) {
        rte_ring_free(tx_ring_);
        tx_ring_ = nullptr;
    }
    
    if (mempool_) {
        rte_mempool_free(mempool_);
        mempool_ = nullptr;
    }
    
    // Cleanup EAL
    rte_eal_cleanup();
    
    std::cout << "Data plane shutdown complete" << std::endl;
}

bool GenieDataPlane::send_tensor(const std::string& transfer_id,
                                const std::string& tensor_id,
                                void* gpu_ptr,
                                size_t size,
                                const std::string& target_node) {
    
    std::lock_guard<std::mutex> lock(transfers_mutex_);
    
    // Create transfer context
    auto context = std::make_unique<TransferContext>();
    context->transfer_id = transfer_id;
    context->tensor_id = tensor_id;
    context->target_node = target_node;
    context->total_size = size;
    
    // Register GPU memory
    if (!gpu_memory_mgr_->register_memory(gpu_ptr, size, context->gpu_handle)) {
        std::cerr << "Failed to register GPU memory for transfer " << transfer_id << std::endl;
        return false;
    }
    
    // Find target node info
    std::lock_guard<std::mutex> nodes_lock(nodes_mutex_);
    auto node_it = target_nodes_.find(target_node);
    if (node_it == target_nodes_.end()) {
        std::cerr << "Target node " << target_node << " not configured" << std::endl;
        return false;
    }
    
    const std::string& target_ip = node_it->second.first;
    const std::string& target_mac = node_it->second.second;
    
    // Fragment and send tensor data
    bool success = fragment_and_send(transfer_id, tensor_id, 
                                   static_cast<const uint8_t*>(gpu_ptr), 
                                   size, target_node);
    
    if (success) {
        active_transfers_[transfer_id] = std::move(context);
        stats_.active_transfers.fetch_add(1);
        std::cout << "Started sending tensor " << tensor_id << " (" << size << " bytes) to " << target_node << std::endl;
    }
    
    return success;
}

bool GenieDataPlane::receive_tensor(const std::string& transfer_id,
                                   const std::string& tensor_id,
                                   void* gpu_ptr,
                                   size_t size,
                                   const std::string& source_node) {
    
    std::lock_guard<std::mutex> lock(transfers_mutex_);
    
    // Create transfer context for receiving
    auto context = std::make_unique<TransferContext>();
    context->transfer_id = transfer_id;
    context->tensor_id = tensor_id;
    context->source_node = source_node;
    context->total_size = size;
    
    // Register GPU memory if provided
    if (gpu_ptr && !gpu_memory_mgr_->register_memory(gpu_ptr, size, context->gpu_handle)) {
        std::cerr << "Failed to register GPU memory for receive transfer " << transfer_id << std::endl;
        return false;
    }
    
    active_transfers_[transfer_id] = std::move(context);
    stats_.active_transfers.fetch_add(1);
    
    std::cout << "Prepared to receive tensor " << tensor_id << " (" << size << " bytes) from " << source_node << std::endl;
    return true;
}

bool GenieDataPlane::set_transfer_metadata(const std::string& transfer_id,
                                          uint8_t dtype_code,
                                          uint8_t phase,
                                          const uint16_t* shape_dims,
                                          uint8_t shape_rank) {
    std::lock_guard<std::mutex> lock(transfers_mutex_);
    auto it = active_transfers_.find(transfer_id);
    if (it == active_transfers_.end()) return false;
    auto& ctx = it->second;
    ctx->meta_dtype_code = dtype_code;
    ctx->meta_phase = phase;
    ctx->meta_shape_rank = std::min<uint8_t>(shape_rank, 4);
    for (size_t i = 0; i < ctx->meta_shape_rank; ++i) ctx->meta_shape_dims[i] = shape_dims[i];
    return true;
}

bool GenieDataPlane::fragment_and_send(const std::string& transfer_id,
                                      const std::string& tensor_id,
                                      const uint8_t* data,
                                      size_t size,
                                      const std::string& target_node) {
    
    // Find target node info
    auto node_it = target_nodes_.find(target_node);
    if (node_it == target_nodes_.end()) {
        return false;
    }
    
    const std::string& target_ip = node_it->second.first;
    const std::string& target_mac = node_it->second.second;
    
    // Calculate fragments
    uint32_t max_payload = MAX_PAYLOAD_SIZE;
    uint32_t total_fragments = (size + max_payload - 1) / max_payload;
    
    std::cout << "Fragmenting tensor into " << total_fragments << " packets" << std::endl;
    
    // Send fragments
    for (uint32_t frag_id = 0; frag_id < total_fragments; frag_id++) {
        uint32_t offset = frag_id * max_payload;
        uint32_t frag_size = std::min(max_payload, static_cast<uint32_t>(size - offset));
        
        // Wait for flow control
        while (!flow_controller_->can_send() && running_.load()) {
            std::this_thread::sleep_for(std::chrono::microseconds(config_.poll_interval_us));
        }
        
        if (!running_.load()) {
            return false;
        }
        
        // Get sequence number
        uint32_t seq_num = reliability_mgr_->get_next_sequence();
        
        // Build packet
        PacketFlags flags = PacketFlags::FRAGMENTED;
        if (frag_id == total_fragments - 1) {
            flags = static_cast<PacketFlags>(static_cast<uint8_t>(flags) | static_cast<uint8_t>(PacketFlags::LAST_FRAGMENT));
        }
        
        struct rte_mbuf* pkt = packet_processor_->build_data_packet(
            mempool_, target_ip, target_mac, tensor_id, seq_num,
            data + offset, frag_size, frag_id, total_fragments, offset, flags
        );
        
        if (!pkt) {
            std::cerr << "Failed to build packet for fragment " << frag_id << std::endl;
            return false;
        }
        
        // Add to reliability tracking
        reliability_mgr_->add_pending_packet(seq_num, rte_pktmbuf_clone(pkt, mempool_));
        
        // Send packet
        if (rte_ring_enqueue(tx_ring_, pkt) != 0) {
            std::cerr << "Failed to enqueue packet for transmission" << std::endl;
            rte_pktmbuf_free(pkt);
            return false;
        }
        
        flow_controller_->packet_sent();
        stats_.packets_sent.fetch_add(1);
        stats_.bytes_sent.fetch_add(frag_size);
    }
    
    return true;
}

void GenieDataPlane::rx_loop() {
    std::cout << "RX loop started" << std::endl;
    
    while (running_.load()) {
        struct rte_mbuf* pkts[config_.burst_size];
        uint16_t nb_rx = rte_eth_rx_burst(port_id_, queue_id_, pkts, config_.burst_size);
        
        if (nb_rx > 0) {
            process_rx_packets(pkts, nb_rx);
            stats_.packets_received.fetch_add(nb_rx);
        }
        
        // Small sleep to prevent busy waiting
        if (nb_rx == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(config_.poll_interval_us));
        }
    }
    
    std::cout << "RX loop stopped" << std::endl;
}

void GenieDataPlane::tx_loop() {
    std::cout << "TX loop started" << std::endl;
    
    while (running_.load()) {
        struct rte_mbuf* pkts[config_.burst_size];
        unsigned int nb_deq = rte_ring_dequeue_burst(tx_ring_, reinterpret_cast<void**>(pkts), config_.burst_size, nullptr);
        
        if (nb_deq > 0) {
            uint16_t nb_tx = rte_eth_tx_burst(port_id_, queue_id_, pkts, nb_deq);
            
            // Free any unsent packets
            for (uint16_t i = nb_tx; i < nb_deq; i++) {
                rte_pktmbuf_free(pkts[i]);
                stats_.packets_dropped.fetch_add(1);
            }
        }
        
        // Small sleep to prevent busy waiting
        if (nb_deq == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(config_.poll_interval_us));
        }
    }
    
    std::cout << "TX loop stopped" << std::endl;
}

void GenieDataPlane::process_rx_packets(struct rte_mbuf** pkts, uint16_t nb_pkts) {
    for (uint16_t i = 0; i < nb_pkts; i++) {
        GeniePacket parsed_pkt;
        uint8_t* payload;
        uint32_t payload_size;
        
        if (packet_processor_->parse_packet(pkts[i], parsed_pkt, payload, payload_size)) {
            // Handle different packet types (switch behavior if KCP enabled)
            PacketType pkt_type = static_cast<PacketType>(parsed_pkt.app.type);
            
            switch (pkt_type) {
                case PacketType::DATA:
                    if (reliability_mode() == GenieDataPlane::ReliabilityMode::KCP && kcp_) {
                        // Feed payload to KCP
                        kcp_->input(reinterpret_cast<const char*>(payload), static_cast<int>(payload_size));
                    } else {
                        handle_fragment(parsed_pkt, payload, payload_size);
                    }
                    break;
                case PacketType::ACK:
                    handle_ack(ntohl(parsed_pkt.app.seq_num));
                    break;
                case PacketType::NACK:
                    // Handle NACK (trigger retransmission)
                    break;
                default:
                    // Unknown packet type
                    break;
            }
            
            stats_.bytes_received.fetch_add(payload_size);
        }
        
        rte_pktmbuf_free(pkts[i]);
    }
}

bool GenieDataPlane::handle_fragment(const GeniePacket& pkt, const uint8_t* payload, uint32_t payload_size) {
    // Extract fragment information
    std::string tensor_id(reinterpret_cast<const char*>(pkt.app.tensor_id), 16);
    // Trim null characters from tensor_id
    tensor_id = tensor_id.substr(0, tensor_id.find('\0'));
    
    uint16_t frag_id = ntohs(pkt.app.frag_id);
    uint16_t frag_count = ntohs(pkt.app.frag_count);
    uint32_t offset = ntohl(pkt.app.offset);
    uint32_t total_size = ntohl(pkt.app.total_size);
    uint32_t seq_num = ntohl(pkt.app.seq_num);
    
    // Validate fragment parameters
    if (frag_id >= frag_count || offset + payload_size > total_size) {
        std::cerr << "Invalid fragment parameters for tensor " << tensor_id << std::endl;
        return false;
    }
    
    // Find or create transfer context
    std::lock_guard<std::mutex> lock(transfers_mutex_);
    
    // Look for existing transfer context by tensor_id
    TransferContext* context = nullptr;
    for (auto& [tid, ctx] : active_transfers_) {
        if (ctx->tensor_id == tensor_id) {
            context = ctx.get();
            break;
        }
    }
    
    // Create new context if this is the first fragment
    if (!context) {
        auto transfer_id = "recv_" + tensor_id;
        auto new_context = std::make_unique<TransferContext>();
        new_context->transfer_id = transfer_id;
        new_context->tensor_id = tensor_id;
        new_context->total_size = total_size;
        new_context->total_fragments = frag_count;
        new_context->received_fragments.resize(frag_count, false);
        
        // Allocate buffer for reassembly
        if (gpudev_available_ && gpu_memory_mgr_) {
            // Try to allocate GPU memory directly
            void* gpu_buffer = nullptr;
#ifdef GENIE_CUDA_SUPPORT
            cudaError_t err = cudaMalloc(&gpu_buffer, total_size);
            if (err == cudaSuccess) {
                new_context->gpu_handle.gpu_ptr = gpu_buffer;
                new_context->gpu_handle.size = total_size;
                new_context->gpu_handle.gpu_id = gpu_device_id_;
                
                // Register for DMA if possible
                gpu_memory_mgr_->register_memory(gpu_buffer, total_size, new_context->gpu_handle);
            } else {
                std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(err) << std::endl;
            }
#endif
        }
        
        // Fallback to CPU buffer if GPU allocation failed
        if (!new_context->gpu_handle.gpu_ptr) {
            new_context->cpu_buffer = aligned_alloc(64, total_size);
            if (!new_context->cpu_buffer) {
                std::cerr << "Failed to allocate memory for tensor " << tensor_id << std::endl;
                return false;
            }
        }
        
        context = new_context.get();
        active_transfers_[transfer_id] = std::move(new_context);
        stats_.active_transfers.fetch_add(1);
        
        std::cout << "Created new transfer context for tensor " << tensor_id 
                  << " (size=" << total_size << ", fragments=" << frag_count << ")" << std::endl;
    }
    
    // Check if fragment was already received (duplicate)
    if (context->received_fragments[frag_id]) {
        std::cout << "Duplicate fragment " << frag_id << " for tensor " << tensor_id << std::endl;
        // Send ACK anyway in case previous ACK was lost
        send_ack_for_fragment(pkt, seq_num);
        return true;
    }
    
    // Copy fragment data to appropriate buffer
    uint8_t* dest_buffer = nullptr;
    if (context->gpu_handle.gpu_ptr) {
        dest_buffer = static_cast<uint8_t*>(context->gpu_handle.gpu_ptr);
    } else if (context->cpu_buffer) {
        dest_buffer = static_cast<uint8_t*>(context->cpu_buffer);
    }
    
    if (dest_buffer) {
#ifdef GENIE_CUDA_SUPPORT
        if (context->gpu_handle.gpu_ptr) {
            // Copy to GPU memory
            cudaError_t err = cudaMemcpy(dest_buffer + offset, payload, payload_size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "Failed to copy fragment to GPU: " << cudaGetErrorString(err) << std::endl;
                return false;
            }
        } else
#endif
        {
            // Copy to CPU memory
            std::memcpy(dest_buffer + offset, payload, payload_size);
        }
    }
    
    // Mark fragment as received
    context->received_fragments[frag_id] = true;
    context->bytes_transferred.fetch_add(payload_size);
    context->packets_received.fetch_add(1);
    
    // Send ACK for this fragment
    send_ack_for_fragment(pkt, seq_num);
    
    // Check if all fragments are received
    bool all_received = true;
    uint32_t received_count = 0;
    for (bool received : context->received_fragments) {
        if (received) {
            received_count++;
        } else {
            all_received = false;
        }
    }
    
    std::cout << "Received fragment " << frag_id << "/" << frag_count 
              << " for tensor " << tensor_id << " (" << received_count 
              << "/" << frag_count << " total)" << std::endl;
    
    if (all_received) {
        // Transfer complete!
        context->is_complete.store(true);
        stats_.transfers_completed.fetch_add(1);
        stats_.active_transfers.fetch_sub(1);
        
        auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - context->start_time).count();
        
        // Update bandwidth statistics
        if (elapsed_ns > 0) {
            uint64_t bandwidth_bps = (context->total_size * 8ULL * 1000000000ULL) / elapsed_ns;
            uint64_t current_peak = stats_.peak_bandwidth_bps.load();
            while (bandwidth_bps > current_peak && 
                   !stats_.peak_bandwidth_bps.compare_exchange_weak(current_peak, bandwidth_bps));
            
            // Update average latency (simple moving average)
            uint64_t current_avg = stats_.avg_latency_ns.load();
            stats_.avg_latency_ns.store((current_avg + elapsed_ns) / 2);
        }
        
        std::cout << "Transfer complete for tensor " << tensor_id 
                  << " (" << context->total_size << " bytes in " 
                  << elapsed_ns / 1000000.0 << " ms)" << std::endl;
        
        // TODO: Notify control plane of completion
        // For now, the context remains in active_transfers until explicitly removed
    }
    
    return true;
}

// Helper function to send ACK for a fragment
void GenieDataPlane::send_ack_for_fragment(const GeniePacket& pkt, uint32_t seq_num) {
    // Extract source information from received packet
    uint32_t src_ip = ntohl(pkt.ip.src_ip);
    char ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &src_ip, ip_str, INET_ADDRSTRLEN);
    
    // Build ACK packet
    struct rte_mbuf* ack_pkt = packet_processor_->build_ack_packet(
        mempool_, 
        std::string(ip_str),
        "", // MAC will be looked up
        seq_num
    );
    
    if (ack_pkt) {
        // Send ACK immediately
        if (rte_ring_enqueue(tx_ring_, ack_pkt) != 0) {
            rte_pktmbuf_free(ack_pkt);
        }
    }
}

void GenieDataPlane::handle_ack(uint32_t seq_num) {
    if (reliability_mgr_->handle_ack(seq_num)) {
        flow_controller_->packet_acked();
    }
}

void GenieDataPlane::check_timeouts() {
    auto retransmit_list = reliability_mgr_->check_timeouts();
    
    for (auto* pkt : retransmit_list) {
        if (rte_ring_enqueue(tx_ring_, pkt) != 0) {
            rte_pktmbuf_free(pkt);
        } else {
            stats_.retransmissions.fetch_add(1);
        }
    }
}

bool GenieDataPlane::register_gpu_memory(void* gpu_ptr, size_t size, GPUMemoryHandle& handle) {
    return gpu_memory_mgr_->register_memory(gpu_ptr, size, handle);
}

void GenieDataPlane::unregister_gpu_memory(const GPUMemoryHandle& handle) {
    gpu_memory_mgr_->unregister_memory(handle);
}

void GenieDataPlane::set_target_node(const std::string& node_id, const std::string& ip, const std::string& mac) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    target_nodes_[node_id] = std::make_pair(ip, mac);
    std::cout << "Configured target node " << node_id << ": " << ip << " (" << mac << ")" << std::endl;
}

void GenieDataPlane::remove_target_node(const std::string& node_id) {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    target_nodes_.erase(node_id);
    std::cout << "Removed target node " << node_id << std::endl;
}

// Phase 3: Configuration methods for C API
void GenieDataPlane::configure_queues(uint16_t rx_queues, uint16_t tx_queues, bool enable_rss) {
    config_.rx_queues = rx_queues;
    config_.tx_queues = tx_queues;
    config_.enable_rss = enable_rss;
    
    std::cout << "Phase 3: Configured queues - RX: " << rx_queues 
              << ", TX: " << tx_queues 
              << ", RSS: " << (enable_rss ? "enabled" : "disabled") << std::endl;
}

void GenieDataPlane::enable_offloads(bool rx_checksum, bool rx_lro, bool tx_ipv4_cksum, bool tx_udp_cksum, bool tx_tso) {
    config_.rx_offload_checksum = rx_checksum;
    config_.rx_offload_lro = rx_lro;
    config_.tx_offload_ipv4_cksum = tx_ipv4_cksum;
    config_.tx_offload_udp_cksum = tx_udp_cksum;
    config_.tx_offload_tso = tx_tso;
    
    std::cout << "Phase 3: Configured offloads - RX checksum: " << (rx_checksum ? "on" : "off")
              << ", RX LRO: " << (rx_lro ? "on" : "off")
              << ", TX IPv4 cksum: " << (tx_ipv4_cksum ? "on" : "off")
              << ", TX UDP cksum: " << (tx_udp_cksum ? "on" : "off")
              << ", TX TSO: " << (tx_tso ? "on" : "off") << std::endl;
}

void GenieDataPlane::enable_cuda_graphs(bool enable) {
    config_.enable_cuda_graphs = enable;
    
    std::cout << "Phase 3: CUDA Graphs " << (enable ? "enabled" : "disabled") << std::endl;
}

void GenieDataPlane::get_statistics(DataPlaneStats& stats) const {
    // Copy atomic values to regular struct
    stats.packets_sent.store(stats_.packets_sent.load());
    stats.packets_received.store(stats_.packets_received.load());
    stats.bytes_sent.store(stats_.bytes_sent.load());
    stats.bytes_received.store(stats_.bytes_received.load());
    stats.packets_dropped.store(stats_.packets_dropped.load());
    stats.retransmissions.store(stats_.retransmissions.load());
    stats.transfers_completed.store(stats_.transfers_completed.load());
    stats.transfers_failed.store(stats_.transfers_failed.load());
    stats.active_transfers.store(stats_.active_transfers.load());
    stats.avg_latency_ns.store(stats_.avg_latency_ns.load());
    stats.peak_bandwidth_bps.store(stats_.peak_bandwidth_bps.load());
    stats.packet_loss_rate.store(stats_.packet_loss_rate.load());
}

std::vector<std::string> GenieDataPlane::get_active_transfers() const {
    std::lock_guard<std::mutex> lock(transfers_mutex_);
    std::vector<std::string> transfer_ids;
    for (const auto& pair : active_transfers_) {
        transfer_ids.push_back(pair.first);
    }
    return transfer_ids;
}

void GenieDataPlane::get_transfer_status(const std::string& transfer_id, TransferContext& context, bool& found) const {
    std::lock_guard<std::mutex> lock(transfers_mutex_);
    auto it = active_transfers_.find(transfer_id);
    if (it != active_transfers_.end()) {
        // Copy non-atomic fields manually
        context.transfer_id = it->second->transfer_id;
        context.tensor_id = it->second->tensor_id;
        context.source_node = it->second->source_node;
        context.target_node = it->second->target_node;
        context.gpu_handle = it->second->gpu_handle;
        context.cpu_buffer = it->second->cpu_buffer;
        context.total_size = it->second->total_size;
        context.total_fragments = it->second->total_fragments;
        context.received_fragments = it->second->received_fragments;
        context.fragment_data = it->second->fragment_data;
        context.start_time = it->second->start_time;
        context.error_message = it->second->error_message;
        
        // Copy atomic values
        context.bytes_transferred.store(it->second->bytes_transferred.load());
        context.packets_sent.store(it->second->packets_sent.load());
        context.packets_received.store(it->second->packets_received.load());
        context.is_complete.store(it->second->is_complete.load());
        context.has_error.store(it->second->has_error.load());
        
        found = true;
    } else {
        found = false;
    }
}

// Thread pool management methods using DPDK native threading
bool GenieDataPlane::initialize_thread_pool(bool use_thread_pool) {
    if (!use_thread_pool) {
        return true;  // Not using thread pool
    }
    
    if (thread_manager_) {
        return true;  // Already initialized
    }
    
    try {
        thread_manager_ = std::make_unique<DPDKThreadManager>();
        
        // Initialize DPDK thread manager
        if (!thread_manager_->initialize()) {
            std::cerr << "Failed to initialize DPDK thread manager" << std::endl;
            thread_manager_.reset();
            return false;
        }
        
        // Assign available lcores
        unsigned lcore_id;
        unsigned lcore_count = 0;
        
        RTE_LCORE_FOREACH_WORKER(lcore_id) {
            if (lcore_count == 0) {
                // First worker lcore for RX
                thread_manager_->assign_lcore(lcore_id, LcoreConfig::RX_LCORE, port_id_, queue_id_);
            } else if (lcore_count == 1) {
                // Second for TX
                thread_manager_->assign_lcore(lcore_id, LcoreConfig::TX_LCORE, port_id_, queue_id_);
            } else {
                // Rest for workers
                thread_manager_->assign_lcore(lcore_id, LcoreConfig::WORKER_LCORE);
            }
            lcore_count++;
            
            if (lcore_count >= config_.rx_threads + config_.tx_threads + config_.worker_threads) {
                break;
            }
        }
        
        if (lcore_count < 2) {
            std::cerr << "Not enough lcores for DPDK threading (need at least 2)" << std::endl;
            thread_manager_.reset();
            return false;
        }
        
        std::cout << "DPDK thread manager initialized with " << lcore_count << " lcores" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception initializing DPDK thread manager: " << e.what() << std::endl;
        thread_manager_.reset();
        return false;
    }
}

void GenieDataPlane::start_thread_pool() {
    if (!thread_manager_) {
        return;
    }
    
    if (!thread_manager_->launch_all()) {
        std::cerr << "Failed to launch DPDK threads" << std::endl;
    }
}

void GenieDataPlane::stop_thread_pool() {
    if (!thread_manager_) {
        return;
    }
    
    thread_manager_->stop_all();
}

void GenieDataPlane::print_thread_statistics() const {
    if (thread_manager_) {
        thread_manager_->print_stats();
    } else {
        std::cout << "DPDK threading not active, using legacy threads" << std::endl;
        // Could print legacy thread stats here
    }
}

} // namespace data_plane
} // namespace genie

// C interface implementation
extern "C" {

void* genie_data_plane_create(const char* config_json) {
    try {
        json config = json::parse(config_json);
        
        genie::data_plane::DataPlaneConfig dp_config;
        
        // Parse configuration
        if (config.contains("eal_args")) {
            dp_config.eal_args = config["eal_args"].get<std::vector<std::string>>();
        }
        if (config.contains("port_id")) {
            dp_config.port_id = config["port_id"].get<uint16_t>();
        }
        if (config.contains("local_ip")) {
            dp_config.local_ip = config["local_ip"].get<std::string>();
        }
        if (config.contains("local_mac")) {
            dp_config.local_mac = config["local_mac"].get<std::string>();
        }
        if (config.contains("data_port")) {
            dp_config.data_port = config["data_port"].get<uint16_t>();
        }
        
        return new genie::data_plane::GenieDataPlane(dp_config);
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to create data plane: " << e.what() << std::endl;
        return nullptr;
    }
}

int genie_data_plane_initialize(void* data_plane) {
    if (!data_plane) return -1;
    
    auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
    return dp->initialize() ? 0 : -1;
}

int genie_data_plane_start(void* data_plane) {
    if (!data_plane) return -1;
    
    auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
    return dp->start() ? 0 : -1;
}

void genie_data_plane_stop(void* data_plane) {
    if (data_plane) {
        auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
        dp->stop();
    }
}

void genie_data_plane_destroy(void* data_plane) {
    if (data_plane) {
        auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
        delete dp;
    }
}

int genie_send_tensor(void* data_plane,
                     const char* transfer_id,
                     const char* tensor_id,
                     void* gpu_ptr,
                     size_t size,
                     const char* target_node) {
    if (!data_plane) return -1;
    
    auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
    return dp->send_tensor(transfer_id, tensor_id, gpu_ptr, size, target_node) ? 0 : -1;
}

int genie_receive_tensor(void* data_plane,
                        const char* transfer_id,
                        const char* tensor_id,
                        void* gpu_ptr,
                        size_t size,
                        const char* source_node) {
    if (!data_plane) return -1;
    
    auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
    return dp->receive_tensor(transfer_id, tensor_id, gpu_ptr, size, source_node) ? 0 : -1;
}

int genie_register_gpu_memory(void* data_plane, void* gpu_ptr, size_t size, uint64_t* iova) {
    if (!data_plane || !iova) return -1;
    
    auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
    genie::data_plane::GPUMemoryHandle handle;
    
    if (dp->register_gpu_memory(gpu_ptr, size, handle)) {
        *iova = handle.iova;
        return 0;
    }
    
    return -1;
}

void genie_unregister_gpu_memory(void* data_plane, void* gpu_ptr) {
    if (data_plane) {
        auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
        genie::data_plane::GPUMemoryHandle handle;
        handle.gpu_ptr = gpu_ptr;
        dp->unregister_gpu_memory(handle);
    }
}

void genie_get_statistics(void* data_plane, char* stats_json, size_t buffer_size) {
    if (!data_plane || !stats_json) return;
    
    auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
    genie::data_plane::DataPlaneStats stats;
    dp->get_statistics(stats);
    
    json stats_obj = {
        {"packets_sent", stats.packets_sent.load()},
        {"packets_received", stats.packets_received.load()},
        {"bytes_sent", stats.bytes_sent.load()},
        {"bytes_received", stats.bytes_received.load()},
        {"packets_dropped", stats.packets_dropped.load()},
        {"retransmissions", stats.retransmissions.load()},
        {"transfers_completed", stats.transfers_completed.load()},
        {"transfers_failed", stats.transfers_failed.load()},
        {"active_transfers", stats.active_transfers.load()}
    };
    
    std::string stats_str = stats_obj.dump();
    std::strncpy(stats_json, stats_str.c_str(), buffer_size - 1);
    stats_json[buffer_size - 1] = '\0';
}

int genie_get_transfer_status(void* data_plane, const char* transfer_id, char* status_json, size_t buffer_size) {
    if (!data_plane || !transfer_id || !status_json) return -1;
    
    auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
    genie::data_plane::TransferContext context;
    bool found = false;
    
    dp->get_transfer_status(transfer_id, context, found);
    
    if (found) {
        json status_obj = {
            {"transfer_id", context.transfer_id},
            {"tensor_id", context.tensor_id},
            {"bytes_transferred", context.bytes_transferred.load()},
            {"packets_sent", context.packets_sent.load()},
            {"packets_received", context.packets_received.load()},
            {"is_complete", context.is_complete.load()},
            {"has_error", context.has_error.load()},
            {"error_message", context.error_message}
        };
        
        std::string status_str = status_obj.dump();
        std::strncpy(status_json, status_str.c_str(), buffer_size - 1);
        status_json[buffer_size - 1] = '\0';
        return 0;
    }
    
    return -1;
}

void genie_set_target_node(void* data_plane, const char* node_id, const char* ip, const char* mac) {
    if (data_plane && node_id && ip && mac) {
        auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
        dp->set_target_node(node_id, ip, mac);
    }
}

void genie_remove_target_node(void* data_plane, const char* node_id) {
    if (data_plane && node_id) {
        auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
        dp->remove_target_node(node_id);
    }
}

int genie_set_transfer_metadata(void* data_plane,
                               const char* transfer_id,
                               uint8_t dtype_code,
                               uint8_t phase,
                               uint8_t shape_rank,
                               const uint16_t* shape_dims,
                               size_t dims_len) {
    if (!data_plane || !transfer_id) return -1;
    auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
    size_t n = std::min<size_t>(dims_len, 4);
    return dp->set_transfer_metadata(transfer_id, dtype_code, phase, shape_dims, static_cast<uint8_t>(n)) ? 0 : -1;
}

int genie_set_reliability_mode(void* data_plane, int mode) {
    if (!data_plane) return -1;
    auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
    if (mode == 1) {
        dp->set_reliability_mode(genie::data_plane::GenieDataPlane::ReliabilityMode::KCP);
    } else {
        dp->set_reliability_mode(genie::data_plane::GenieDataPlane::ReliabilityMode::CUSTOM);
    }
    return 0;
}

int genie_get_reliability_mode(void* data_plane) {
    if (!data_plane) return -1;
    auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
    return dp->reliability_mode() == genie::data_plane::GenieDataPlane::ReliabilityMode::KCP ? 1 : 0;
}

// Phase 3: Multi-queue & RSS configuration
int genie_configure_queues(void* data_plane, uint16_t rx_queues, uint16_t tx_queues, int enable_rss) {
    if (!data_plane) return -1;
    
    auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
    dp->configure_queues(rx_queues, tx_queues, enable_rss != 0);
    
    // Note: In a full implementation, this would reconfigure the DPDK port
    // For now, we just update the configuration for future use
    return 0;
}

// Phase 3: NIC offloads configuration  
int genie_enable_offloads(void* data_plane, int rx_checksum, int rx_lro, int tx_ipv4_cksum, int tx_udp_cksum, int tx_tso) {
    if (!data_plane) return -1;
    
    auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
    dp->enable_offloads(rx_checksum != 0, rx_lro != 0, tx_ipv4_cksum != 0, tx_udp_cksum != 0, tx_tso != 0);
    
    // Note: In a full implementation, this would reconfigure the DPDK port offloads
    // For now, we just update the configuration for future use
    return 0;
}

// Phase 3: CUDA Graphs integration
int genie_enable_cuda_graphs(void* data_plane, int enable) {
    if (!data_plane) return -1;
    
    auto* dp = static_cast<genie::data_plane::GenieDataPlane*>(data_plane);
    dp->enable_cuda_graphs(enable != 0);
    
    // Note: In a full implementation, this would initialize CUDA Graph capture
    // For now, we just update the configuration for future use
    return 0;
}

} // extern "C"
