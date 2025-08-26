/**
 * Zero-Copy RX Path Implementation
 * 
 * This implements the true zero-copy receive path where NIC DMAs
 * directly into pre-allocated GPU buffers.
 */

#include "genie_data_plane.hpp"
#include <rte_malloc.h>
#include <rte_memzone.h>
#include <set>
#include <iostream>
#include <arpa/inet.h>

namespace genie {
namespace data_plane {

/**
 * Zero-Copy RX Manager
 * 
 * Pre-allocates GPU buffers and exposes them to NIC as RX descriptors
 * so NIC DMAs payloads directly into GPU memory.
 */
class ZeroCopyRxManager {
public:
    ZeroCopyRxManager(int gpu_id, size_t buffer_size = 1024 * 1024 * 1024)  // 1GB default
        : gpu_id_(gpu_id), buffer_size_(buffer_size), gpu_buffer_(nullptr) {
    }
    
    ~ZeroCopyRxManager() {
        cleanup();
    }
    
    bool initialize(struct rte_mempool* header_pool) {
        // Step 1: Pre-allocate large GPU buffer pool
        if (!allocate_gpu_buffer()) {
            return false;
        }
        
        // Step 2: Register GPU memory for DMA
        if (!register_gpu_memory()) {
            cleanup();
            return false;
        }
        
        // Step 3: Create mempool with GPU-backed mbufs
        if (!create_gpu_mempool(header_pool)) {
            cleanup();
            return false;
        }
        
        // Step 4: Setup extbuf info for zero-copy
        if (!setup_extbuf_info()) {
            cleanup();
            return false;
        }
        
        std::cout << "Zero-copy RX initialized with " 
                  << buffer_size_ / (1024*1024) << " MB GPU buffer" << std::endl;
        
        return true;
    }
    
    struct rte_mempool* get_rx_mempool() {
        return gpu_mempool_;
    }
    
    void* get_gpu_buffer_at_offset(size_t offset) {
        if (offset >= buffer_size_) {
            return nullptr;
        }
        return static_cast<uint8_t*>(gpu_buffer_) + offset;
    }
    
private:
    bool allocate_gpu_buffer() {
#ifdef GENIE_CUDA_SUPPORT
        cudaError_t err = cudaMalloc(&gpu_buffer_, buffer_size_);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate GPU buffer: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // Initialize to zero for safety
        cudaMemset(gpu_buffer_, 0, buffer_size_);
#else
        // Use GPUDev allocation
        gpu_buffer_ = rte_gpu_mem_alloc(gpu_id_, buffer_size_, 0);
        if (!gpu_buffer_) {
            std::cerr << "Failed to allocate GPU buffer via GPUDev" << std::endl;
            return false;
        }
#endif
        return true;
    }
    
    bool register_gpu_memory() {
        // Register with GPUDev for DMA
        int ret = rte_gpu_mem_register(gpu_id_, buffer_size_, gpu_buffer_);
        if (ret < 0) {
            std::cerr << "Failed to register GPU memory: " << rte_strerror(-ret) << std::endl;
            return false;
        }
        
        // Get IOVA for DMA operations
        gpu_iova_ = rte_mem_virt2iova(gpu_buffer_);
        if (gpu_iova_ == RTE_BAD_IOVA) {
            // Try external memory registration
            ret = register_as_external_memory();
            if (ret < 0) {
                return false;
            }
        }
        
        std::cout << "GPU memory registered: IOVA=0x" << std::hex << gpu_iova_ << std::dec << std::endl;
        return true;
    }
    
    int register_as_external_memory() {
        // Register GPU memory as external memory for DPDK
        size_t page_size = 2 * 1024 * 1024;  // 2MB pages
        
        // Register external memory (simplified - using VA as IOVA)
        uint64_t iova = reinterpret_cast<uint64_t>(gpu_buffer_);
        int ret = rte_extmem_register(gpu_buffer_, buffer_size_, nullptr, 0, page_size);
        if (ret < 0) {
            std::cerr << "Failed to register external memory: " << rte_strerror(-ret) << std::endl;
            return ret;
        }
        
        // Map to all devices
        ret = rte_extmem_attach(gpu_buffer_, buffer_size_);
        if (ret < 0) {
            std::cerr << "Failed to attach external memory: " << rte_strerror(-ret) << std::endl;
            rte_extmem_unregister(gpu_buffer_, buffer_size_);
            return ret;
        }
        
        gpu_iova_ = iova;
        return 0;
    }
    
    bool create_gpu_mempool(struct rte_mempool* header_pool) {
        // Create custom mempool for GPU-backed mbufs
        // We only need space for mbuf headers, data points to GPU memory
        
        unsigned n_mbufs = buffer_size_ / MBUF_DATA_SIZE;
        
        gpu_mempool_ = rte_pktmbuf_pool_create(
            "gpu_rx_pool",
            n_mbufs,
            256,  // Cache size
            sizeof(struct rte_mbuf_ext_shared_info),  // Private data for extbuf
            RTE_MBUF_DEFAULT_DATAROOM,  // Header room only
            rte_socket_id()
        );
        
        if (!gpu_mempool_) {
            std::cerr << "Failed to create GPU mempool" << std::endl;
            return false;
        }
        
        // Initialize each mbuf to point to GPU memory slice
        if (!attach_gpu_memory_to_mbufs()) {
            rte_mempool_free(gpu_mempool_);
            gpu_mempool_ = nullptr;
            return false;
        }
        
        return true;
    }
    
    bool attach_gpu_memory_to_mbufs() {
        // Iterate through mempool and attach GPU memory slices
        struct rte_mbuf* mbufs[256];
        unsigned n = rte_mempool_get_bulk(gpu_mempool_, (void**)mbufs, 256);
        
        if (n == 0) {
            return true;  // No mbufs to process
        }
        
        for (unsigned i = 0; i < n; i++) {
            size_t offset = i * MBUF_DATA_SIZE;
            if (offset >= buffer_size_) {
                break;
            }
            
            // Get shared info from private data
            struct rte_mbuf_ext_shared_info* shinfo = 
                (struct rte_mbuf_ext_shared_info*)RTE_PTR_ADD(mbufs[i], sizeof(struct rte_mbuf));
            
            // Initialize shared info
            shinfo->free_cb = gpu_mbuf_free_cb;
            shinfo->fcb_opaque = this;
            rte_mbuf_ext_refcnt_set(shinfo, 1);
            
            // Attach GPU memory slice as external buffer
            rte_pktmbuf_attach_extbuf(
                mbufs[i],
                static_cast<uint8_t*>(gpu_buffer_) + offset,
                gpu_iova_ + offset,
                MBUF_DATA_SIZE,
                shinfo
            );
            
            // Reset mbuf data
            mbufs[i]->data_len = 0;
            mbufs[i]->pkt_len = 0;
        }
        
        // Put mbufs back
        rte_mempool_put_bulk(gpu_mempool_, (void**)mbufs, n);
        
        return true;
    }
    
    bool setup_extbuf_info() {
        // Setup shared info for all GPU memory slices
        size_t n_slices = buffer_size_ / MBUF_DATA_SIZE;
        
        extbuf_infos_.resize(n_slices);
        
        for (size_t i = 0; i < n_slices; i++) {
            extbuf_infos_[i].free_cb = gpu_mbuf_free_cb;
            extbuf_infos_[i].fcb_opaque = this;
            rte_mbuf_ext_refcnt_set(&extbuf_infos_[i], 1);
        }
        
        return true;
    }
    
    static void gpu_mbuf_free_cb(void* addr, void* opaque) {
        // Called when mbuf with GPU memory is freed
        // GPU memory is persistent, so we don't actually free it
        // Just track that this slice is available again
        ZeroCopyRxManager* mgr = static_cast<ZeroCopyRxManager*>(opaque);
        mgr->mark_slice_available(addr);
    }
    
    void mark_slice_available(void* addr) {
        // Track available slices for reuse
        size_t offset = static_cast<uint8_t*>(addr) - static_cast<uint8_t*>(gpu_buffer_);
        size_t slice_idx = offset / MBUF_DATA_SIZE;
        
        if (slice_idx < available_slices_.size()) {
            available_slices_[slice_idx] = true;
        }
    }
    
    void cleanup() {
        if (gpu_mempool_) {
            rte_mempool_free(gpu_mempool_);
            gpu_mempool_ = nullptr;
        }
        
        if (gpu_buffer_) {
            // Unregister from DPDK
            if (gpu_iova_ != RTE_BAD_IOVA) {
                rte_extmem_detach(gpu_buffer_, buffer_size_);
                rte_extmem_unregister(gpu_buffer_, buffer_size_);
            }
            
            // Unregister from GPUDev
            rte_gpu_mem_unregister(gpu_id_, gpu_buffer_);
            
            // Free GPU memory
#ifdef GENIE_CUDA_SUPPORT
            cudaFree(gpu_buffer_);
#else
            rte_gpu_mem_free(gpu_id_, gpu_buffer_);
#endif
            gpu_buffer_ = nullptr;
        }
    }
    
private:
    static constexpr size_t MBUF_DATA_SIZE = 9000;  // Jumbo frame size
    
    int gpu_id_;
    size_t buffer_size_;
    void* gpu_buffer_;
    rte_iova_t gpu_iova_;
    struct rte_mempool* gpu_mempool_;
    std::vector<struct rte_mbuf_ext_shared_info> extbuf_infos_;
    std::vector<bool> available_slices_;
};

/**
 * Address Resolution Cache
 * 
 * Maps IP addresses to MAC addresses for L2 header construction
 */
class AddressResolutionCache {
public:
    struct AddressEntry {
        uint32_t ip_addr;
        struct rte_ether_addr mac_addr;
        uint64_t last_seen_ns;
        bool is_valid;
    };
    
    AddressResolutionCache() {
        // Initialize with some static entries for testing
        add_static_entry("192.168.1.100", "aa:bb:cc:dd:ee:01");
        add_static_entry("192.168.1.101", "aa:bb:cc:dd:ee:02");
    }
    
    bool lookup(uint32_t ip_addr, struct rte_ether_addr& mac_addr) {
        auto it = cache_.find(ip_addr);
        if (it != cache_.end() && it->second.is_valid) {
            mac_addr = it->second.mac_addr;
            it->second.last_seen_ns = rte_get_tsc_cycles();
            return true;
        }
        
        // Not found, trigger ARP request (simplified)
        return request_arp(ip_addr, mac_addr);
    }
    
    void add_entry(uint32_t ip_addr, const struct rte_ether_addr& mac_addr) {
        AddressEntry entry;
        entry.ip_addr = ip_addr;
        entry.mac_addr = mac_addr;
        entry.last_seen_ns = rte_get_tsc_cycles();
        entry.is_valid = true;
        
        cache_[ip_addr] = entry;
    }
    
    void add_static_entry(const std::string& ip_str, const std::string& mac_str) {
        uint32_t ip = inet_addr(ip_str.c_str());
        struct rte_ether_addr mac;
        
        // Parse MAC address
        int values[6];
        if (sscanf(mac_str.c_str(), "%x:%x:%x:%x:%x:%x",
                   &values[0], &values[1], &values[2],
                   &values[3], &values[4], &values[5]) == 6) {
            for (int i = 0; i < 6; i++) {
                mac.addr_bytes[i] = (uint8_t)values[i];
            }
            add_entry(ip, mac);
        }
    }
    
    void cleanup_stale_entries(uint64_t timeout_ns = 300 * 1000000000ULL) {  // 5 minutes
        uint64_t now = rte_get_tsc_cycles();
        
        for (auto it = cache_.begin(); it != cache_.end();) {
            if (now - it->second.last_seen_ns > timeout_ns) {
                it = cache_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
private:
    bool request_arp(uint32_t ip_addr, struct rte_ether_addr& mac_addr) {
        // Simplified: return broadcast MAC for unknown addresses
        memset(&mac_addr, 0xff, sizeof(mac_addr));
        return true;
    }
    
    std::unordered_map<uint32_t, AddressEntry> cache_;
};

/**
 * Security Validator
 * 
 * Validates packets to prevent attacks and corruption
 */
class PacketSecurityValidator {
public:
    PacketSecurityValidator(size_t max_tensor_size = 10ULL * 1024 * 1024 * 1024)  // 10GB max
        : max_tensor_size_(max_tensor_size) {
    }
    
    bool validate_packet(const GeniePacket& pkt, size_t packet_size) {
        // 1. Magic number validation
        if (ntohl(pkt.app.magic) != GENIE_MAGIC) {
            stats_.invalid_magic++;
            return false;
        }
        
        // 2. Version check
        if (pkt.app.version != GENIE_VERSION) {
            stats_.invalid_version++;
            return false;
        }
        
        // 3. Bounds checking
        uint32_t offset = ntohl(pkt.app.offset);
        uint32_t length = ntohl(pkt.app.length);
        uint32_t total_size = ntohl(pkt.app.total_size);
        
        if (offset + length > total_size) {
            stats_.bounds_violation++;
            return false;
        }
        
        // 4. Size sanity checks
        if (total_size > max_tensor_size_) {
            stats_.size_too_large++;
            return false;
        }
        
        if (length > MAX_PACKET_SIZE) {
            stats_.fragment_too_large++;
            return false;
        }
        
        // 5. Fragment validation
        uint16_t frag_id = ntohs(pkt.app.frag_id);
        uint16_t frag_count = ntohs(pkt.app.frag_count);
        
        if (frag_id >= frag_count) {
            stats_.invalid_fragment++;
            return false;
        }
        
        // 6. Replay attack prevention
        uint32_t seq_num = ntohl(pkt.app.seq_num);
        if (!check_replay_window(seq_num)) {
            stats_.replay_attack++;
            return false;
        }
        
        // 7. Checksum validation (if not offloaded)
        if (pkt.app.checksum != 0) {
            uint32_t expected = calculate_checksum(&pkt, packet_size);
            if (ntohl(pkt.app.checksum) != expected) {
                stats_.checksum_fail++;
                return false;
            }
        }
        
        stats_.valid_packets++;
        return true;
    }
    
    bool validate_dma_boundaries(void* gpu_ptr, size_t offset, size_t length, 
                                 void* gpu_base, size_t gpu_size) {
        // Ensure DMA won't write outside allocated GPU memory
        uintptr_t base = reinterpret_cast<uintptr_t>(gpu_base);
        uintptr_t target = reinterpret_cast<uintptr_t>(gpu_ptr) + offset;
        
        if (target < base || target + length > base + gpu_size) {
            stats_.dma_violation++;
            return false;
        }
        
        return true;
    }
    
    struct ValidationStats {
        uint64_t valid_packets = 0;
        uint64_t invalid_magic = 0;
        uint64_t invalid_version = 0;
        uint64_t bounds_violation = 0;
        uint64_t size_too_large = 0;
        uint64_t fragment_too_large = 0;
        uint64_t invalid_fragment = 0;
        uint64_t replay_attack = 0;
        uint64_t checksum_fail = 0;
        uint64_t dma_violation = 0;
    };
    
    const ValidationStats& get_stats() const { return stats_; }
    
private:
    bool check_replay_window(uint32_t seq_num) {
        // Simple replay window check
        if (replay_window_.find(seq_num) != replay_window_.end()) {
            return false;  // Already seen
        }
        
        replay_window_.insert(seq_num);
        
        // Cleanup old entries
        if (replay_window_.size() > MAX_REPLAY_WINDOW) {
            // Remove oldest (simplified - in production use sliding window)
            auto it = replay_window_.begin();
            std::advance(it, replay_window_.size() - MAX_REPLAY_WINDOW);
            replay_window_.erase(replay_window_.begin(), it);
        }
        
        return true;
    }
    
    uint32_t calculate_checksum(const GeniePacket* pkt, size_t size) {
        // Simple checksum calculation (CRC32 would be better)
        const uint8_t* data = reinterpret_cast<const uint8_t*>(pkt);
        uint32_t sum = 0;
        
        for (size_t i = 0; i < size; i++) {
            sum += data[i];
        }
        
        return sum;
    }
    
    static constexpr uint32_t GENIE_MAGIC = 0x47454E49;
    static constexpr uint8_t GENIE_VERSION = 1;
    static constexpr size_t MAX_PACKET_SIZE = 9216;
    static constexpr size_t MAX_REPLAY_WINDOW = 10000;
    
    size_t max_tensor_size_;
    std::set<uint32_t> replay_window_;
    ValidationStats stats_;
};

} // namespace data_plane
} // namespace genie
