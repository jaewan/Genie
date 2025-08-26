/**
 * External Buffer TX Implementation
 * 
 * Implements zero-copy TX using rte_pktmbuf_attach_extbuf to attach
 * GPU memory directly to mbufs for DMA by the NIC.
 */

#include "genie_data_plane.hpp"
#include "genie_zero_copy_rx.cpp"  // For AddressResolutionCache

namespace genie {
namespace data_plane {

/**
 * External Buffer Manager for Zero-Copy TX
 * 
 * Manages the attachment of GPU memory to mbufs for zero-copy transmission
 */
class ExternalBufferTxManager {
public:
    ExternalBufferTxManager() {
        // Pre-allocate shared info structures
        shared_infos_.reserve(MAX_CONCURRENT_TRANSFERS);
    }
    
    /**
     * Attach GPU memory to mbuf for zero-copy TX
     * 
     * This is the KEY function that enables zero-copy from GPU to NIC
     */
    bool attach_gpu_memory_to_mbuf(struct rte_mbuf* mbuf,
                                   void* gpu_ptr,
                                   rte_iova_t gpu_iova,
                                   size_t offset,
                                   size_t length) {
        
        // Validate parameters
        if (!mbuf || !gpu_ptr || gpu_iova == RTE_BAD_IOVA || length == 0) {
            return false;
        }
        
        // Get or create shared info
        struct rte_mbuf_ext_shared_info* shinfo = get_shared_info();
        if (!shinfo) {
            return false;
        }
        
        // Initialize shared info
        shinfo->free_cb = gpu_memory_free_cb;
        shinfo->fcb_opaque = this;
        rte_mbuf_ext_refcnt_set(shinfo, 1);
        
        // Calculate actual GPU address and IOVA
        void* data_addr = static_cast<uint8_t*>(gpu_ptr) + offset;
        rte_iova_t data_iova = gpu_iova + offset;
        
        // Attach external GPU buffer to mbuf
        rte_pktmbuf_attach_extbuf(mbuf, data_addr, data_iova, length, shinfo);
        
        // Set mbuf data pointer and length
        mbuf->data_off = 0;  // Data starts at beginning of external buffer
        mbuf->data_len = length;
        mbuf->pkt_len = length;
        
        // Track active external buffer
        track_active_buffer(data_addr, length, shinfo);
        
        stats_.buffers_attached++;
        stats_.bytes_attached += length;
        
        return true;
    }
    
    /**
     * Build complete packet with headers and GPU payload
     */
    struct rte_mbuf* build_zero_copy_packet(struct rte_mempool* pool,
                                           const GeniePacket& headers,
                                           void* gpu_ptr,
                                           rte_iova_t gpu_iova,
                                           size_t offset,
                                           size_t length) {
        
        // Allocate mbuf for headers
        struct rte_mbuf* hdr_mbuf = rte_pktmbuf_alloc(pool);
        if (!hdr_mbuf) {
            return nullptr;
        }
        
        // Write headers to mbuf
        uint8_t* hdr_data = rte_pktmbuf_mtod(hdr_mbuf, uint8_t*);
        memcpy(hdr_data, &headers, sizeof(GeniePacket));
        hdr_mbuf->data_len = sizeof(GeniePacket);
        hdr_mbuf->pkt_len = sizeof(GeniePacket);
        
        // Allocate second mbuf for GPU data (zero-copy)
        struct rte_mbuf* data_mbuf = rte_pktmbuf_alloc(pool);
        if (!data_mbuf) {
            rte_pktmbuf_free(hdr_mbuf);
            return nullptr;
        }
        
        // Attach GPU memory as external buffer
        if (!attach_gpu_memory_to_mbuf(data_mbuf, gpu_ptr, gpu_iova, offset, length)) {
            rte_pktmbuf_free(hdr_mbuf);
            rte_pktmbuf_free(data_mbuf);
            return nullptr;
        }
        
        // Chain mbufs: headers -> GPU data
        rte_pktmbuf_chain(hdr_mbuf, data_mbuf);
        
        // Update total packet length
        hdr_mbuf->pkt_len = sizeof(GeniePacket) + length;
        
        return hdr_mbuf;
    }
    
    /**
     * Fragment large tensor for transmission with zero-copy
     */
    std::vector<struct rte_mbuf*> fragment_tensor_zero_copy(
        struct rte_mempool* pool,
        const std::string& tensor_id,
        void* gpu_ptr,
        rte_iova_t gpu_iova,
        size_t total_size,
        const AddressResolutionCache::AddressEntry& target,
        uint32_t& seq_num) {
        
        std::vector<struct rte_mbuf*> packets;
        
        // Calculate fragments
        size_t mtu_payload = MTU_SIZE - sizeof(GeniePacket);
        size_t num_fragments = (total_size + mtu_payload - 1) / mtu_payload;
        
        for (size_t frag = 0; frag < num_fragments; frag++) {
            size_t offset = frag * mtu_payload;
            size_t length = std::min(mtu_payload, total_size - offset);
            
            // Build headers
            GeniePacket headers;
            build_packet_headers(headers, tensor_id, seq_num++, 
                                frag, num_fragments, offset, length, total_size, target);
            
            // Create zero-copy packet
            struct rte_mbuf* pkt = build_zero_copy_packet(
                pool, headers, gpu_ptr, gpu_iova, offset, length
            );
            
            if (pkt) {
                packets.push_back(pkt);
                stats_.fragments_created++;
            } else {
                // Cleanup on failure
                for (auto* p : packets) {
                    rte_pktmbuf_free(p);
                }
                packets.clear();
                break;
            }
        }
        
        return packets;
    }
    
    struct TxStats {
        uint64_t buffers_attached = 0;
        uint64_t bytes_attached = 0;
        uint64_t fragments_created = 0;
        uint64_t buffers_freed = 0;
    };
    
    const TxStats& get_stats() const { return stats_; }
    
private:
    static void gpu_memory_free_cb(void* addr, void* opaque) {
        // Called when mbuf with external GPU buffer is freed
        ExternalBufferTxManager* mgr = static_cast<ExternalBufferTxManager*>(opaque);
        mgr->release_buffer(addr);
    }
    
    struct rte_mbuf_ext_shared_info* get_shared_info() {
        // Get from pool or allocate new
        if (free_shared_infos_.empty()) {
            shared_infos_.emplace_back();
            return &shared_infos_.back();
        }
        
        auto* info = free_shared_infos_.back();
        free_shared_infos_.pop_back();
        return info;
    }
    
    void track_active_buffer(void* addr, size_t length, 
                            struct rte_mbuf_ext_shared_info* shinfo) {
        ActiveBuffer buffer;
        buffer.addr = addr;
        buffer.length = length;
        buffer.shinfo = shinfo;
        buffer.timestamp = rte_get_tsc_cycles();
        
        active_buffers_[addr] = buffer;
    }
    
    void release_buffer(void* addr) {
        auto it = active_buffers_.find(addr);
        if (it != active_buffers_.end()) {
            // Return shared info to pool
            free_shared_infos_.push_back(it->second.shinfo);
            active_buffers_.erase(it);
            stats_.buffers_freed++;
        }
    }
    
    void build_packet_headers(GeniePacket& pkt,
                             const std::string& tensor_id,
                             uint32_t seq_num,
                             uint16_t frag_id,
                             uint16_t frag_count,
                             uint32_t offset,
                             uint32_t length,
                             uint32_t total_size,
                             const AddressResolutionCache::AddressEntry& target) {
        
        // Ethernet header
        // Source MAC (would be from config)
        static uint8_t src_mac[6] = {0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0x01};
        memcpy(pkt.eth.src_mac, src_mac, 6);
        memcpy(pkt.eth.dst_mac, target.mac_addr.addr_bytes, 6);
        pkt.eth.ethertype = htons(0x0800);  // IPv4
        
        // IPv4 header
        pkt.ip.version_ihl = 0x45;
        pkt.ip.tos = 0;
        pkt.ip.total_length = htons(sizeof(IPv4Header) + sizeof(UDPHeader) + 
                                    sizeof(GeniePacketHeader) + length);
        pkt.ip.identification = htons(seq_num & 0xFFFF);
        pkt.ip.flags_fragment = 0;
        pkt.ip.ttl = 64;
        pkt.ip.protocol = 17;  // UDP
        pkt.ip.checksum = 0;  // Will be calculated
        pkt.ip.src_ip = htonl(0xC0A80164);  // 192.168.1.100
        pkt.ip.dst_ip = target.ip_addr;
        
        // Calculate IP checksum
        pkt.ip.checksum = calculate_ip_checksum(&pkt.ip);
        
        // UDP header
        pkt.udp.src_port = htons(5556);
        pkt.udp.dst_port = htons(5556);
        pkt.udp.length = htons(sizeof(UDPHeader) + sizeof(GeniePacketHeader) + length);
        pkt.udp.checksum = 0;  // Optional for IPv4
        
        // Genie application header
        pkt.app.magic = htonl(0x47454E49);
        pkt.app.version = 1;
        pkt.app.flags = (frag_id == frag_count - 1) ? 0x02 : 0x01;  // LAST_FRAGMENT flag
        pkt.app.type = 0;  // DATA packet
        pkt.app.reserved = 0;
        
        // Copy tensor ID (first 16 bytes)
        memset(pkt.app.tensor_id, 0, 16);
        strncpy(reinterpret_cast<char*>(pkt.app.tensor_id), tensor_id.c_str(), 15);
        
        pkt.app.seq_num = htonl(seq_num);
        pkt.app.frag_id = htons(frag_id);
        pkt.app.frag_count = htons(frag_count);
        pkt.app.offset = htonl(offset);
        pkt.app.length = htonl(length);
        pkt.app.total_size = htonl(total_size);
        pkt.app.checksum = 0;  // Will be offloaded
        pkt.app.timestamp_ns = htobe64(rte_get_tsc_cycles());
        
        memset(pkt.app.padding, 0, 8);
    }
    
    uint16_t calculate_ip_checksum(const IPv4Header* ip) {
        uint32_t sum = 0;
        const uint16_t* ptr = reinterpret_cast<const uint16_t*>(ip);
        
        for (int i = 0; i < 10; i++) {
            if (i != 5) {  // Skip checksum field
                sum += ntohs(ptr[i]);
            }
        }
        
        while (sum >> 16) {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        
        return htons(~sum);
    }
    
    struct ActiveBuffer {
        void* addr;
        size_t length;
        struct rte_mbuf_ext_shared_info* shinfo;
        uint64_t timestamp;
    };
    
    static constexpr size_t MAX_CONCURRENT_TRANSFERS = 1000;
    static constexpr size_t MTU_SIZE = 9000;
    
    std::vector<struct rte_mbuf_ext_shared_info> shared_infos_;
    std::vector<struct rte_mbuf_ext_shared_info*> free_shared_infos_;
    std::unordered_map<void*, ActiveBuffer> active_buffers_;
    TxStats stats_;
};

} // namespace data_plane
} // namespace genie
