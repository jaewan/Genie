/**
 * Genie Enhanced Data Plane with DPDK Libraries
 * 
 * This file contains enhancements using DPDK's built-in libraries
 * for better performance and maintainability.
 */

#include "genie_data_plane.hpp"
#include <rte_ip_frag.h>
#include <rte_reorder.h>
#include <rte_hash.h>
#include <rte_jhash.h>
#include <rte_lru.h>
#include <stdexcept>
#include <iostream>

namespace genie {
namespace data_plane {

/**
 * Enhanced Fragment Manager using rte_ip_frag
 * 
 * Replaces custom fragmentation with DPDK's optimized implementation
 */
class DPDKFragmentManager {
public:
    DPDKFragmentManager(struct rte_mempool* direct_pool, 
                       struct rte_mempool* indirect_pool,
                       uint32_t max_flows = 1024,
                       uint32_t max_frags_per_flow = 64) {
        
        // Create IP fragmentation table
        uint64_t frag_cycles = (rte_get_tsc_hz() + MS_PER_S - 1) / MS_PER_S * FRAGMENT_TIMEOUT_MS;
        
        frag_table_ = rte_ip_frag_table_create(
            max_flows,
            max_frags_per_flow,
            max_flows * max_frags_per_flow,
            frag_cycles,
            rte_socket_id()
        );
        
        if (!frag_table_) {
            throw std::runtime_error("Failed to create IP fragmentation table");
        }
        
        direct_pool_ = direct_pool;
        indirect_pool_ = indirect_pool;
    }
    
    ~DPDKFragmentManager() {
        if (frag_table_) {
            rte_ip_frag_table_destroy(frag_table_);
        }
    }
    
    // Fragment a large packet into MTU-sized fragments
    int fragment_packet(struct rte_mbuf* pkt, 
                       struct rte_mbuf** pkts_out,
                       uint16_t max_pkts,
                       uint16_t mtu) {
        
        // Get IP header
        struct rte_ipv4_hdr* ip_hdr = rte_pktmbuf_mtod_offset(
            pkt, struct rte_ipv4_hdr*, sizeof(struct rte_ether_hdr)
        );
        
        // Fragment using DPDK library
        int nb_frags = rte_ipv4_fragment_packet(
            pkt,
            pkts_out,
            max_pkts,
            mtu,
            direct_pool_,
            indirect_pool_
        );
        
        if (nb_frags < 0) {
            return -1;
        }
        
        // Update checksums for each fragment
        for (int i = 0; i < nb_frags; i++) {
            struct rte_ipv4_hdr* frag_ip = rte_pktmbuf_mtod_offset(
                pkts_out[i], struct rte_ipv4_hdr*, sizeof(struct rte_ether_hdr)
            );
            frag_ip->hdr_checksum = rte_ipv4_cksum(frag_ip);
        }
        
        return nb_frags;
    }
    
    // Reassemble fragments into complete packet
    struct rte_mbuf* reassemble_packet(struct rte_mbuf* pkt) {
        uint64_t cur_tsc = rte_rdtsc();
        
        // Get IP header
        struct rte_ipv4_hdr* ip_hdr = rte_pktmbuf_mtod_offset(
            pkt, struct rte_ipv4_hdr*, sizeof(struct rte_ether_hdr)
        );
        
        // Try to reassemble
        struct rte_mbuf* reassembled = rte_ipv4_frag_reassemble_packet(
            frag_table_,
            &death_row_,
            pkt,
            cur_tsc,
            ip_hdr
        );
        
        // Free expired fragments
        rte_ip_frag_free_death_row(&death_row_, PREFETCH_OFFSET);
        
        return reassembled;
    }
    
    // Check for timeouts
    void check_timeouts() {
        uint64_t cur_tsc = rte_rdtsc();
        rte_ip_frag_table_statistics_dump(stdout, frag_table_);
        
        // This will mark timed-out fragments for deletion
        // They'll be freed on next reassemble_packet call
    }
    
private:
    static constexpr uint32_t FRAGMENT_TIMEOUT_MS = 2000;  // 2 seconds
    // MS_PER_S already defined by DPDK
    static constexpr uint32_t PREFETCH_OFFSET = 3;
    
    struct rte_ip_frag_tbl* frag_table_;
    struct rte_ip_frag_death_row death_row_ = {};
    struct rte_mempool* direct_pool_;
    struct rte_mempool* indirect_pool_;
};

/**
 * Enhanced Packet Reorder using rte_reorder
 * 
 * Handles out-of-order packet delivery efficiently
 */
class DPDKReorderBuffer {
public:
    DPDKReorderBuffer(const char* name, uint32_t size = 1024) {
        reorder_buffer_ = rte_reorder_create(
            name,
            rte_socket_id(),
            size
        );
        
        if (!reorder_buffer_) {
            throw std::runtime_error("Failed to create reorder buffer");
        }
        
        expected_seq_ = 0;
    }
    
    ~DPDKReorderBuffer() {
        if (reorder_buffer_) {
            // Drain any remaining packets
            struct rte_mbuf* pkts[32];
            while (rte_reorder_drain(reorder_buffer_, pkts, 32) > 0) {
                // Free packets
                for (int i = 0; i < 32 && pkts[i]; i++) {
                    rte_pktmbuf_free(pkts[i]);
                }
            }
            rte_reorder_free(reorder_buffer_);
        }
    }
    
    // Insert packet with sequence number
    int insert_packet(struct rte_mbuf* pkt, uint32_t seq_num) {
        return rte_reorder_insert(reorder_buffer_, pkt);
    }
    
    // Get ordered packets
    uint32_t get_ordered_packets(struct rte_mbuf** pkts, uint32_t max_pkts) {
        return rte_reorder_drain(reorder_buffer_, pkts, max_pkts);
    }
    
    // Get statistics
    void get_stats() const {
        // Note: rte_reorder doesn't have built-in stats
        // We'd track these ourselves in production
    }
    
private:
    struct rte_reorder_buffer* reorder_buffer_;
    uint32_t expected_seq_;
};

/**
 * Enhanced Connection Tracking using rte_hash
 * 
 * Efficiently tracks active connections and their state
 */
class DPDKConnectionTracker {
public:
    struct ConnectionKey {
        uint32_t src_ip;
        uint32_t dst_ip;
        uint16_t src_port;
        uint16_t dst_port;
        uint8_t protocol;
        uint8_t padding[3];
    } __attribute__((packed));
    
    struct ConnectionState {
        std::string transfer_id;
        std::string tensor_id;
        uint64_t bytes_transferred;
        uint32_t packets_received;
        uint32_t last_seq_num;
        std::chrono::steady_clock::time_point last_activity;
        DPDKReorderBuffer* reorder_buffer;
        DPDKFragmentManager* fragment_manager;
        
        ConnectionState() : reorder_buffer(nullptr), fragment_manager(nullptr) {}
        ~ConnectionState() {
            delete reorder_buffer;
            // fragment_manager is shared, don't delete
        }
    };
    
    DPDKConnectionTracker(uint32_t max_connections = 4096) 
        : max_connections_(max_connections) {
        
        // Configure hash table
        struct rte_hash_parameters hash_params = {};
        hash_params.name = "conn_hash";
        hash_params.entries = max_connections;
        hash_params.key_len = sizeof(ConnectionKey);
        hash_params.hash_func = rte_jhash;
        hash_params.hash_func_init_val = 0;
        hash_params.socket_id = rte_socket_id();
        hash_params.extra_flag = RTE_HASH_EXTRA_FLAGS_RW_CONCURRENCY;
        
        // Create hash table
        hash_table_ = rte_hash_create(&hash_params);
        if (!hash_table_) {
            throw std::runtime_error("Failed to create connection hash table");
        }
        
        // Pre-allocate connection states
        connection_states_.resize(max_connections);
    }
    
    ~DPDKConnectionTracker() {
        if (hash_table_) {
            rte_hash_free(hash_table_);
        }
    }
    
    // Add or update connection
    ConnectionState* add_connection(const ConnectionKey& key, 
                                   const std::string& transfer_id,
                                   const std::string& tensor_id) {
        int32_t pos = rte_hash_add_key(hash_table_, &key);
        if (pos < 0) {
            return nullptr;
        }
        
        ConnectionState* state = &connection_states_[pos];
        state->transfer_id = transfer_id;
        state->tensor_id = tensor_id;
        state->bytes_transferred = 0;
        state->packets_received = 0;
        state->last_seq_num = 0;
        state->last_activity = std::chrono::steady_clock::now();
        
        // Create reorder buffer for this connection
        char reorder_name[64];
        snprintf(reorder_name, sizeof(reorder_name), "reorder_%d", pos);
        state->reorder_buffer = new DPDKReorderBuffer(reorder_name);
        
        return state;
    }
    
    // Lookup connection
    ConnectionState* lookup_connection(const ConnectionKey& key) {
        int32_t pos = rte_hash_lookup(hash_table_, &key);
        if (pos < 0) {
            return nullptr;
        }
        
        ConnectionState* state = &connection_states_[pos];
        state->last_activity = std::chrono::steady_clock::now();
        return state;
    }
    
    // Remove connection
    void remove_connection(const ConnectionKey& key) {
        int32_t pos = rte_hash_del_key(hash_table_, &key);
        if (pos >= 0) {
            // Clean up connection state
            ConnectionState* state = &connection_states_[pos];
            delete state->reorder_buffer;
            state->reorder_buffer = nullptr;
            *state = ConnectionState();  // Reset
        }
    }
    
    // Extract connection key from packet
    static ConnectionKey extract_key(const struct rte_mbuf* pkt) {
        ConnectionKey key = {};
        
        // Get headers
        struct rte_ether_hdr* eth = rte_pktmbuf_mtod(pkt, struct rte_ether_hdr*);
        struct rte_ipv4_hdr* ip = (struct rte_ipv4_hdr*)(eth + 1);
        struct rte_udp_hdr* udp = (struct rte_udp_hdr*)(ip + 1);
        
        key.src_ip = ip->src_addr;
        key.dst_ip = ip->dst_addr;
        key.src_port = udp->src_port;
        key.dst_port = udp->dst_port;
        key.protocol = ip->next_proto_id;
        
        return key;
    }
    
    // Clean up stale connections
    void cleanup_stale_connections(std::chrono::seconds timeout = std::chrono::seconds(30)) {
        auto now = std::chrono::steady_clock::now();
        const void* key = nullptr;
        uint32_t iter = 0;
        int32_t pos;
        
        while ((pos = rte_hash_iterate(hash_table_, &key, (void**)&pos, &iter)) >= 0) {
            ConnectionState* state = &connection_states_[pos];
            if (now - state->last_activity > timeout) {
                // Connection is stale, remove it
                rte_hash_del_key(hash_table_, key);
                delete state->reorder_buffer;
                *state = ConnectionState();
            }
        }
    }
    
    // Get statistics
    void print_stats() const {
        printf("Connection Tracker Statistics:\n");
        printf("  Max connections: %u\n", max_connections_);
        printf("  Active connections: %u\n", rte_hash_count(hash_table_));
        
        // Print hash table stats
        rte_hash_dump(hash_table_);
    }
    
private:
    struct rte_hash* hash_table_;
    std::vector<ConnectionState> connection_states_;
    uint32_t max_connections_;
};

/**
 * Enhanced Data Plane with DPDK Libraries
 * 
 * This class extends the base GenieDataPlane with optimized
 * implementations using DPDK's built-in libraries.
 */
class EnhancedDataPlane : public GenieDataPlane {
public:
    EnhancedDataPlane(const DataPlaneConfig& config) 
        : GenieDataPlane(config) {
        
        // Initialize enhanced components after base initialization
    }
    
    bool initialize() override {
        // Initialize base first
        if (!GenieDataPlane::initialize()) {
            return false;
        }
        
        try {
            // Create fragment manager with DPDK implementation
            fragment_manager_ = std::make_unique<DPDKFragmentManager>(
                mempool_,  // direct pool
                mempool_   // indirect pool (same for now)
            );
            
            // Create connection tracker
            connection_tracker_ = std::make_unique<DPDKConnectionTracker>(
                config_.max_connections
            );
            
            std::cout << "Enhanced DPDK components initialized:" << std::endl;
            std::cout << "  - rte_ip_frag for fragmentation/reassembly" << std::endl;
            std::cout << "  - rte_reorder for packet ordering" << std::endl;
            std::cout << "  - rte_hash for connection tracking" << std::endl;
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize enhanced components: " << e.what() << std::endl;
            return false;
        }
    }
    
protected:
    // Override packet processing to use enhanced components
    void process_rx_packets(struct rte_mbuf** pkts, uint16_t nb_pkts) override {
        for (uint16_t i = 0; i < nb_pkts; i++) {
            struct rte_mbuf* pkt = pkts[i];
            
            // Extract connection key
            auto key = DPDKConnectionTracker::extract_key(pkt);
            
            // Look up or create connection state
            auto* conn_state = connection_tracker_->lookup_connection(key);
            if (!conn_state) {
                // New connection - would extract transfer info from packet
                // For now, create with placeholder info
                conn_state = connection_tracker_->add_connection(
                    key, "transfer_" + std::to_string(i), "tensor_" + std::to_string(i)
                );
            }
            
            if (conn_state) {
                // Check if packet is fragmented
                struct rte_ipv4_hdr* ip_hdr = rte_pktmbuf_mtod_offset(
                    pkt, struct rte_ipv4_hdr*, sizeof(struct rte_ether_hdr)
                );
                
                if (rte_ipv4_frag_pkt_is_fragmented(ip_hdr)) {
                    // Reassemble fragments
                    struct rte_mbuf* reassembled = fragment_manager_->reassemble_packet(pkt);
                    if (reassembled) {
                        pkt = reassembled;
                    } else {
                        // Fragment stored, waiting for more
                        continue;
                    }
                }
                
                // Extract sequence number from our custom header
                GeniePacket* genie_pkt = rte_pktmbuf_mtod(pkt, GeniePacket*);
                uint32_t seq_num = ntohl(genie_pkt->app.seq_num);
                
                // Insert into reorder buffer
                if (conn_state->reorder_buffer) {
                    conn_state->reorder_buffer->insert_packet(pkt, seq_num);
                    
                    // Try to get ordered packets
                    struct rte_mbuf* ordered_pkts[32];
                    uint32_t nb_ordered = conn_state->reorder_buffer->get_ordered_packets(
                        ordered_pkts, 32
                    );
                    
                    // Process ordered packets
                    for (uint32_t j = 0; j < nb_ordered; j++) {
                        process_ordered_packet(ordered_pkts[j], conn_state);
                    }
                }
                
                // Update connection stats
                conn_state->packets_received++;
                conn_state->last_seq_num = seq_num;
            } else {
                // Failed to track connection
                rte_pktmbuf_free(pkt);
            }
        }
        
        // Periodically clean up stale connections
        static auto last_cleanup = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (now - last_cleanup > std::chrono::seconds(10)) {
            connection_tracker_->cleanup_stale_connections();
            last_cleanup = now;
        }
    }
    
    // Override fragmentation to use DPDK implementation
    bool fragment_and_send(const std::string& transfer_id,
                          const std::string& tensor_id,
                          const uint8_t* data,
                          size_t size,
                          const std::string& target_node) override {
        
        // Create large packet with all data
        struct rte_mbuf* large_pkt = rte_pktmbuf_alloc(mempool_);
        if (!large_pkt) {
            return false;
        }
        
        // Build packet headers (simplified)
        // In production, this would properly construct all headers
        
        // Fragment using DPDK
        struct rte_mbuf* fragments[64];
        int nb_frags = fragment_manager_->fragment_packet(
            large_pkt,
            fragments,
            64,
            config_.mtu - sizeof(GeniePacket)
        );
        
        if (nb_frags < 0) {
            rte_pktmbuf_free(large_pkt);
            return false;
        }
        
        // Send fragments
        for (int i = 0; i < nb_frags; i++) {
            if (rte_ring_enqueue(tx_ring_, fragments[i]) != 0) {
                // Failed to enqueue
                for (int j = i; j < nb_frags; j++) {
                    rte_pktmbuf_free(fragments[j]);
                }
                return false;
            }
        }
        
        return true;
    }
    
private:
    void process_ordered_packet(struct rte_mbuf* pkt, 
                               DPDKConnectionTracker::ConnectionState* conn_state) {
        // Process packet that's now in order
        GeniePacket parsed_pkt;
        uint8_t* payload;
        uint32_t payload_size;
        
        if (packet_processor_->parse_packet(pkt, parsed_pkt, payload, payload_size)) {
            // Update connection state
            conn_state->bytes_transferred += payload_size;
            
            // Handle based on packet type
            PacketType pkt_type = static_cast<PacketType>(parsed_pkt.app.type);
            switch (pkt_type) {
                case PacketType::DATA:
                    // Process data packet
                    handle_data_packet(parsed_pkt, payload, payload_size, conn_state);
                    break;
                case PacketType::ACK:
                    handle_ack(ntohl(parsed_pkt.app.seq_num));
                    break;
                case PacketType::NACK:
                    // Handle NACK
                    break;
                default:
                    break;
            }
        }
        
        rte_pktmbuf_free(pkt);
    }
    
    void handle_data_packet(const GeniePacket& pkt, 
                           const uint8_t* payload, 
                           uint32_t payload_size,
                           DPDKConnectionTracker::ConnectionState* conn_state) {
        // Implementation would handle the data packet
        // This is simplified - real implementation would reassemble tensor
        std::cout << "Received ordered data packet for transfer " 
                  << conn_state->transfer_id 
                  << ", size=" << payload_size << std::endl;
    }
    
private:
    std::unique_ptr<DPDKFragmentManager> fragment_manager_;
    std::unique_ptr<DPDKConnectionTracker> connection_tracker_;
};

} // namespace data_plane
} // namespace genie
