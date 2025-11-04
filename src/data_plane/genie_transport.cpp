#include "genie_transport.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <cstring>

using json = nlohmann::json;

namespace genie {

GenieTransport::GenieTransport(const Config& config)
    : config_(config)
    , mbuf_pool_(nullptr)
    , port_id_(0)
    , gpu_dev_id_(-1)
    , gpu_available_(false)
    , running_(false) {
}

GenieTransport::~GenieTransport() {
    running_ = false;
    
    if (mbuf_pool_) {
        rte_mempool_free(mbuf_pool_);
    }
    
    rte_eal_cleanup();
}

bool GenieTransport::initialize() {
    std::cout << "Initializing Genie Transport..." << std::endl;
    
    // Step 1: DPDK
    if (!init_dpdk()) {
        std::cerr << "DPDK initialization failed" << std::endl;
        return false;
    }
    std::cout << "  ✓ DPDK initialized" << std::endl;
    
    // Step 2: Port
    if (!init_port()) {
        std::cerr << "Port initialization failed" << std::endl;
        return false;
    }
    std::cout << "  ✓ Port initialized" << std::endl;
    
    // Step 3: GPU (optional)
    if (config_.use_gpu_direct) {
        if (init_gpu()) {
            std::cout << "  ✓ GPU Direct available" << std::endl;
        } else {
            std::cout << "  ✗ GPU Direct not available (will use CPU staging)" << std::endl;
        }
    }
    
    running_ = true;
    return true;
}

bool GenieTransport::init_dpdk() {
    // Minimal EAL args
    const char* argv[] = {
        "genie",
        "-l", "0-1",  // 2 cores
        "-n", "4",    // 4 mem channels
        "--proc-type", "auto",
    };
    int argc = 7;
    
    int ret = rte_eal_init(argc, (char**)argv);
    if (ret < 0) {
        return false;
    }
    
    // Create mbuf pool
    mbuf_pool_ = rte_pktmbuf_pool_create(
        "mbuf_pool",
        8192,     // num mbufs
        256,      // cache size
        0,        // priv size
        RTE_MBUF_DEFAULT_BUF_SIZE,
        rte_socket_id()
    );
    
    return mbuf_pool_ != nullptr;
}

bool GenieTransport::init_port() {
    // Find first available port
    uint16_t nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0) {
        std::cerr << "No Ethernet ports available" << std::endl;
        return false;
    }
    
    port_id_ = 0;
    
    // Configure port (1 RX queue, 1 TX queue)
    struct rte_eth_conf port_conf = {};
    port_conf.rxmode.mtu = config_.mtu;
    
    int ret = rte_eth_dev_configure(port_id_, 1, 1, &port_conf);
    if (ret < 0) {
        return false;
    }
    
    // Setup RX queue
    ret = rte_eth_rx_queue_setup(
        port_id_, 0, 1024,
        rte_eth_dev_socket_id(port_id_),
        nullptr,
        mbuf_pool_
    );
    if (ret < 0) {
        return false;
    }
    
    // Setup TX queue
    ret = rte_eth_tx_queue_setup(
        port_id_, 0, 1024,
        rte_eth_dev_socket_id(port_id_),
        nullptr
    );
    if (ret < 0) {
        return false;
    }
    
    // Start port
    ret = rte_eth_dev_start(port_id_);
    if (ret < 0) {
        return false;
    }
    
    return true;
}

bool GenieTransport::init_gpu() {
    // Initialize GPUDev
    int ret = rte_gpu_init(16);  // Max 16 GPUs
    if (ret < 0) {
        return false;
    }
    
    // Find first GPU
    gpu_dev_id_ = rte_gpu_find_next(0, RTE_GPU_ID_ANY);
    if (gpu_dev_id_ < 0) {
        return false;
    }
    
    gpu_available_ = true;
    return true;
}

bool GenieTransport::register_gpu_memory(void* ptr, size_t size) {
    if (!gpu_available_) {
        return false;
    }
    
    int ret = rte_gpu_mem_register(gpu_dev_id_, size, ptr);
    return ret == 0;
}

bool GenieTransport::send_tensor(
    void* gpu_ptr,
    size_t size,
    const std::string& target
) {
    // Register GPU memory if not already
    if (gpu_available_) {
        if (!register_gpu_memory(gpu_ptr, size)) {
            std::cerr << "GPU memory registration failed" << std::endl;
            return false;
        }
    }
    
    // Fragment if needed
    size_t max_payload = config_.mtu - 128;  // Reserve for headers
    size_t num_fragments = (size + max_payload - 1) / max_payload;
    
    // Send each fragment
    for (size_t i = 0; i < num_fragments; i++) {
        size_t offset = i * max_payload;
        size_t frag_size = std::min(max_payload, size - offset);
        
        if (!send_packet(
            (const uint8_t*)gpu_ptr + offset,
            frag_size,
            target
        )) {
            return false;
        }
    }
    
    return true;
}

bool GenieTransport::send_packet(
    const uint8_t* data,
    size_t size,
    const std::string& target
) {
    // Allocate mbuf
    struct rte_mbuf* pkt = rte_pktmbuf_alloc(mbuf_pool_);
    if (!pkt) {
        return false;
    }
    
    // Append data
    void* pkt_data = rte_pktmbuf_append(pkt, size);
    if (!pkt_data) {
        rte_pktmbuf_free(pkt);
        return false;
    }
    
    // Copy data (CPU staging if GPU not available)
    std::memcpy(pkt_data, data, size);
    
    // Send via DPDK
    uint16_t nb_tx = rte_eth_tx_burst(port_id_, 0, &pkt, 1);
    
    if (nb_tx == 0) {
        rte_pktmbuf_free(pkt);
        return false;
    }
    
    return true;
}

} // namespace genie

// C API
extern "C" {

void* genie_transport_create(const char* config_json) {
    try {
        auto j = json::parse(config_json);
        
        genie::GenieTransport::Config config;
        config.data_port = j.value("data_port", 5556);
        config.mtu = j.value("mtu", 9000);
        
        auto* transport = new genie::GenieTransport(config);
        if (!transport->initialize()) {
            delete transport;
            return nullptr;
        }
        
        return transport;
    } catch (...) {
        return nullptr;
    }
}

int genie_transport_send(
    void* transport,
    void* gpu_ptr,
    size_t size,
    const char* target
) {
    if (!transport) return -1;
    
    auto* t = static_cast<genie::GenieTransport*>(transport);
    return t->send_tensor(gpu_ptr, size, target) ? 0 : -1;
}

void genie_transport_destroy(void* transport) {
    delete static_cast<genie::GenieTransport*>(transport);
}

} // extern "C"