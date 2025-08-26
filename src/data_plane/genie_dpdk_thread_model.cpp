/**
 * DPDK Native Thread Model Implementation
 * 
 * Uses DPDK's standard lcore and service core framework
 */

#include "genie_dpdk_thread_model.hpp"
#include <iostream>
#include <iomanip>
#include <unistd.h>  // for sleep
#include <rte_malloc.h>
#include <rte_log.h>

namespace genie {
namespace data_plane {

// Static port configuration
const struct rte_eth_conf DPDKApplication::port_conf_default = {
    .rxmode = {
        .mq_mode = RTE_ETH_MQ_RX_NONE,
    },
    .txmode = {
        .mq_mode = RTE_ETH_MQ_TX_NONE,
    },
};

// ============================================================================
// DPDKThreadManager Implementation
// ============================================================================

DPDKThreadManager::DPDKThreadManager() 
    : rx_to_worker_ring_(nullptr), worker_to_tx_ring_(nullptr) {
    lcore_configs_.resize(RTE_MAX_LCORE);
    lcore_stats_.resize(RTE_MAX_LCORE);
}

DPDKThreadManager::~DPDKThreadManager() {
    stop_all();
    destroy_rings();
}

bool DPDKThreadManager::initialize() {
    // Create inter-lcore communication rings
    if (!create_rings()) {
        return false;
    }
    
    std::cout << "DPDK Thread Manager initialized" << std::endl;
    std::cout << "Available lcores: ";
    
    unsigned lcore_id;
    RTE_LCORE_FOREACH(lcore_id) {
        std::cout << lcore_id << " ";
    }
    std::cout << std::endl;
    
    return true;
}

bool DPDKThreadManager::create_rings() {
    // Create ring for RX -> Worker communication
    rx_to_worker_ring_ = rte_ring_create("rx_to_worker",
                                         1024,
                                         rte_socket_id(),
                                         RING_F_SP_ENQ | RING_F_SC_DEQ);
    if (!rx_to_worker_ring_) {
        std::cerr << "Failed to create rx_to_worker ring" << std::endl;
        return false;
    }
    
    // Create ring for Worker -> TX communication
    worker_to_tx_ring_ = rte_ring_create("worker_to_tx",
                                         1024,
                                         rte_socket_id(),
                                         RING_F_SP_ENQ | RING_F_SC_DEQ);
    if (!worker_to_tx_ring_) {
        rte_ring_free(rx_to_worker_ring_);
        rx_to_worker_ring_ = nullptr;
        std::cerr << "Failed to create worker_to_tx ring" << std::endl;
        return false;
    }
    
    return true;
}

void DPDKThreadManager::destroy_rings() {
    if (rx_to_worker_ring_) {
        rte_ring_free(rx_to_worker_ring_);
        rx_to_worker_ring_ = nullptr;
    }
    if (worker_to_tx_ring_) {
        rte_ring_free(worker_to_tx_ring_);
        worker_to_tx_ring_ = nullptr;
    }
}

bool DPDKThreadManager::assign_lcore(unsigned lcore_id, 
                                     LcoreConfig::LcoreRole role,
                                     uint16_t port_id,
                                     uint16_t queue_id) {
    if (lcore_id >= RTE_MAX_LCORE) {
        std::cerr << "Invalid lcore_id: " << lcore_id << std::endl;
        return false;
    }
    
    if (!rte_lcore_is_enabled(lcore_id)) {
        std::cerr << "Lcore " << lcore_id << " is not enabled" << std::endl;
        return false;
    }
    
    LcoreConfig& config = lcore_configs_[lcore_id];
    config.lcore_id = lcore_id;
    config.role = role;
    config.port_id = port_id;
    config.queue_id = queue_id;
    config.rx_ring = rx_to_worker_ring_;
    config.tx_ring = worker_to_tx_ring_;
    config.user_data = this;
    
    std::cout << "Assigned lcore " << lcore_id << " to role ";
    switch (role) {
        case LcoreConfig::RX_LCORE:
            std::cout << "RX (port=" << port_id << ", queue=" << queue_id << ")";
            break;
        case LcoreConfig::TX_LCORE:
            std::cout << "TX (port=" << port_id << ", queue=" << queue_id << ")";
            break;
        case LcoreConfig::WORKER_LCORE:
            std::cout << "WORKER";
            break;
        case LcoreConfig::SERVICE_LCORE:
            std::cout << "SERVICE";
            break;
    }
    std::cout << std::endl;
    
    return true;
}

bool DPDKThreadManager::launch_all() {
    force_quit_ = false;
    
    unsigned lcore_id;
    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        LcoreConfig& config = lcore_configs_[lcore_id];
        
        int (*func)(void*) = nullptr;
        switch (config.role) {
            case LcoreConfig::RX_LCORE:
                func = rx_lcore_main;
                break;
            case LcoreConfig::TX_LCORE:
                func = tx_lcore_main;
                break;
            case LcoreConfig::WORKER_LCORE:
                func = worker_lcore_main;
                break;
            default:
                continue;  // Skip unassigned lcores
        }
        
        if (func) {
            int ret = rte_eal_remote_launch(func, 
                                           &config, lcore_id);
            if (ret < 0) {
                std::cerr << "Failed to launch lcore " << lcore_id << std::endl;
                return false;
            }
        }
    }
    
    std::cout << "All lcores launched successfully" << std::endl;
    return true;
}

void DPDKThreadManager::wait_all() {
    unsigned lcore_id;
    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        if (rte_eal_wait_lcore(lcore_id) < 0) {
            std::cerr << "Error waiting for lcore " << lcore_id << std::endl;
        }
    }
}

void DPDKThreadManager::stop_all() {
    force_quit_ = true;
    wait_all();
}

void DPDKThreadManager::print_stats() const {
    std::cout << "\n=== DPDK Thread Statistics ===" << std::endl;
    
    unsigned lcore_id;
    RTE_LCORE_FOREACH(lcore_id) {
        const LcoreConfig& config = lcore_configs_[lcore_id];
        const LcoreStats& stats = lcore_stats_[lcore_id];
        
        if (stats.packets_processed > 0) {
            std::cout << "\nLcore " << lcore_id;
            
            switch (config.role) {
                case LcoreConfig::RX_LCORE: std::cout << " (RX)"; break;
                case LcoreConfig::TX_LCORE: std::cout << " (TX)"; break;
                case LcoreConfig::WORKER_LCORE: std::cout << " (WORKER)"; break;
                case LcoreConfig::SERVICE_LCORE: std::cout << " (SERVICE)"; break;
            }
            
            std::cout << ":" << std::endl;
            std::cout << "  Packets: " << stats.packets_processed << std::endl;
            std::cout << "  Bytes: " << stats.bytes_processed << std::endl;
            std::cout << "  Bursts: " << stats.bursts << std::endl;
            
            uint64_t total_cycles = stats.busy_cycles + stats.idle_cycles;
            if (total_cycles > 0) {
                double utilization = 100.0 * stats.busy_cycles / total_cycles;
                std::cout << "  CPU Utilization: " << std::fixed 
                         << std::setprecision(2) << utilization << "%" << std::endl;
            }
        }
    }
}

// Static lcore main functions
int DPDKThreadManager::rx_lcore_main(void* arg) {
    LcoreConfig* config = static_cast<LcoreConfig*>(arg);
    DPDKThreadManager* manager = static_cast<DPDKThreadManager*>(config->user_data);
    LcoreStats& stats = manager->lcore_stats_[config->lcore_id];
    
    std::cout << "RX lcore " << config->lcore_id << " started on port " 
              << config->port_id << std::endl;
    
    struct rte_mbuf* bufs[BURST_SIZE];
    
    while (!manager->force_quit_) {
        uint64_t start_tsc = rte_rdtsc();
        
        // Receive packets
        const uint16_t nb_rx = rte_eth_rx_burst(config->port_id,
                                                config->queue_id,
                                                bufs, BURST_SIZE);
        
        if (likely(nb_rx > 0)) {
            stats.packets_processed += nb_rx;
            stats.bursts++;
            
            // Forward to worker ring
            for (uint16_t i = 0; i < nb_rx; i++) {
                stats.bytes_processed += bufs[i]->pkt_len;
                
                if (rte_ring_enqueue(config->rx_ring, bufs[i]) < 0) {
                    // Ring full, drop packet
                    rte_pktmbuf_free(bufs[i]);
                }
            }
            
            stats.busy_cycles += rte_rdtsc() - start_tsc;
        } else {
            stats.idle_cycles += rte_rdtsc() - start_tsc;
        }
    }
    
    std::cout << "RX lcore " << config->lcore_id << " stopped" << std::endl;
    return 0;
}

int DPDKThreadManager::tx_lcore_main(void* arg) {
    LcoreConfig* config = static_cast<LcoreConfig*>(arg);
    DPDKThreadManager* manager = static_cast<DPDKThreadManager*>(config->user_data);
    LcoreStats& stats = manager->lcore_stats_[config->lcore_id];
    
    std::cout << "TX lcore " << config->lcore_id << " started on port " 
              << config->port_id << std::endl;
    
    struct rte_mbuf* bufs[BURST_SIZE];
    
    while (!manager->force_quit_) {
        uint64_t start_tsc = rte_rdtsc();
        
        // Dequeue from worker ring
        unsigned nb_tx = rte_ring_dequeue_burst(config->tx_ring,
                                               (void**)bufs,
                                               BURST_SIZE,
                                               nullptr);
        
        if (likely(nb_tx > 0)) {
            // Transmit packets
            uint16_t nb_sent = rte_eth_tx_burst(config->port_id,
                                               config->queue_id,
                                               bufs, nb_tx);
            
            stats.packets_processed += nb_sent;
            stats.bursts++;
            
            // Free any unsent packets
            if (unlikely(nb_sent < nb_tx)) {
                for (uint16_t i = nb_sent; i < nb_tx; i++) {
                    rte_pktmbuf_free(bufs[i]);
                }
            }
            
            // Update byte count
            for (uint16_t i = 0; i < nb_sent; i++) {
                stats.bytes_processed += bufs[i]->pkt_len;
            }
            
            stats.busy_cycles += rte_rdtsc() - start_tsc;
        } else {
            stats.idle_cycles += rte_rdtsc() - start_tsc;
        }
    }
    
    std::cout << "TX lcore " << config->lcore_id << " stopped" << std::endl;
    return 0;
}

int DPDKThreadManager::worker_lcore_main(void* arg) {
    LcoreConfig* config = static_cast<LcoreConfig*>(arg);
    DPDKThreadManager* manager = static_cast<DPDKThreadManager*>(config->user_data);
    LcoreStats& stats = manager->lcore_stats_[config->lcore_id];
    
    std::cout << "Worker lcore " << config->lcore_id << " started" << std::endl;
    
    void* obj;
    
    while (!manager->force_quit_) {
        uint64_t start_tsc = rte_rdtsc();
        
        // Dequeue from RX ring
        if (rte_ring_dequeue(config->rx_ring, &obj) == 0) {
            struct rte_mbuf* mbuf = static_cast<struct rte_mbuf*>(obj);
            
            // Process packet (simplified - just forward)
            stats.packets_processed++;
            stats.bytes_processed += mbuf->pkt_len;
            
            // Forward to TX ring
            if (rte_ring_enqueue(config->tx_ring, mbuf) < 0) {
                // Ring full, drop packet
                rte_pktmbuf_free(mbuf);
            }
            
            stats.busy_cycles += rte_rdtsc() - start_tsc;
        } else {
            stats.idle_cycles += rte_rdtsc() - start_tsc;
        }
    }
    
    std::cout << "Worker lcore " << config->lcore_id << " stopped" << std::endl;
    return 0;
}

// ============================================================================
// ServiceCoreManager Implementation
// ============================================================================

ServiceCoreManager::ServiceCoreManager() {
}

ServiceCoreManager::~ServiceCoreManager() {
    stop_service_cores();
}

bool ServiceCoreManager::register_service(const char* name,
                                         rte_service_func service_func,
                                         void* user_data) {
    struct rte_service_spec service_spec = {};
    snprintf(service_spec.name, sizeof(service_spec.name), "%s", name);
    service_spec.callback = service_func;
    service_spec.callback_userdata = user_data;
    service_spec.capabilities = 0;
    
    uint32_t service_id;
    int ret = rte_service_component_register(&service_spec, &service_id);
    if (ret != 0) {
        std::cerr << "Failed to register service " << name 
                  << ": " << rte_strerror(-ret) << std::endl;
        return false;
    }
    
    service_ids_.push_back(service_id);
    std::cout << "Registered service '" << name << "' with ID " 
              << service_id << std::endl;
    
    return true;
}

bool ServiceCoreManager::map_service_to_lcore(uint32_t service_id, 
                                              uint32_t lcore_id) {
    // Set lcore as service lcore
    int ret = rte_service_lcore_add(lcore_id);
    if (ret != 0 && ret != -EALREADY) {
        std::cerr << "Failed to add lcore " << lcore_id 
                  << " as service lcore: " << rte_strerror(-ret) << std::endl;
        return false;
    }
    
    // Map service to lcore
    ret = rte_service_map_lcore_set(service_id, lcore_id, 1);
    if (ret != 0) {
        std::cerr << "Failed to map service " << service_id 
                  << " to lcore " << lcore_id << ": " 
                  << rte_strerror(-ret) << std::endl;
        return false;
    }
    
    service_lcores_.push_back(lcore_id);
    std::cout << "Mapped service " << service_id 
              << " to lcore " << lcore_id << std::endl;
    
    return true;
}

bool ServiceCoreManager::start_service_cores() {
    // Start all service lcores
    int ret = rte_service_lcore_start(rte_service_lcore_list(nullptr, 0));
    if (ret < 0) {
        std::cerr << "Failed to start service cores: " 
                  << rte_strerror(-ret) << std::endl;
        return false;
    }
    
    // Enable all registered services
    for (uint32_t service_id : service_ids_) {
        ret = rte_service_component_runstate_set(service_id, 1);
        if (ret != 0) {
            std::cerr << "Failed to enable service " << service_id 
                      << ": " << rte_strerror(-ret) << std::endl;
        }
    }
    
    std::cout << "Service cores started" << std::endl;
    return true;
}

void ServiceCoreManager::stop_service_cores() {
    // Disable all services
    for (uint32_t service_id : service_ids_) {
        rte_service_component_runstate_set(service_id, 0);
    }
    
    // Stop service lcores
    for (uint32_t lcore_id : service_lcores_) {
        rte_service_lcore_stop(lcore_id);
    }
}

// Example service functions
int32_t ServiceCoreManager::stats_service(void* args) {
    static uint64_t counter = 0;
    counter++;
    
    // Print stats every 1000 iterations (adjust as needed)
    if (counter % 1000 == 0) {
        std::cout << "Stats service iteration: " << counter << std::endl;
    }
    
    return 0;
}

int32_t ServiceCoreManager::timeout_service(void* args) {
    // Check for timeouts (simplified)
    return 0;
}

// ============================================================================
// DPDKApplication Implementation
// ============================================================================

DPDKApplication::DPDKApplication() 
    : mbuf_pool_(nullptr), nb_ports_(0) {
}

DPDKApplication::~DPDKApplication() {
    cleanup();
}

bool DPDKApplication::init_eal(int argc, char* argv[]) {
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        std::cerr << "Error with EAL initialization" << std::endl;
        return false;
    }
    
    std::cout << "DPDK EAL initialized with " << rte_lcore_count() 
              << " logical cores" << std::endl;
    
    return true;
}

bool DPDKApplication::init_ports(uint16_t nb_ports) {
    nb_ports_ = nb_ports;
    
    // Create mbuf pool
    mbuf_pool_ = rte_pktmbuf_pool_create("MBUF_POOL",
                                         DPDKThreadManager::NUM_MBUFS * nb_ports,
                                         DPDKThreadManager::MBUF_CACHE_SIZE,
                                         0,
                                         RTE_MBUF_DEFAULT_BUF_SIZE,
                                         rte_socket_id());
    
    if (mbuf_pool_ == nullptr) {
        std::cerr << "Cannot create mbuf pool" << std::endl;
        return false;
    }
    
    // Initialize all ports
    uint16_t portid;
    RTE_ETH_FOREACH_DEV(portid) {
        if (portid >= nb_ports)
            break;
            
        if (!port_init(portid, mbuf_pool_))
            return false;
    }
    
    return true;
}

bool DPDKApplication::port_init(uint16_t port, struct rte_mempool* mbuf_pool) {
    struct rte_eth_conf port_conf = port_conf_default;
    const uint16_t rx_rings = 1, tx_rings = 1;
    uint16_t nb_rxd = DPDKThreadManager::RX_RING_SIZE;
    uint16_t nb_txd = DPDKThreadManager::TX_RING_SIZE;
    int retval;
    struct rte_eth_dev_info dev_info;
    struct rte_eth_txconf txconf;
    
    if (!rte_eth_dev_is_valid_port(port))
        return false;
    
    retval = rte_eth_dev_info_get(port, &dev_info);
    if (retval != 0) {
        std::cerr << "Error during getting device (port " << port 
                  << ") info: " << strerror(-retval) << std::endl;
        return false;
    }
    
    // Configure the Ethernet device
    retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);
    if (retval != 0) {
        std::cerr << "Cannot configure device: err=" << retval 
                  << ", port=" << port << std::endl;
        return false;
    }
    
    retval = rte_eth_dev_adjust_nb_rx_tx_desc(port, &nb_rxd, &nb_txd);
    if (retval != 0) {
        std::cerr << "Cannot adjust number of descriptors: err=" << retval
                  << ", port=" << port << std::endl;
        return false;
    }
    
    // Allocate and set up RX queue
    for (uint16_t q = 0; q < rx_rings; q++) {
        retval = rte_eth_rx_queue_setup(port, q, nb_rxd,
                                        rte_eth_dev_socket_id(port),
                                        nullptr, mbuf_pool);
        if (retval < 0) {
            std::cerr << "RX queue setup failed: err=" << retval 
                      << ", port=" << port << std::endl;
            return false;
        }
    }
    
    txconf = dev_info.default_txconf;
    txconf.offloads = port_conf.txmode.offloads;
    
    // Allocate and set up TX queue
    for (uint16_t q = 0; q < tx_rings; q++) {
        retval = rte_eth_tx_queue_setup(port, q, nb_txd,
                                        rte_eth_dev_socket_id(port),
                                        &txconf);
        if (retval < 0) {
            std::cerr << "TX queue setup failed: err=" << retval 
                      << ", port=" << port << std::endl;
            return false;
        }
    }
    
    // Start the Ethernet port
    retval = rte_eth_dev_start(port);
    if (retval < 0) {
        std::cerr << "Cannot start device: err=" << retval 
                  << ", port=" << port << std::endl;
        return false;
    }
    
    // Display the port MAC address
    struct rte_ether_addr addr;
    retval = rte_eth_macaddr_get(port, &addr);
    if (retval != 0) {
        std::cerr << "Cannot get MAC address: err=" << retval 
                  << ", port=" << port << std::endl;
        return false;
    }
    
    std::cout << "Port " << port << " MAC: ";
    printf("%02x:%02x:%02x:%02x:%02x:%02x\n",
           addr.addr_bytes[0], addr.addr_bytes[1],
           addr.addr_bytes[2], addr.addr_bytes[3],
           addr.addr_bytes[4], addr.addr_bytes[5]);
    
    // Enable RX in promiscuous mode for the Ethernet device
    retval = rte_eth_promiscuous_enable(port);
    if (retval != 0) {
        std::cerr << "Cannot enable promiscuous mode: err=" << retval
                  << ", port=" << port << std::endl;
        return false;
    }
    
    return true;
}

bool DPDKApplication::setup_lcores() {
    thread_manager_ = std::make_unique<DPDKThreadManager>();
    if (!thread_manager_->initialize()) {
        return false;
    }
    
    // Simple assignment: use available lcores in order
    unsigned lcore_id;
    unsigned lcore_count = 0;
    
    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        if (lcore_count == 0) {
            // First worker lcore for RX
            thread_manager_->assign_lcore(lcore_id, LcoreConfig::RX_LCORE, 0, 0);
        } else if (lcore_count == 1) {
            // Second for TX
            thread_manager_->assign_lcore(lcore_id, LcoreConfig::TX_LCORE, 0, 0);
        } else {
            // Rest for workers
            thread_manager_->assign_lcore(lcore_id, LcoreConfig::WORKER_LCORE);
        }
        lcore_count++;
        
        if (lcore_count >= 3) break;  // Use max 3 lcores for now
    }
    
    if (lcore_count < 2) {
        std::cerr << "Not enough lcores available (need at least 2)" << std::endl;
        return false;
    }
    
    return true;
}

int DPDKApplication::run() {
    // Launch all lcores
    if (!thread_manager_->launch_all()) {
        return -1;
    }
    
    // Main lcore can do control plane work or just wait
    std::cout << "Application running. Press Ctrl+C to stop." << std::endl;
    
    // Simple main loop
    while (true) {
        sleep(1);
        thread_manager_->print_stats();
        
        // Check for signal to quit (simplified)
        // In production, use a proper signal handler
    }
    
    return 0;
}

void DPDKApplication::cleanup() {
    if (thread_manager_) {
        thread_manager_->stop_all();
        thread_manager_.reset();
    }
    
    // Stop and close all ports
    uint16_t portid;
    RTE_ETH_FOREACH_DEV(portid) {
        rte_eth_dev_stop(portid);
        rte_eth_dev_close(portid);
    }
    
    // Free mbuf pool
    if (mbuf_pool_) {
        rte_mempool_free(mbuf_pool_);
        mbuf_pool_ = nullptr;
    }
}

// ============================================================================
// DPDKThreadHelpers Implementation
// ============================================================================

int DPDKThreadHelpers::get_lcore_for_role(LcoreConfig::LcoreRole role) {
    // Simple heuristic: use first available lcore for each role
    unsigned lcore_id;
    static unsigned next_worker_lcore = 0;
    
    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        if (role == LcoreConfig::RX_LCORE && next_worker_lcore == 0) {
            next_worker_lcore++;
            return lcore_id;
        } else if (role == LcoreConfig::TX_LCORE && next_worker_lcore == 1) {
            next_worker_lcore++;
            return lcore_id;
        } else if (role == LcoreConfig::WORKER_LCORE && next_worker_lcore >= 2) {
            next_worker_lcore++;
            return lcore_id;
        }
    }
    
    return -1;
}

bool DPDKThreadHelpers::is_lcore_available(unsigned lcore_id) {
    return rte_lcore_is_enabled(lcore_id) && 
           lcore_id != rte_get_main_lcore();
}

int DPDKThreadHelpers::get_numa_node(unsigned lcore_id) {
    return rte_lcore_to_socket_id(lcore_id);
}

void DPDKThreadHelpers::print_lcore_layout() {
    std::cout << "\n=== DPDK Lcore Layout ===" << std::endl;
    std::cout << "Main lcore: " << rte_get_main_lcore() << std::endl;
    std::cout << "Worker lcores: ";
    
    unsigned lcore_id;
    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        std::cout << lcore_id << " (NUMA " << get_numa_node(lcore_id) << ") ";
    }
    std::cout << std::endl;
}

} // namespace data_plane
} // namespace genie
