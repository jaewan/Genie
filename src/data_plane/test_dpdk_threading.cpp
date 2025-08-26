/**
 * Test program for DPDK native threading model
 * 
 * Demonstrates the standard DPDK approach using lcores
 */

#include "genie_dpdk_thread_model.hpp"
#include <signal.h>
#include <iostream>

using namespace genie::data_plane;

static volatile bool force_quit = false;

static void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        std::cout << "\n\nSignal " << signum 
                  << " received, preparing to exit..." << std::endl;
        force_quit = true;
    }
}

int main(int argc, char* argv[]) {
    // Register signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "==================================" << std::endl;
    std::cout << "DPDK Native Threading Model Test" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Create DPDK application
    DPDKApplication app;
    
    // Initialize EAL
    std::cout << "\n1. Initializing DPDK EAL..." << std::endl;
    if (!app.init_eal(argc, argv)) {
        std::cerr << "Failed to initialize DPDK EAL" << std::endl;
        return -1;
    }
    
    // Print lcore layout
    DPDKThreadHelpers::print_lcore_layout();
    
    // Check if we have enough lcores
    if (rte_lcore_count() < 3) {
        std::cerr << "\nError: Need at least 3 lcores (1 main + 2 workers)" << std::endl;
        std::cerr << "Run with: " << argv[0] << " -l 0-2" << std::endl;
        return -1;
    }
    
    // Initialize ports (using 1 port for testing)
    std::cout << "\n2. Initializing ports..." << std::endl;
    uint16_t nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0) {
        std::cerr << "No Ethernet ports found" << std::endl;
        std::cerr << "Make sure DPDK-compatible NICs are bound" << std::endl;
        return -1;
    }
    
    std::cout << "Found " << nb_ports << " Ethernet port(s)" << std::endl;
    
    if (!app.init_ports(1)) {  // Use only first port
        std::cerr << "Failed to initialize ports" << std::endl;
        return -1;
    }
    
    // Setup lcore assignments
    std::cout << "\n3. Setting up lcore assignments..." << std::endl;
    if (!app.setup_lcores()) {
        std::cerr << "Failed to setup lcores" << std::endl;
        return -1;
    }
    
    // Create and test service cores (optional)
    std::cout << "\n4. Setting up service cores (optional)..." << std::endl;
    ServiceCoreManager service_mgr;
    
    // Register a stats service
    service_mgr.register_service("stats_service", 
                                 ServiceCoreManager::stats_service,
                                 nullptr);
    
    // If we have extra lcores, use one for services
    if (rte_lcore_count() > 4) {
        unsigned service_lcore = 3;  // Use lcore 3 for services
        service_mgr.map_service_to_lcore(0, service_lcore);
        service_mgr.start_service_cores();
        std::cout << "Service cores started" << std::endl;
    } else {
        std::cout << "Not enough lcores for service cores (skipping)" << std::endl;
    }
    
    // Run the application
    std::cout << "\n5. Running application..." << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "Press Ctrl+C to stop" << std::endl;
    std::cout << "================================\n" << std::endl;
    
    // Simple main loop - in production, app.run() would handle this
    DPDKThreadManager thread_mgr;
    thread_mgr.initialize();
    
    // Assign lcores based on what's available
    unsigned lcore_id;
    int lcore_idx = 0;
    
    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        if (lcore_idx == 0) {
            thread_mgr.assign_lcore(lcore_id, LcoreConfig::RX_LCORE, 0, 0);
            std::cout << "Assigned lcore " << lcore_id << " for RX" << std::endl;
        } else if (lcore_idx == 1) {
            thread_mgr.assign_lcore(lcore_id, LcoreConfig::TX_LCORE, 0, 0);
            std::cout << "Assigned lcore " << lcore_id << " for TX" << std::endl;
        } else if (lcore_idx == 2) {
            thread_mgr.assign_lcore(lcore_id, LcoreConfig::WORKER_LCORE);
            std::cout << "Assigned lcore " << lcore_id << " for WORKER" << std::endl;
        }
        lcore_idx++;
        if (lcore_idx >= 3) break;
    }
    
    // Launch all lcores
    if (!thread_mgr.launch_all()) {
        std::cerr << "Failed to launch lcores" << std::endl;
        return -1;
    }
    
    // Main loop - print stats periodically
    uint64_t prev_tsc = rte_rdtsc();
    uint64_t timer_tsc = 0;
    uint64_t hz = rte_get_tsc_hz();
    
    while (!force_quit) {
        uint64_t cur_tsc = rte_rdtsc();
        uint64_t diff_tsc = cur_tsc - prev_tsc;
        timer_tsc += diff_tsc;
        
        // Print stats every second
        if (timer_tsc >= hz) {
            thread_mgr.print_stats();
            timer_tsc = 0;
        }
        
        prev_tsc = cur_tsc;
        rte_delay_ms(100);  // Sleep 100ms
    }
    
    // Cleanup
    std::cout << "\n\nShutting down..." << std::endl;
    thread_mgr.stop_all();
    service_mgr.stop_service_cores();
    app.cleanup();
    
    std::cout << "Application terminated cleanly" << std::endl;
    
    return 0;
}
