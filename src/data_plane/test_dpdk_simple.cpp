/**
 * Simple DPDK Threading Demonstration
 * 
 * This demonstrates DPDK's native lcore management without requiring
 * actual network ports. It shows how to use DPDK's threading model
 * for high-performance parallel processing.
 */

#include <rte_eal.h>
#include <rte_lcore.h>
#include <rte_launch.h>
#include <rte_cycles.h>
#include <rte_ring.h>
#include <rte_malloc.h>
#include <iostream>
#include <atomic>
#include <chrono>
#include <iomanip>

// Global flag for clean shutdown
static std::atomic<bool> force_quit{false};

// Statistics per lcore
struct lcore_stats {
    uint64_t messages_processed;
    uint64_t cycles_busy;
    uint64_t cycles_idle;
};

static struct lcore_stats stats[RTE_MAX_LCORE];

// Inter-lcore communication ring
static struct rte_ring* message_ring;

/**
 * Producer lcore function
 * Generates messages and puts them in the ring
 */
static int producer_main(void* arg) {
    unsigned lcore_id = rte_lcore_id();
    std::cout << "Producer started on lcore " << lcore_id << std::endl;
    
    uint64_t counter = 0;
    
    while (!force_quit.load()) {
        uint64_t start_tsc = rte_rdtsc();
        
        // Generate some "work" (just incrementing counter)
        void* msg = reinterpret_cast<void*>(++counter);
        
        // Try to enqueue message
        if (rte_ring_enqueue(message_ring, msg) == 0) {
            stats[lcore_id].messages_processed++;
            stats[lcore_id].cycles_busy += rte_rdtsc() - start_tsc;
        } else {
            // Ring full, count as idle
            stats[lcore_id].cycles_idle += rte_rdtsc() - start_tsc;
            rte_pause(); // CPU pause to reduce power consumption
        }
    }
    
    std::cout << "Producer on lcore " << lcore_id << " stopping" << std::endl;
    return 0;
}

/**
 * Consumer lcore function
 * Processes messages from the ring
 */
static int consumer_main(void* arg) {
    unsigned lcore_id = rte_lcore_id();
    std::cout << "Consumer started on lcore " << lcore_id << std::endl;
    
    while (!force_quit.load()) {
        uint64_t start_tsc = rte_rdtsc();
        
        void* msg;
        if (rte_ring_dequeue(message_ring, &msg) == 0) {
            // Process message (simulate work)
            uint64_t value = reinterpret_cast<uint64_t>(msg);
            volatile uint64_t result = value * value; // Simple computation
            (void)result; // Suppress unused warning
            
            stats[lcore_id].messages_processed++;
            stats[lcore_id].cycles_busy += rte_rdtsc() - start_tsc;
        } else {
            // No messages, count as idle
            stats[lcore_id].cycles_idle += rte_rdtsc() - start_tsc;
            rte_pause();
        }
    }
    
    std::cout << "Consumer on lcore " << lcore_id << " stopping" << std::endl;
    return 0;
}

/**
 * Print statistics
 */
static void print_stats() {
    const uint64_t hz = rte_get_tsc_hz();
    
    std::cout << "\n=== DPDK Lcore Statistics ===" << std::endl;
    std::cout << std::setw(8) << "Lcore"
              << std::setw(15) << "Messages"
              << std::setw(15) << "Utilization %"
              << std::endl;
    std::cout << std::string(38, '-') << std::endl;
    
    unsigned lcore_id;
    RTE_LCORE_FOREACH(lcore_id) {
        if (stats[lcore_id].messages_processed > 0) {
            uint64_t total_cycles = stats[lcore_id].cycles_busy + 
                                   stats[lcore_id].cycles_idle;
            double utilization = 0.0;
            if (total_cycles > 0) {
                utilization = 100.0 * stats[lcore_id].cycles_busy / total_cycles;
            }
            
            std::cout << std::setw(8) << lcore_id
                      << std::setw(15) << stats[lcore_id].messages_processed
                      << std::setw(14) << std::fixed << std::setprecision(2) 
                      << utilization << "%"
                      << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Simple DPDK Threading Demonstration" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Initialize EAL
    int ret = rte_eal_init(argc, argv);
    if (ret < 0) {
        std::cerr << "Error with EAL initialization" << std::endl;
        return -1;
    }
    
    // Check available lcores
    unsigned nb_lcores = rte_lcore_count();
    if (nb_lcores < 2) {
        std::cerr << "Error: Need at least 2 lcores" << std::endl;
        std::cerr << "Run with: " << argv[0] << " -l 0-1" << std::endl;
        return -1;
    }
    
    std::cout << "DPDK initialized with " << nb_lcores << " lcores" << std::endl;
    
    // Create ring for inter-lcore communication
    message_ring = rte_ring_create("MSG_RING", 1024, rte_socket_id(),
                                   RING_F_SP_ENQ | RING_F_SC_DEQ);
    if (!message_ring) {
        std::cerr << "Failed to create ring" << std::endl;
        return -1;
    }
    
    std::cout << "Created message ring with 1024 slots\n" << std::endl;
    
    // Launch lcores
    unsigned lcore_id;
    bool first_worker = true;
    
    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        if (first_worker) {
            // First worker is producer
            std::cout << "Launching producer on lcore " << lcore_id << std::endl;
            rte_eal_remote_launch(producer_main, nullptr, lcore_id);
            first_worker = false;
        } else {
            // Rest are consumers
            std::cout << "Launching consumer on lcore " << lcore_id << std::endl;
            rte_eal_remote_launch(consumer_main, nullptr, lcore_id);
        }
    }
    
    std::cout << "\nRunning... Press Ctrl+C to stop\n" << std::endl;
    
    // Main lcore monitors and prints stats
    auto start_time = std::chrono::steady_clock::now();
    while (!force_quit.load()) {
        rte_delay_ms(1000); // Sleep 1 second
        
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>
                      (current_time - start_time).count();
        
        if (elapsed % 5 == 0) { // Print stats every 5 seconds
            print_stats();
        }
        
        // Simple way to stop after 30 seconds for demo
        if (elapsed >= 30) {
            std::cout << "\nDemo time limit reached (30 seconds)" << std::endl;
            force_quit = true;
        }
    }
    
    std::cout << "\nShutting down..." << std::endl;
    
    // Wait for all lcores to finish
    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        rte_eal_wait_lcore(lcore_id);
    }
    
    // Print final stats
    print_stats();
    
    // Cleanup
    rte_ring_free(message_ring);
    
    std::cout << "\nDemo completed successfully!" << std::endl;
    
    return 0;
}
