/**
 * Monitoring System Implementation
 */

#include "genie_monitoring.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <thread>
#include <cmath>
#include <rte_cycles.h>
#include <sys/sysinfo.h>

namespace genie {
namespace data_plane {

// Global monitoring instance
static std::unique_ptr<MonitoringSystem> g_monitoring_system;

MonitoringSystem& get_monitoring_system() {
    if (!g_monitoring_system) {
        g_monitoring_system = std::make_unique<MonitoringSystem>();
        g_monitoring_system->initialize();
    }
    return *g_monitoring_system;
}

// ============================================================================
// MonitoringSystem Implementation
// ============================================================================

MonitoringSystem::MonitoringSystem() 
    : running_(false),
      update_interval_(std::chrono::milliseconds(1000)),
      realtime_display_(false) {
}

MonitoringSystem::~MonitoringSystem() {
    stop();
}

bool MonitoringSystem::initialize() {
    std::cout << "Initializing monitoring system..." << std::endl;
    
    // Initialize NUMA node tracking
    for (int i = 0; i < 4; i++) {  // Assume max 4 NUMA nodes
        system_metrics_.numa_allocations[i] = 0;
        system_metrics_.numa_accesses[i] = 0;
    }
    
    start_time_ = std::chrono::steady_clock::now();
    last_update_ = start_time_;
    
    std::cout << "Monitoring system initialized" << std::endl;
    
    return true;
}

void MonitoringSystem::start() {
    if (running_) {
        return;
    }
    
    running_ = true;
    start_time_ = std::chrono::steady_clock::now();
    
    if (realtime_display_) {
        display_thread_ = std::thread(&MonitoringSystem::display_loop, this);
    }
    
    std::cout << "Monitoring started" << std::endl;
}

void MonitoringSystem::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    if (display_thread_.joinable()) {
        display_thread_.join();
    }
    
    std::cout << "Monitoring stopped" << std::endl;
}

void MonitoringSystem::record_packet_rx(uint16_t nb_pkts, uint64_t bytes, uint64_t cycles) {
    packet_metrics_.rx_packets += nb_pkts;
    packet_metrics_.rx_bytes += bytes;
    packet_metrics_.processing_cycles += cycles;
    packet_metrics_.batch_count++;
}

void MonitoringSystem::record_packet_tx(uint16_t nb_pkts, uint64_t bytes, uint64_t cycles) {
    packet_metrics_.tx_packets += nb_pkts;
    packet_metrics_.tx_bytes += bytes;
    packet_metrics_.processing_cycles += cycles;
}

void MonitoringSystem::record_packet_drop(uint16_t nb_dropped, bool is_rx) {
    if (is_rx) {
        packet_metrics_.rx_dropped += nb_dropped;
    } else {
        packet_metrics_.tx_dropped += nb_dropped;
    }
}

void MonitoringSystem::record_packet_error(uint16_t nb_errors, bool is_rx) {
    if (is_rx) {
        packet_metrics_.rx_errors += nb_errors;
    } else {
        packet_metrics_.tx_errors += nb_errors;
    }
}

void MonitoringSystem::record_gpu_allocation(size_t bytes) {
    gpu_metrics_.memory_allocated += bytes;
    gpu_metrics_.memory_used += bytes;
    
    uint64_t current = gpu_metrics_.memory_used.load();
    uint64_t peak = gpu_metrics_.memory_peak.load();
    
    while (current > peak && 
           !gpu_metrics_.memory_peak.compare_exchange_weak(peak, current)) {
        // Update peak if current is higher
    }
    
    gpu_metrics_.registrations++;
}

void MonitoringSystem::record_gpu_deallocation(size_t bytes) {
    gpu_metrics_.memory_used -= bytes;
    gpu_metrics_.deregistrations++;
}

void MonitoringSystem::record_gpu_transfer(size_t bytes, bool success) {
    gpu_metrics_.dma_transfers++;
    gpu_metrics_.dma_bytes += bytes;
    
    if (!success) {
        gpu_metrics_.dma_errors++;
    }
    
    // Calculate bandwidth (simplified)
    double bandwidth_gbps = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    gpu_metrics_.bandwidth_history.add(bandwidth_gbps);
}

void MonitoringSystem::record_transfer_start(const std::string& transfer_id) {
    transfer_metrics_.transfers_initiated++;
}

void MonitoringSystem::record_transfer_complete(const std::string& transfer_id, 
                                               size_t bytes, double duration_s) {
    transfer_metrics_.transfers_completed++;
    transfer_metrics_.bytes_sent += bytes;
    
    transfer_metrics_.completion_time_history.add(duration_s);
    
    if (duration_s > 0) {
        double throughput_gbps = static_cast<double>(bytes) / 
                                (duration_s * 1024.0 * 1024.0 * 1024.0);
        transfer_metrics_.throughput_history.add(throughput_gbps);
    }
}

void MonitoringSystem::record_transfer_failure(const std::string& transfer_id, 
                                              const std::string& reason) {
    transfer_metrics_.transfers_failed++;
    
    // Trigger alert for transfer failure
    trigger_alert(Alert::WARNING, "Transfer " + transfer_id + " failed: " + reason);
}

void MonitoringSystem::record_connection_established(const std::string& node_id) {
    connection_metrics_.active_connections++;
    connection_metrics_.total_connections++;
    
    std::lock_guard<std::mutex> lock(connection_metrics_.stats_mutex);
    connection_metrics_.per_connection_stats[node_id] = {
        .remote_node = node_id,
        .packets_sent = 0,
        .packets_received = 0,
        .bytes_sent = 0,
        .bytes_received = 0,
        .rtt_ms = 0.0,
        .bandwidth_mbps = 0.0,
        .established = std::chrono::steady_clock::now()
    };
}

void MonitoringSystem::record_connection_closed(const std::string& node_id) {
    connection_metrics_.active_connections--;
    connection_metrics_.disconnections++;
    
    std::lock_guard<std::mutex> lock(connection_metrics_.stats_mutex);
    connection_metrics_.per_connection_stats.erase(node_id);
}

void MonitoringSystem::update_rates() {
    auto now = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double>(now - last_update_).count();
    
    if (duration <= 0) {
        return;
    }
    
    // Calculate packet rates
    static uint64_t last_rx_packets = 0;
    static uint64_t last_tx_packets = 0;
    static uint64_t last_rx_bytes = 0;
    static uint64_t last_tx_bytes = 0;
    
    uint64_t rx_packets = packet_metrics_.rx_packets.load();
    uint64_t tx_packets = packet_metrics_.tx_packets.load();
    uint64_t rx_bytes = packet_metrics_.rx_bytes.load();
    uint64_t tx_bytes = packet_metrics_.tx_bytes.load();
    
    double rx_pps = (rx_packets - last_rx_packets) / duration;
    double tx_pps = (tx_packets - last_tx_packets) / duration;
    double rx_bps = (rx_bytes - last_rx_bytes) * 8 / duration;
    double tx_bps = (tx_bytes - last_tx_bytes) * 8 / duration;
    
    packet_metrics_.pps_history.add(rx_pps + tx_pps);
    packet_metrics_.bps_history.add(rx_bps + tx_bps);
    
    // Calculate latency
    uint64_t cycles = packet_metrics_.processing_cycles.load();
    uint64_t packets = rx_packets + tx_packets;
    
    if (packets > 0) {
        uint64_t hz = rte_get_tsc_hz();
        double avg_cycles = static_cast<double>(cycles) / packets;
        double latency_ns = (avg_cycles / hz) * 1e9;
        packet_metrics_.latency_history.add(latency_ns);
    }
    
    // Update system metrics
    update_system_metrics();
    
    // Check thresholds
    check_thresholds();
    
    // Update counters
    last_rx_packets = rx_packets;
    last_tx_packets = tx_packets;
    last_rx_bytes = rx_bytes;
    last_tx_bytes = tx_bytes;
    last_update_ = now;
}

void MonitoringSystem::update_system_metrics() {
    // CPU utilization
    double cpu_util = get_cpu_utilization();
    system_metrics_.cpu_utilization.add(cpu_util);
    
    // Memory usage
    uint64_t mem_used = get_memory_usage();
    uint64_t mem_avail = get_available_memory();
    
    system_metrics_.memory_used = mem_used;
    system_metrics_.memory_available = mem_avail;
    
    // Cache hit rate
    uint64_t hits = system_metrics_.cache_hits.load();
    uint64_t misses = system_metrics_.cache_misses.load();
    
    if (hits + misses > 0) {
        double hit_rate = static_cast<double>(hits) / (hits + misses);
        system_metrics_.cache_hit_rate.add(hit_rate);
    }
}

MonitoringSystem::Report MonitoringSystem::generate_report() const {
    Report report = {};
    
    auto now = std::chrono::steady_clock::now();
    report.duration_seconds = std::chrono::duration<double>(now - start_time_).count();
    
    // Packet statistics
    report.total_packets = packet_metrics_.rx_packets + packet_metrics_.tx_packets;
    report.total_bytes = packet_metrics_.rx_bytes + packet_metrics_.tx_bytes;
    report.avg_pps = packet_metrics_.pps_history.get_average();
    report.avg_bps = packet_metrics_.bps_history.get_average();
    
    uint64_t total_rx = packet_metrics_.rx_packets + packet_metrics_.rx_dropped;
    if (total_rx > 0) {
        report.packet_loss_rate = static_cast<double>(packet_metrics_.rx_dropped) / total_rx;
    }
    
    // Transfer statistics
    report.total_transfers = transfer_metrics_.transfers_completed + 
                           transfer_metrics_.transfers_failed;
    
    if (report.total_transfers > 0) {
        report.transfer_success_rate = static_cast<double>(transfer_metrics_.transfers_completed) / 
                                      report.total_transfers;
    }
    
    report.avg_transfer_time = transfer_metrics_.completion_time_history.get_average();
    report.avg_throughput = transfer_metrics_.throughput_history.get_average();
    
    // System statistics
    report.avg_cpu_utilization = system_metrics_.cpu_utilization.get_average();
    
    if (system_metrics_.memory_available > 0) {
        report.memory_utilization = static_cast<double>(system_metrics_.memory_used) / 
                                  (system_metrics_.memory_used + system_metrics_.memory_available);
    }
    
    report.cache_hit_rate = system_metrics_.cache_hit_rate.get_average();
    
    // GPU statistics
    report.gpu_memory_used = gpu_metrics_.memory_used;
    report.gpu_bandwidth_utilization = gpu_metrics_.bandwidth_history.get_average();
    
    return report;
}

void MonitoringSystem::print_report() const {
    Report report = generate_report();
    
    std::cout << "\n";
    std::cout << "================== MONITORING REPORT ==================\n";
    std::cout << "Duration: " << std::fixed << std::setprecision(2) 
              << report.duration_seconds << " seconds\n";
    std::cout << "\n";
    
    std::cout << "PACKET STATISTICS:\n";
    std::cout << "  Total Packets:    " << report.total_packets << "\n";
    std::cout << "  Total Bytes:      " << report.total_bytes << "\n";
    std::cout << "  Average PPS:      " << std::scientific << report.avg_pps << "\n";
    std::cout << "  Average BPS:      " << report.avg_bps << "\n";
    std::cout << "  Packet Loss Rate: " << std::fixed << std::setprecision(4) 
              << report.packet_loss_rate * 100 << "%\n";
    std::cout << "\n";
    
    std::cout << "TRANSFER STATISTICS:\n";
    std::cout << "  Total Transfers:     " << report.total_transfers << "\n";
    std::cout << "  Success Rate:        " << std::fixed << std::setprecision(2) 
              << report.transfer_success_rate * 100 << "%\n";
    std::cout << "  Avg Transfer Time:   " << report.avg_transfer_time << " s\n";
    std::cout << "  Avg Throughput:      " << report.avg_throughput << " GB/s\n";
    std::cout << "\n";
    
    std::cout << "SYSTEM STATISTICS:\n";
    std::cout << "  CPU Utilization:     " << std::fixed << std::setprecision(2) 
              << report.avg_cpu_utilization * 100 << "%\n";
    std::cout << "  Memory Utilization:  " << report.memory_utilization * 100 << "%\n";
    std::cout << "  Cache Hit Rate:      " << report.cache_hit_rate * 100 << "%\n";
    std::cout << "\n";
    
    std::cout << "GPU STATISTICS:\n";
    std::cout << "  Memory Used:         " << report.gpu_memory_used / (1024*1024) << " MB\n";
    std::cout << "  Bandwidth Usage:     " << report.gpu_bandwidth_utilization << " GB/s\n";
    std::cout << "\n";
    
    std::cout << "CONNECTION STATISTICS:\n";
    std::cout << "  Active Connections:  " << connection_metrics_.active_connections << "\n";
    std::cout << "  Total Connections:   " << connection_metrics_.total_connections << "\n";
    std::cout << "========================================================\n";
}

void MonitoringSystem::export_json(const std::string& filename) const {
    Report report = generate_report();
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    file << "{\n";
    file << "  \"duration_seconds\": " << report.duration_seconds << ",\n";
    file << "  \"packet_stats\": {\n";
    file << "    \"total_packets\": " << report.total_packets << ",\n";
    file << "    \"total_bytes\": " << report.total_bytes << ",\n";
    file << "    \"avg_pps\": " << report.avg_pps << ",\n";
    file << "    \"avg_bps\": " << report.avg_bps << ",\n";
    file << "    \"packet_loss_rate\": " << report.packet_loss_rate << "\n";
    file << "  },\n";
    file << "  \"transfer_stats\": {\n";
    file << "    \"total_transfers\": " << report.total_transfers << ",\n";
    file << "    \"success_rate\": " << report.transfer_success_rate << ",\n";
    file << "    \"avg_transfer_time\": " << report.avg_transfer_time << ",\n";
    file << "    \"avg_throughput\": " << report.avg_throughput << "\n";
    file << "  },\n";
    file << "  \"system_stats\": {\n";
    file << "    \"cpu_utilization\": " << report.avg_cpu_utilization << ",\n";
    file << "    \"memory_utilization\": " << report.memory_utilization << ",\n";
    file << "    \"cache_hit_rate\": " << report.cache_hit_rate << "\n";
    file << "  },\n";
    file << "  \"gpu_stats\": {\n";
    file << "    \"memory_used\": " << report.gpu_memory_used << ",\n";
    file << "    \"bandwidth_utilization\": " << report.gpu_bandwidth_utilization << "\n";
    file << "  }\n";
    file << "}\n";
    
    file.close();
    std::cout << "Report exported to " << filename << std::endl;
}

void MonitoringSystem::export_csv(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "timestamp,rx_packets,tx_packets,rx_bytes,tx_bytes,pps,bps,cpu_util,mem_util\n";
    
    // Get recent data points
    auto pps_data = packet_metrics_.pps_history.get_recent(100);
    auto bps_data = packet_metrics_.bps_history.get_recent(100);
    auto cpu_data = system_metrics_.cpu_utilization.get_recent(100);
    
    // Write data rows
    size_t max_points = std::max({pps_data.size(), bps_data.size(), cpu_data.size()});
    
    for (size_t i = 0; i < max_points; i++) {
        file << i << ",";
        file << packet_metrics_.rx_packets << ",";
        file << packet_metrics_.tx_packets << ",";
        file << packet_metrics_.rx_bytes << ",";
        file << packet_metrics_.tx_bytes << ",";
        
        if (i < pps_data.size()) {
            file << pps_data[i].value << ",";
        } else {
            file << "0,";
        }
        
        if (i < bps_data.size()) {
            file << bps_data[i].value << ",";
        } else {
            file << "0,";
        }
        
        if (i < cpu_data.size()) {
            file << cpu_data[i].value << ",";
        } else {
            file << "0,";
        }
        
        file << system_metrics_.memory_used << "\n";
    }
    
    file.close();
    std::cout << "Data exported to " << filename << std::endl;
}

void MonitoringSystem::enable_realtime_display(bool enable) {
    if (realtime_display_ == enable) {
        return;
    }
    
    realtime_display_ = enable;
    
    if (enable && running_) {
        display_thread_ = std::thread(&MonitoringSystem::display_loop, this);
    } else if (!enable && display_thread_.joinable()) {
        display_thread_.join();
    }
}

void MonitoringSystem::set_update_interval(std::chrono::milliseconds interval) {
    update_interval_ = interval;
}

void MonitoringSystem::register_alert_callback(AlertCallback callback) {
    alert_callbacks_.push_back(callback);
}

void MonitoringSystem::set_thresholds(const Thresholds& thresholds) {
    thresholds_ = thresholds;
}

void MonitoringSystem::check_thresholds() {
    // Check packet loss
    uint64_t total_rx = packet_metrics_.rx_packets + packet_metrics_.rx_dropped;
    if (total_rx > 0) {
        double loss_rate = static_cast<double>(packet_metrics_.rx_dropped) / total_rx;
        if (loss_rate > thresholds_.max_packet_loss_rate) {
            trigger_alert(Alert::WARNING, 
                         "High packet loss rate: " + std::to_string(loss_rate * 100) + "%");
        }
    }
    
    // Check CPU utilization
    double cpu_util = system_metrics_.cpu_utilization.get_average(10);
    if (cpu_util > thresholds_.max_cpu_utilization) {
        trigger_alert(Alert::WARNING, 
                     "High CPU utilization: " + std::to_string(cpu_util * 100) + "%");
    }
    
    // Check cache hit rate
    double cache_hit = system_metrics_.cache_hit_rate.get_average(10);
    if (cache_hit < thresholds_.min_cache_hit_rate) {
        trigger_alert(Alert::WARNING, 
                     "Low cache hit rate: " + std::to_string(cache_hit * 100) + "%");
    }
    
    // Check memory usage
    if (system_metrics_.memory_used > thresholds_.max_memory_usage) {
        trigger_alert(Alert::ERROR, 
                     "High memory usage: " + 
                     std::to_string(system_metrics_.memory_used / (1024*1024*1024)) + " GB");
    }
    
    // Check latency
    double avg_latency = packet_metrics_.latency_history.get_average(10) / 1e6;  // Convert to ms
    if (avg_latency > thresholds_.max_latency_ms) {
        trigger_alert(Alert::WARNING, 
                     "High packet latency: " + std::to_string(avg_latency) + " ms");
    }
}

void MonitoringSystem::display_loop() {
    while (running_ && realtime_display_) {
        // Clear screen (ANSI escape code)
        std::cout << "\033[2J\033[H";
        
        // Print real-time stats
        print_report();
        
        std::this_thread::sleep_for(update_interval_);
        update_rates();
    }
}

void MonitoringSystem::trigger_alert(Alert::Level level, const std::string& message) {
    Alert alert = {
        .level = level,
        .message = message,
        .timestamp = std::chrono::steady_clock::now()
    };
    
    // Store in history
    alert_history_.push_back(alert);
    if (alert_history_.size() > MAX_ALERT_HISTORY) {
        alert_history_.pop_front();
    }
    
    // Call callbacks
    for (const auto& callback : alert_callbacks_) {
        callback(alert);
    }
    
    // Print to console
    const char* level_str = "";
    switch (level) {
        case Alert::INFO: level_str = "INFO"; break;
        case Alert::WARNING: level_str = "WARNING"; break;
        case Alert::ERROR: level_str = "ERROR"; break;
        case Alert::CRITICAL: level_str = "CRITICAL"; break;
    }
    
    std::cerr << "[" << level_str << "] " << message << std::endl;
}

double MonitoringSystem::calculate_rate(uint64_t count, double duration_s) const {
    if (duration_s <= 0) {
        return 0;
    }
    return static_cast<double>(count) / duration_s;
}

double MonitoringSystem::get_cpu_utilization() const {
    // Simplified CPU utilization calculation
    // In production, would use /proc/stat or performance counters
    static uint64_t last_idle = 0;
    static uint64_t last_total = 0;
    
    std::ifstream file("/proc/stat");
    if (!file.is_open()) {
        return 0;
    }
    
    std::string cpu;
    uint64_t user, nice, system, idle, iowait, irq, softirq, steal;
    
    file >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
    file.close();
    
    uint64_t total = user + nice + system + idle + iowait + irq + softirq + steal;
    uint64_t total_diff = total - last_total;
    uint64_t idle_diff = idle - last_idle;
    
    double utilization = 0;
    if (total_diff > 0) {
        utilization = 1.0 - (static_cast<double>(idle_diff) / total_diff);
    }
    
    last_total = total;
    last_idle = idle;
    
    return utilization;
}

uint64_t MonitoringSystem::get_memory_usage() const {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return (info.totalram - info.freeram) * info.mem_unit;
    }
    return 0;
}

uint64_t MonitoringSystem::get_available_memory() const {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.freeram * info.mem_unit;
    }
    return 0;
}

} // namespace data_plane
} // namespace genie














