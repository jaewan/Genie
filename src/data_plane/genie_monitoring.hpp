/**
 * Comprehensive Monitoring System for Genie Data Plane
 * 
 * Provides detailed metrics collection, analysis, and reporting for:
 * - Packet processing statistics
 * - GPU memory usage and transfers
 * - Network performance metrics
 * - System resource utilization
 * - Error tracking and diagnostics
 */

#pragma once

#include <atomic>
#include <chrono>
#include <vector>
#include <map>
#include <deque>
#include <mutex>
#include <memory>
#include <string>
#include <rte_cycles.h>

namespace genie {
namespace data_plane {

/**
 * Time series data structure for metrics
 */
template<typename T>
class TimeSeries {
public:
    struct DataPoint {
        uint64_t timestamp;
        T value;
    };
    
    TimeSeries(size_t max_points = 1000) : max_points_(max_points) {}
    
    void add(T value) {
        uint64_t now = rte_rdtsc();
        std::lock_guard<std::mutex> lock(mutex_);
        
        data_.push_back({now, value});
        
        if (data_.size() > max_points_) {
            data_.pop_front();
        }
    }
    
    std::vector<DataPoint> get_recent(size_t count) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t start = data_.size() > count ? data_.size() - count : 0;
        return std::vector<DataPoint>(data_.begin() + start, data_.end());
    }
    
    T get_average(size_t last_n = 0) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (data_.empty()) return T{};
        
        size_t count = last_n > 0 ? std::min(last_n, data_.size()) : data_.size();
        size_t start = data_.size() - count;
        
        T sum = T{};
        for (size_t i = start; i < data_.size(); i++) {
            sum += data_[i].value;
        }
        
        return sum / count;
    }
    
    T get_max() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (data_.empty()) return T{};
        
        T max_val = data_[0].value;
        for (const auto& dp : data_) {
            if (dp.value > max_val) max_val = dp.value;
        }
        
        return max_val;
    }
    
    T get_min() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (data_.empty()) return T{};
        
        T min_val = data_[0].value;
        for (const auto& dp : data_) {
            if (dp.value < min_val) min_val = dp.value;
        }
        
        return min_val;
    }
    
private:
    mutable std::mutex mutex_;
    std::deque<DataPoint> data_;
    size_t max_points_;
};

/**
 * Packet processing metrics
 */
struct PacketMetrics {
    // Basic counters
    std::atomic<uint64_t> rx_packets{0};
    std::atomic<uint64_t> tx_packets{0};
    std::atomic<uint64_t> rx_bytes{0};
    std::atomic<uint64_t> tx_bytes{0};
    std::atomic<uint64_t> rx_dropped{0};
    std::atomic<uint64_t> tx_dropped{0};
    
    // Error counters
    std::atomic<uint64_t> rx_errors{0};
    std::atomic<uint64_t> tx_errors{0};
    std::atomic<uint64_t> checksum_errors{0};
    std::atomic<uint64_t> fragmentation_errors{0};
    
    // Performance metrics
    std::atomic<uint64_t> processing_cycles{0};
    std::atomic<uint64_t> batch_count{0};
    TimeSeries<double> pps_history;  // Packets per second
    TimeSeries<double> bps_history;  // Bytes per second
    TimeSeries<double> latency_history;  // Processing latency
    
    void reset() {
        rx_packets = 0;
        tx_packets = 0;
        rx_bytes = 0;
        tx_bytes = 0;
        rx_dropped = 0;
        tx_dropped = 0;
        rx_errors = 0;
        tx_errors = 0;
        checksum_errors = 0;
        fragmentation_errors = 0;
        processing_cycles = 0;
        batch_count = 0;
    }
};

/**
 * GPU metrics
 */
struct GPUMetrics {
    // Memory usage
    std::atomic<uint64_t> memory_allocated{0};
    std::atomic<uint64_t> memory_used{0};
    std::atomic<uint64_t> memory_peak{0};
    
    // Transfer statistics
    std::atomic<uint64_t> dma_transfers{0};
    std::atomic<uint64_t> dma_bytes{0};
    std::atomic<uint64_t> dma_errors{0};
    
    // Registration stats
    std::atomic<uint64_t> registrations{0};
    std::atomic<uint64_t> deregistrations{0};
    std::atomic<uint64_t> registration_failures{0};
    
    // Performance
    TimeSeries<double> bandwidth_history;  // GB/s
    TimeSeries<double> utilization_history;  // Percentage
    
    void reset() {
        memory_allocated = 0;
        memory_used = 0;
        memory_peak = 0;
        dma_transfers = 0;
        dma_bytes = 0;
        dma_errors = 0;
        registrations = 0;
        deregistrations = 0;
        registration_failures = 0;
    }
};

/**
 * Transfer metrics
 */
struct TransferMetrics {
    // Transfer counters
    std::atomic<uint64_t> transfers_initiated{0};
    std::atomic<uint64_t> transfers_completed{0};
    std::atomic<uint64_t> transfers_failed{0};
    std::atomic<uint64_t> transfers_cancelled{0};
    
    // Data volume
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> bytes_received{0};
    
    // Reliability metrics
    std::atomic<uint64_t> retransmissions{0};
    std::atomic<uint64_t> acks_sent{0};
    std::atomic<uint64_t> acks_received{0};
    std::atomic<uint64_t> nacks_sent{0};
    std::atomic<uint64_t> nacks_received{0};
    std::atomic<uint64_t> timeouts{0};
    
    // Flow control
    std::atomic<uint64_t> flow_control_events{0};
    std::atomic<uint64_t> congestion_events{0};
    
    // Timing
    TimeSeries<double> completion_time_history;  // Seconds
    TimeSeries<double> throughput_history;  // GB/s
    
    void reset() {
        transfers_initiated = 0;
        transfers_completed = 0;
        transfers_failed = 0;
        transfers_cancelled = 0;
        bytes_sent = 0;
        bytes_received = 0;
        retransmissions = 0;
        acks_sent = 0;
        acks_received = 0;
        nacks_sent = 0;
        nacks_received = 0;
        timeouts = 0;
        flow_control_events = 0;
        congestion_events = 0;
    }
};

/**
 * System resource metrics
 */
struct SystemMetrics {
    // CPU metrics
    std::atomic<uint64_t> cpu_cycles{0};
    TimeSeries<double> cpu_utilization;  // Per-core utilization
    TimeSeries<double> cpu_frequency;    // MHz
    
    // Memory metrics
    std::atomic<uint64_t> memory_used{0};
    std::atomic<uint64_t> memory_available{0};
    std::atomic<uint64_t> hugepages_used{0};
    std::atomic<uint64_t> hugepages_free{0};
    
    // Cache metrics
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    TimeSeries<double> cache_hit_rate;
    
    // NUMA metrics
    std::map<int, std::atomic<uint64_t>> numa_allocations;
    std::map<int, std::atomic<uint64_t>> numa_accesses;
    
    void reset() {
        cpu_cycles = 0;
        memory_used = 0;
        memory_available = 0;
        hugepages_used = 0;
        hugepages_free = 0;
        cache_hits = 0;
        cache_misses = 0;
        
        for (auto& kv : numa_allocations) {
            kv.second = 0;
        }
        for (auto& kv : numa_accesses) {
            kv.second = 0;
        }
    }
};

/**
 * Connection metrics
 */
struct ConnectionMetrics {
    std::atomic<uint64_t> active_connections{0};
    std::atomic<uint64_t> total_connections{0};
    std::atomic<uint64_t> connection_failures{0};
    std::atomic<uint64_t> disconnections{0};
    
    // Per-connection stats
    struct ConnectionStats {
        std::string remote_node;
        uint64_t packets_sent;
        uint64_t packets_received;
        uint64_t bytes_sent;
        uint64_t bytes_received;
        double rtt_ms;
        double bandwidth_mbps;
        std::chrono::steady_clock::time_point established;
    };
    
    std::map<std::string, ConnectionStats> per_connection_stats;
    std::mutex stats_mutex;
    
    void reset() {
        active_connections = 0;
        total_connections = 0;
        connection_failures = 0;
        disconnections = 0;
        
        std::lock_guard<std::mutex> lock(stats_mutex);
        per_connection_stats.clear();
    }
};

/**
 * Monitoring aggregator
 */
class MonitoringSystem {
public:
    MonitoringSystem();
    ~MonitoringSystem();
    
    // Initialize monitoring
    bool initialize();
    
    // Start/stop monitoring
    void start();
    void stop();
    
    // Get metric instances
    PacketMetrics& packet_metrics() { return packet_metrics_; }
    GPUMetrics& gpu_metrics() { return gpu_metrics_; }
    TransferMetrics& transfer_metrics() { return transfer_metrics_; }
    SystemMetrics& system_metrics() { return system_metrics_; }
    ConnectionMetrics& connection_metrics() { return connection_metrics_; }
    
    // Update metrics (called from data path)
    void record_packet_rx(uint16_t nb_pkts, uint64_t bytes, uint64_t cycles);
    void record_packet_tx(uint16_t nb_pkts, uint64_t bytes, uint64_t cycles);
    void record_packet_drop(uint16_t nb_dropped, bool is_rx);
    void record_packet_error(uint16_t nb_errors, bool is_rx);
    
    void record_gpu_allocation(size_t bytes);
    void record_gpu_deallocation(size_t bytes);
    void record_gpu_transfer(size_t bytes, bool success);
    
    void record_transfer_start(const std::string& transfer_id);
    void record_transfer_complete(const std::string& transfer_id, size_t bytes, double duration_s);
    void record_transfer_failure(const std::string& transfer_id, const std::string& reason);
    
    void record_connection_established(const std::string& node_id);
    void record_connection_closed(const std::string& node_id);
    
    // Periodic updates
    void update_rates();  // Calculate rates (pps, bps, etc.)
    void update_system_metrics();  // Update CPU, memory stats
    
    // Reporting
    struct Report {
        double duration_seconds;
        
        // Packet stats
        uint64_t total_packets;
        uint64_t total_bytes;
        double avg_pps;
        double avg_bps;
        double packet_loss_rate;
        
        // Transfer stats
        uint64_t total_transfers;
        double transfer_success_rate;
        double avg_transfer_time;
        double avg_throughput;
        
        // System stats
        double avg_cpu_utilization;
        double memory_utilization;
        double cache_hit_rate;
        
        // GPU stats
        uint64_t gpu_memory_used;
        double gpu_bandwidth_utilization;
    };
    
    Report generate_report() const;
    void print_report() const;
    void export_json(const std::string& filename) const;
    void export_csv(const std::string& filename) const;
    
    // Real-time monitoring
    void enable_realtime_display(bool enable);
    void set_update_interval(std::chrono::milliseconds interval);
    
    // Alerting
    struct Alert {
        enum Level { INFO, WARNING, ERROR, CRITICAL };
        Level level;
        std::string message;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    using AlertCallback = std::function<void(const Alert&)>;
    void register_alert_callback(AlertCallback callback);
    
    // Thresholds for alerts
    struct Thresholds {
        double max_packet_loss_rate = 0.01;  // 1%
        double max_cpu_utilization = 0.90;   // 90%
        double min_cache_hit_rate = 0.80;    // 80%
        uint64_t max_memory_usage = 8ULL * 1024 * 1024 * 1024;  // 8GB
        double max_latency_ms = 10.0;
    };
    
    void set_thresholds(const Thresholds& thresholds);
    void check_thresholds();
    
private:
    // Metrics instances
    PacketMetrics packet_metrics_;
    GPUMetrics gpu_metrics_;
    TransferMetrics transfer_metrics_;
    SystemMetrics system_metrics_;
    ConnectionMetrics connection_metrics_;
    
    // Monitoring state
    bool running_;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_update_;
    std::chrono::milliseconds update_interval_;
    
    // Real-time display
    bool realtime_display_;
    std::thread display_thread_;
    
    // Alerting
    Thresholds thresholds_;
    std::vector<AlertCallback> alert_callbacks_;
    std::deque<Alert> alert_history_;
    static constexpr size_t MAX_ALERT_HISTORY = 100;
    
    // Helper functions
    void display_loop();
    void trigger_alert(Alert::Level level, const std::string& message);
    double calculate_rate(uint64_t count, double duration_s) const;
    
    // System metric collection
    double get_cpu_utilization() const;
    uint64_t get_memory_usage() const;
    uint64_t get_available_memory() const;
};

/**
 * Global monitoring instance accessor
 */
MonitoringSystem& get_monitoring_system();

} // namespace data_plane
} // namespace genie
