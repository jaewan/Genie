/**
 * Factory for creating the appropriate data plane implementation
 */

#include "genie_data_plane.hpp"
#include "genie_data_plane_enhanced.cpp"  // Include enhanced implementation
#include <iostream>
#include <nlohmann/json.hpp>

namespace genie {
namespace data_plane {

/**
 * Create data plane based on configuration
 * 
 * If use_dpdk_libs is true and DPDK libraries are available,
 * creates the enhanced version. Otherwise, creates the basic version.
 */
std::unique_ptr<GenieDataPlane> create_data_plane(const DataPlaneConfig& config) {
    if (config.use_dpdk_libs) {
        // Try to create enhanced version
        try {
            auto enhanced = std::make_unique<EnhancedDataPlane>(config);
            std::cout << "Creating enhanced data plane with DPDK libraries" << std::endl;
            return enhanced;
        } catch (const std::exception& e) {
            std::cerr << "Failed to create enhanced data plane: " << e.what() << std::endl;
            std::cerr << "Falling back to basic implementation" << std::endl;
        }
    }
    
    // Create basic version
    std::cout << "Creating basic data plane" << std::endl;
    return std::make_unique<GenieDataPlane>(config);
}

} // namespace data_plane
} // namespace genie

// Update C interface to use factory
extern "C" {

void* genie_data_plane_create_enhanced(const char* config_json) {
    try {
        nlohmann::json config = nlohmann::json::parse(config_json);
        
        genie::data_plane::DataPlaneConfig dp_config;
        
        // Parse configuration (same as before)
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
        if (config.contains("mtu")) {
            dp_config.mtu = config["mtu"].get<uint16_t>();
        }
        
        // Enhanced features
        if (config.contains("use_dpdk_libs")) {
            dp_config.use_dpdk_libs = config["use_dpdk_libs"].get<bool>();
        }
        if (config.contains("max_connections")) {
            dp_config.max_connections = config["max_connections"].get<uint32_t>();
        }
        
        // Create using factory
        auto data_plane = genie::data_plane::create_data_plane(dp_config);
        return data_plane.release();  // Transfer ownership to C
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to create data plane: " << e.what() << std::endl;
        return nullptr;
    }
}

} // extern "C"
