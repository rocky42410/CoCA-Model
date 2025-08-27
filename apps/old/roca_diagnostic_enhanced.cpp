// ============================================================================
// apps/roca_diagnostic_enhanced.cpp - Deep analysis of problematic features
// ============================================================================
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

#include "../src/io/binary_log.hpp"

using namespace roca;

// ============================================================================
// Feature mapping for Unitree Go2
// ============================================================================
struct FeatureInfo {
    size_t index;
    std::string name;
    std::string modality;
    std::string description;
    bool is_critical;  // Critical for basic operation
};

class FeatureMapper {
private:
    std::vector<FeatureInfo> feature_map;
    
public:
    FeatureMapper() {
        // Initialize feature map based on typical Go2 configuration
        // Joint states: 12 joints × 3 values (position, velocity, torque) = 36 features
        for (int joint = 0; joint < 12; joint++) {
            std::string joint_names[] = {"FR_hip", "FR_thigh", "FR_calf", 
                                        "FL_hip", "FL_thigh", "FL_calf",
                                        "RR_hip", "RR_thigh", "RR_calf",
                                        "RL_hip", "RL_thigh", "RL_calf"};
            
            feature_map.push_back({size_t(joint*3), joint_names[joint] + "_pos", 
                                 "motor", "Joint position (rad)", true});
            feature_map.push_back({size_t(joint*3+1), joint_names[joint] + "_vel", 
                                 "motor", "Joint velocity (rad/s)", true});
            feature_map.push_back({size_t(joint*3+2), joint_names[joint] + "_tau", 
                                 "motor", "Joint torque (Nm)", true});
        }
        
        // IMU: 36-41
        feature_map.push_back({36, "imu_roll", "imu", "Roll angle (rad)", true});
        feature_map.push_back({37, "imu_pitch", "imu", "Pitch angle (rad)", true});
        feature_map.push_back({38, "imu_yaw", "imu", "Yaw angle (rad)", false});
        feature_map.push_back({39, "imu_gyro_x", "imu", "Gyro X (rad/s)", true});
        feature_map.push_back({40, "imu_gyro_y", "imu", "Gyro Y (rad/s)", true});
        feature_map.push_back({41, "imu_gyro_z", "imu", "Gyro Z (rad/s)", true});
        
        // Accelerometer: 42-44
        feature_map.push_back({42, "imu_acc_x", "imu", "Accel X (m/s²)", true});
        feature_map.push_back({43, "imu_acc_y", "imu", "Accel Y (m/s²)", true});
        feature_map.push_back({44, "imu_acc_z", "imu", "Accel Z (m/s²)", true});
        
        // Foot force: 45-48
        feature_map.push_back({45, "foot_force_FR", "motor", "FR foot force (N)", false});
        feature_map.push_back({46, "foot_force_FL", "motor", "FL foot force (N)", false});
        feature_map.push_back({47, "foot_force_RR", "motor", "RR foot force (N)", false});
        feature_map.push_back({48, "foot_force_RL", "motor", "RL foot force (N)", false});
        
        // Battery: 49-51
        feature_map.push_back({49, "battery_voltage", "bms", "Battery voltage (V)", false});
        feature_map.push_back({50, "battery_current", "bms", "Battery current (A)", false});
        feature_map.push_back({51, "battery_soc", "bms", "Battery SOC (%)", false});
        
        // UWB position: 52-54
        feature_map.push_back({52, "uwb_x", "uwb", "UWB X position (m)", false});
        feature_map.push_back({53, "uwb_y", "uwb", "UWB Y position (m)", false});
        feature_map.push_back({54, "uwb_z", "uwb", "UWB Z position (m)", false});
        
        // Foot position (from kinematics): 55-66
        for (int foot = 0; foot < 4; foot++) {
            std::string foot_names[] = {"FR", "FL", "RR", "RL"};
            feature_map.push_back({size_t(55 + foot*3), foot_names[foot] + "_foot_x", 
                                 "kinematics", "Foot X position (m)", false});
            feature_map.push_back({size_t(56 + foot*3), foot_names[foot] + "_foot_y", 
                                 "kinematics", "Foot Y position (m)", false});
            feature_map.push_back({size_t(57 + foot*3), foot_names[foot] + "_foot_z", 
                                 "kinematics", "Foot Z position (m)", false});
        }
        
        // Body state: 67-75
        feature_map.push_back({67, "body_x", "state", "Body X position (m)", false});
        feature_map.push_back({68, "body_y", "state", "Body Y position (m)", false});
        feature_map.push_back({69, "body_z", "state", "Body height (m)", false});
        feature_map.push_back({70, "body_vx", "state", "Body X velocity (m/s)", false});
        feature_map.push_back({71, "body_vy", "state", "Body Y velocity (m/s)", false});
        feature_map.push_back({72, "body_vz", "state", "Body Z velocity (m/s)", false});
        feature_map.push_back({73, "body_roll_rate", "state", "Body roll rate (rad/s)", false});
        feature_map.push_back({74, "body_pitch_rate", "state", "Body pitch rate (rad/s)", false});
        feature_map.push_back({75, "body_yaw_rate", "state", "Body yaw rate (rad/s)", false});
        
        // Command/Reference: 76-83
        feature_map.push_back({76, "cmd_x_vel", "command", "Commanded X velocity", false});
        feature_map.push_back({77, "cmd_y_vel", "command", "Commanded Y velocity", false});
        feature_map.push_back({78, "cmd_yaw_rate", "command", "Commanded yaw rate", false});
        feature_map.push_back({79, "cmd_height", "command", "Commanded height", false});
        feature_map.push_back({80, "cmd_pitch", "command", "Commanded pitch", false});
        feature_map.push_back({81, "cmd_roll", "command", "Commanded roll", false});
        feature_map.push_back({82, "cmd_mode", "command", "Control mode", false});
        feature_map.push_back({83, "gait_type", "command", "Gait type", false});
        
        // Extended features: 84+ (THESE ARE THE PROBLEMATIC ONES)
        // These might be:
        // - Additional sensor data
        // - Computed features
        // - Reserved/unused slots
        // - Camera/vision features (if enabled)
        
        for (size_t i = 84; i < 256; i++) {
            feature_map.push_back({i, "extended_" + std::to_string(i), 
                                 "extended", "Extended/Reserved feature", false});
        }
    }
    
    FeatureInfo get_info(size_t index) const {
        if (index < feature_map.size()) {
            return feature_map[index];
        }
        return {index, "unknown_" + std::to_string(index), "unknown", "Unknown feature", false};
    }
    
    std::string get_modality(size_t index) const {
        return get_info(index).modality;
    }
    
    bool is_critical(size_t index) const {
        return get_info(index).is_critical;
    }
};

// ============================================================================
// Enhanced diagnostic analysis
// ============================================================================
class EnhancedDiagnostic {
private:
    struct DetailedStats {
        size_t index;
        std::string name;
        std::string modality;
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        float mean = 0.0f;
        float std = 0.0f;
        size_t nan_count = 0;
        size_t inf_count = 0;
        size_t zero_count = 0;
        size_t valid_count = 0;
        float nan_percentage = 0.0f;
        bool is_constant = false;
        bool is_valid = false;
        std::vector<float> sample_values;  // First 10 non-NaN values
    };
    
    std::vector<DetailedStats> stats;
    FeatureMapper mapper;
    size_t total_frames = 0;
    size_t num_features = 0;
    
    // Modality summaries
    std::map<std::string, std::vector<size_t>> modality_features;
    std::map<std::string, size_t> modality_nan_counts;
    std::map<std::string, size_t> modality_valid_counts;
    
public:
    void analyze_file(const std::string& path) {
        std::cout << "\n╔════════════════════════════════════════╗\n";
        std::cout << "║   Enhanced Feature Diagnostic Report   ║\n";
        std::cout << "╚════════════════════════════════════════╝\n\n";
        
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot open file " << path << "\n";
            return;
        }
        
        // Read header
        BinaryFileHeader header;
        if (!read_header(file, header)) {
            std::cerr << "Error: Invalid header\n";
            return;
        }
        
        num_features = header.feature_count;
        stats.resize(num_features);
        
        // Initialize stats with feature info
        for (size_t i = 0; i < num_features; ++i) {
            auto info = mapper.get_info(i);
            stats[i].index = i;
            stats[i].name = info.name;
            stats[i].modality = info.modality;
            
            // Group by modality
            modality_features[info.modality].push_back(i);
        }
        
        std::cout << "File: " << path << "\n";
        std::cout << "Features: " << num_features << "\n";
        std::cout << "Fusion rate: " << header.fusion_rate_hz << " Hz\n\n";
        
        // Collect all data
        std::vector<std::vector<float>> all_frames;
        AutoencoderFrame frame;
        
        while (read_frame(file, frame)) {
            total_frames++;
            std::vector<float> features(num_features);
            for (size_t i = 0; i < num_features; ++i) {
                features[i] = frame.features[i];
            }
            all_frames.push_back(features);
        }
        
        std::cout << "Total frames: " << total_frames << "\n";
        std::cout << "Duration: " << std::fixed << std::setprecision(1) 
                  << (total_frames / (float)header.fusion_rate_hz) << " seconds\n\n";
        
        // Analyze each feature
        analyze_features(all_frames);
        
        // Report findings
        report_modality_summary();
        report_critical_features();
        report_valid_features();
        report_problematic_features();
        generate_recommendations();
    }
    
private:
    void analyze_features(const std::vector<std::vector<float>>& frames) {
        std::cout << "Analyzing features...\n";
        
        for (size_t f = 0; f < num_features; ++f) {
            auto& s = stats[f];
            std::vector<float> valid_values;
            
            for (const auto& frame : frames) {
                float val = frame[f];
                
                if (std::isnan(val)) {
                    s.nan_count++;
                } else if (std::isinf(val)) {
                    s.inf_count++;
                } else {
                    valid_values.push_back(val);
                    s.valid_count++;
                    
                    // Update min/max
                    s.min_val = std::min(s.min_val, val);
                    s.max_val = std::max(s.max_val, val);
                    
                    // Collect samples
                    if (s.sample_values.size() < 10) {
                        s.sample_values.push_back(val);
                    }
                    
                    if (std::abs(val) < 1e-10f) {
                        s.zero_count++;
                    }
                }
            }
            
            // Calculate statistics
            s.nan_percentage = 100.0f * s.nan_count / frames.size();
            
            if (!valid_values.empty()) {
                // Mean
                s.mean = std::accumulate(valid_values.begin(), valid_values.end(), 0.0f) 
                        / valid_values.size();
                
                // Std
                float variance = 0.0f;
                for (float val : valid_values) {
                    float diff = val - s.mean;
                    variance += diff * diff;
                }
                s.std = std::sqrt(variance / valid_values.size());
                
                // Check if constant
                s.is_constant = (s.std < 1e-8f);
                
                // Mark as valid if >50% non-NaN and not infinite
                s.is_valid = (s.nan_percentage < 50.0f) && !std::isinf(s.mean) && !std::isnan(s.std);
            }
            
            // Update modality counts
            if (s.is_valid) {
                modality_valid_counts[s.modality]++;
            } else {
                modality_nan_counts[s.modality]++;
            }
        }
    }
    
    void report_modality_summary() {
        std::cout << "\n┌─────────────────────────────────────┐\n";
        std::cout << "│      MODALITY-LEVEL ANALYSIS       │\n";
        std::cout << "└─────────────────────────────────────┘\n\n";
        
        for (const auto& [modality, features] : modality_features) {
            size_t valid = modality_valid_counts[modality];
            size_t invalid = modality_nan_counts[modality];
            size_t total = features.size();
            
            std::cout << std::setw(12) << modality << ": ";
            
            if (valid == total) {
                std::cout << "✅ ALL VALID (" << total << " features)\n";
            } else if (valid == 0) {
                std::cout << "❌ ALL INVALID (" << total << " features)\n";
            } else {
                std::cout << "⚠️  " << valid << "/" << total << " valid (";
                std::cout << std::fixed << std::setprecision(1) 
                         << (100.0f * valid / total) << "%)\n";
            }
            
            // Show sample of problematic features in this modality
            if (valid < total) {
                std::cout << "     Problem features: ";
                int shown = 0;
                for (size_t idx : features) {
                    if (!stats[idx].is_valid && shown < 5) {
                        std::cout << idx << " ";
                        shown++;
                    }
                }
                if (invalid > 5) std::cout << "...";
                std::cout << "\n";
            }
        }
    }
    
    void report_critical_features() {
        std::cout << "\n┌─────────────────────────────────────┐\n";
        std::cout << "│    CRITICAL FEATURES STATUS        │\n";
        std::cout << "└─────────────────────────────────────┘\n\n";
        
        std::cout << "Critical features (required for basic operation):\n\n";
        
        for (size_t i = 0; i < num_features; ++i) {
            if (mapper.is_critical(i)) {
                auto& s = stats[i];
                std::cout << "  [" << std::setw(3) << i << "] " 
                         << std::setw(20) << s.name << ": ";
                
                if (s.is_valid) {
                    std::cout << "✅ OK";
                    if (s.is_constant) {
                        std::cout << " (constant: " << std::fixed << std::setprecision(3) 
                                 << s.mean << ")";
                    } else {
                        std::cout << " (mean: " << std::fixed << std::setprecision(3) 
                                 << s.mean << ", std: " << s.std << ")";
                    }
                } else if (s.nan_percentage > 99) {
                    std::cout << "❌ ALL NaN";
                } else if (std::isinf(s.mean)) {
                    std::cout << "❌ INFINITE VALUES";
                } else {
                    std::cout << "⚠️  " << std::fixed << std::setprecision(1) 
                             << s.nan_percentage << "% NaN";
                }
                std::cout << "\n";
            }
        }
    }
    
    void report_valid_features() {
        std::cout << "\n┌─────────────────────────────────────┐\n";
        std::cout << "│      VALID FEATURES SUMMARY        │\n";
        std::cout << "└─────────────────────────────────────┘\n\n";
        
        std::vector<size_t> valid_indices;
        std::vector<size_t> constant_indices;
        std::vector<size_t> variable_indices;
        
        for (size_t i = 0; i < num_features; ++i) {
            if (stats[i].is_valid) {
                valid_indices.push_back(i);
                if (stats[i].is_constant) {
                    constant_indices.push_back(i);
                } else {
                    variable_indices.push_back(i);
                }
            }
        }
        
        std::cout << "Total valid features: " << valid_indices.size() << "/" << num_features << "\n";
        std::cout << "  Variable: " << variable_indices.size() << "\n";
        std::cout << "  Constant: " << constant_indices.size() << "\n\n";
        
        if (!variable_indices.empty()) {
            std::cout << "Sample variable features (good for training):\n";
            for (size_t j = 0; j < std::min(size_t(10), variable_indices.size()); ++j) {
                size_t i = variable_indices[j];
                auto& s = stats[i];
                std::cout << "  [" << std::setw(3) << i << "] " << std::setw(20) << s.name
                         << " | mean: " << std::setw(8) << std::fixed << std::setprecision(3) << s.mean
                         << " | std: " << std::setw(8) << s.std
                         << " | range: [" << s.min_val << ", " << s.max_val << "]\n";
            }
        }
        
        // Save valid feature indices to file
        std::ofstream valid_file("valid_features.txt");
        valid_file << "# Valid feature indices for training\n";
        valid_file << "# Total: " << valid_indices.size() << "\n";
        for (size_t idx : valid_indices) {
            valid_file << idx << "\n";
        }
        valid_file.close();
        
        std::cout << "\n✅ Valid feature indices saved to: valid_features.txt\n";
    }
    
    void report_problematic_features() {
        std::cout << "\n┌─────────────────────────────────────┐\n";
        std::cout << "│    PROBLEMATIC FEATURES DETAIL     │\n";
        std::cout << "└─────────────────────────────────────┘\n\n";
        
        // Group problems by type
        std::vector<size_t> all_nan;
        std::vector<size_t> mostly_nan;
        std::vector<size_t> infinite;
        
        for (size_t i = 0; i < num_features; ++i) {
            auto& s = stats[i];
            if (s.nan_percentage > 99.9f) {
                all_nan.push_back(i);
            } else if (s.nan_percentage > 50.0f) {
                mostly_nan.push_back(i);
            } else if (std::isinf(s.mean) || s.inf_count > 0) {
                infinite.push_back(i);
            }
        }
        
        if (!all_nan.empty()) {
            std::cout << "Features that are ALWAYS NaN (" << all_nan.size() << "):\n";
            std::cout << "  Indices: ";
            for (size_t i = 0; i < std::min(size_t(20), all_nan.size()); ++i) {
                std::cout << all_nan[i] << " ";
            }
            if (all_nan.size() > 20) std::cout << "... (" << (all_nan.size() - 20) << " more)";
            std::cout << "\n";
            
            // Identify pattern
            if (all_nan.size() > 50 && all_nan.front() >= 84) {
                std::cout << "  ⚠️  Pattern: Extended features (84+) not populated\n";
                std::cout << "     These may be reserved/unused feature slots\n";
            }
        }
        
        if (!mostly_nan.empty()) {
            std::cout << "\nFeatures with >50% NaN (" << mostly_nan.size() << "):\n";
            for (size_t idx : mostly_nan) {
                auto& s = stats[idx];
                std::cout << "  [" << idx << "] " << s.name << ": " 
                         << std::fixed << std::setprecision(1) << s.nan_percentage << "% NaN\n";
            }
        }
        
        if (!infinite.empty()) {
            std::cout << "\nFeatures with infinite values (" << infinite.size() << "):\n";
            for (size_t idx : infinite) {
                auto& s = stats[idx];
                std::cout << "  [" << idx << "] " << s.name;
                if (!s.sample_values.empty()) {
                    std::cout << " | samples: ";
                    for (size_t j = 0; j < std::min(size_t(3), s.sample_values.size()); ++j) {
                        std::cout << s.sample_values[j] << " ";
                    }
                }
                std::cout << "\n";
            }
        }
    }
    
    void generate_recommendations() {
        std::cout << "\n╔════════════════════════════════════════╗\n";
        std::cout << "║         RECOMMENDATIONS                ║\n";
        std::cout << "╚════════════════════════════════════════╝\n\n";
        
        // Count valid features
        size_t valid_count = 0;
        for (const auto& s : stats) {
            if (s.is_valid) valid_count++;
        }
        
        if (valid_count < 50) {
            std::cout << "⚠️  CRITICAL: Only " << valid_count << " valid features!\n\n";
            std::cout << "Immediate actions:\n";
            std::cout << "1. Check your feature_logger configuration\n";
            std::cout << "2. Verify all ROS2/DDS topics are being subscribed\n";
            std::cout << "3. Ensure sensor drivers are running\n";
            std::cout << "4. Check feature fusion logic\n\n";
        } else {
            std::cout << "✅ " << valid_count << " valid features available for training\n\n";
        }
        
        std::cout << "Training approach:\n";
        std::cout << "1. Use ONLY the " << valid_count << " valid features\n";
        std::cout << "2. Modify your config to set D = " << valid_count << "\n";
        std::cout << "3. Create a feature mask from valid_features.txt\n";
        std::cout << "4. Use the idle-specialized implementation\n\n";
        
        std::cout << "Code modification needed:\n";
        std::cout << "```cpp\n";
        std::cout << "// In your training code:\n";
        std::cout << "std::vector<size_t> valid_indices = load_valid_indices(\"valid_features.txt\");\n";
        std::cout << "config.D = " << valid_count << ";  // Use only valid features\n";
        std::cout << "\n";
        std::cout << "// When creating windows, extract only valid features:\n";
        std::cout << "std::vector<float> extract_valid_features(const std::vector<float>& full_features) {\n";
        std::cout << "    std::vector<float> valid(" << valid_count << ");\n";
        std::cout << "    for (size_t i = 0; i < valid_indices.size(); ++i) {\n";
        std::cout << "        valid[i] = full_features[valid_indices[i]];\n";
        std::cout << "    }\n";
        std::cout << "    return valid;\n";
        std::cout << "}\n";
        std::cout << "```\n\n";
        
        // Specific issues
        auto check_modality = [this](const std::string& mod) {
            return modality_valid_counts[mod] > 0;
        };
        
        if (!check_modality("motor")) {
            std::cout << "❌ Motor data missing - check /rt/motorstate topic\n";
        }
        if (!check_modality("imu")) {
            std::cout << "❌ IMU data missing - check /rt/sportmodestate topic\n";
        }
        if (!check_modality("extended")) {
            std::cout << "⚠️  Extended features (84-255) not populated\n";
            std::cout << "   This is OK if these are reserved slots\n";
        }
    }
};

// ============================================================================
// Main function
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <binary_log>\n";
        std::cerr << "\nThis tool provides detailed feature-level diagnostics\n";
        std::cerr << "and identifies which features can be used for training.\n";
        return 1;
    }
    
    EnhancedDiagnostic diagnostic;
    diagnostic.analyze_file(argv[1]);
    
    return 0;
}