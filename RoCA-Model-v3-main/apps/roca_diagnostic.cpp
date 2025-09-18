// ============================================================================
// apps/roca_diagnostic.cpp - Diagnostic tool to identify data issues
// ============================================================================
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <map>

#include "../src/io/binary_log.hpp"
#include "../src/data/window_maker.hpp"

using namespace roca;

struct FeatureStats {
    size_t index;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    float mean = 0.0f;
    float std = 0.0f;
    size_t nan_count = 0;
    size_t zero_count = 0;
    size_t constant_count = 0;
    bool is_constant = false;
    std::vector<float> samples;  // Store some samples for inspection
};

class DataDiagnostic {
private:
    std::vector<FeatureStats> feature_stats;
    size_t num_features;
    size_t total_frames = 0;
    size_t valid_frames = 0;
    size_t windows_created = 0;
    
    // Issues found
    std::vector<size_t> nan_features;
    std::vector<size_t> constant_features;
    std::vector<size_t> extreme_features;
    
public:
    void analyze_log(const std::string& path, size_t window_size = 10, size_t stride = 5) {
        std::cout << "\n=== RoCA Data Diagnostic Tool ===\n\n";
        std::cout << "Analyzing: " << path << "\n";
        
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot open file\n";
            return;
        }
        
        // Read header
        BinaryFileHeader header;
        if (!read_header(file, header)) {
            std::cerr << "Error: Invalid header\n";
            return;
        }
        
        num_features = header.feature_count;
        feature_stats.resize(num_features);
        for (size_t i = 0; i < num_features; ++i) {
            feature_stats[i].index = i;
        }
        
        std::cout << "\nFile Header Info:\n";
        std::cout << "  Feature count: " << header.feature_count << "\n";
        std::cout << "  Fusion rate: " << header.fusion_rate_hz << " Hz\n";
        std::cout << "  Is little endian: " << (int)header.is_little_endian << "\n";
        
        // First pass: collect all data
        std::vector<std::vector<float>> all_frames;
        AutoencoderFrame frame;
        
        while (read_frame(file, frame)) {
            total_frames++;
            
            // Check staleness
            bool too_stale = false;
            for (int i = 0; i < 5; ++i) {
                if (frame.staleness_ms[i] > 100.0f) {
                    too_stale = true;
                    break;
                }
            }
            
            if (!too_stale) {
                valid_frames++;
                std::vector<float> features(num_features);
                for (size_t i = 0; i < num_features; ++i) {
                    features[i] = frame.features[i];
                }
                all_frames.push_back(features);
            }
            
            // Sample first 100 frames for detailed inspection
            if (total_frames <= 100) {
                for (size_t i = 0; i < num_features; ++i) {
                    feature_stats[i].samples.push_back(frame.features[i]);
                }
            }
        }
        
        std::cout << "\nFrame Statistics:\n";
        std::cout << "  Total frames: " << total_frames << "\n";
        std::cout << "  Valid frames: " << valid_frames << "\n";
        std::cout << "  Dropped frames: " << (total_frames - valid_frames) << "\n";
        
        if (all_frames.empty()) {
            std::cerr << "\nError: No valid frames found!\n";
            return;
        }
        
        // Analyze features
        analyze_features(all_frames);
        
        // Create windows and check for issues
        analyze_windows(all_frames, window_size, stride);
        
        // Report findings
        report_issues();
        
        // Suggest fixes
        suggest_fixes();
    }
    
private:
    void analyze_features(const std::vector<std::vector<float>>& frames) {
        std::cout << "\nAnalyzing " << num_features << " features across " 
                  << frames.size() << " frames...\n";
        
        // Compute basic stats
        for (size_t f = 0; f < num_features; ++f) {
            auto& stats = feature_stats[f];
            std::vector<float> values;
            
            for (const auto& frame : frames) {
                float val = frame[f];
                
                if (std::isnan(val)) {
                    stats.nan_count++;
                } else {
                    values.push_back(val);
                    stats.min_val = std::min(stats.min_val, val);
                    stats.max_val = std::max(stats.max_val, val);
                    
                    if (std::abs(val) < 1e-10f) {
                        stats.zero_count++;
                    }
                }
            }
            
            if (!values.empty()) {
                // Compute mean
                stats.mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
                
                // Compute std
                float variance = 0.0f;
                for (float val : values) {
                    float diff = val - stats.mean;
                    variance += diff * diff;
                }
                stats.std = std::sqrt(variance / values.size());
                
                // Check if constant
                if (stats.std < 1e-8f) {
                    stats.is_constant = true;
                    constant_features.push_back(f);
                }
                
                // Check for extreme values
                if (std::abs(stats.mean) > 1e6f || stats.std > 1e6f) {
                    extreme_features.push_back(f);
                }
            }
            
            // Check for high NaN rate
            if (stats.nan_count > frames.size() / 2) {
                nan_features.push_back(f);
            }
        }
    }
    
    void analyze_windows(const std::vector<std::vector<float>>& frames,
                        size_t window_size, size_t stride) {
        std::cout << "\nCreating windows (T=" << window_size << ", stride=" << stride << ")...\n";
        
        WindowConfig cfg;
        cfg.T = window_size;
        cfg.stride = stride;
        cfg.D = num_features;
        
        WindowMaker maker(cfg);
        
        for (const auto& frame : frames) {
            // Replace NaN with 0 for window creation
            std::vector<float> clean_frame = frame;
            for (auto& val : clean_frame) {
                if (std::isnan(val)) val = 0.0f;
            }
            
            maker.push(clean_frame);
            if (maker.ready()) {
                windows_created++;
                maker.get_window();
            }
        }
        
        std::cout << "  Windows created: " << windows_created << "\n";
    }
    
    void report_issues() {
        std::cout << "\n=== ISSUES DETECTED ===\n";
        
        bool has_issues = false;
        
        // Report NaN features
        if (!nan_features.empty()) {
            has_issues = true;
            std::cout << "\n❌ Features with >50% NaN values: " << nan_features.size() << "\n";
            std::cout << "   Feature indices: ";
            for (size_t i = 0; i < std::min(size_t(10), nan_features.size()); ++i) {
                std::cout << nan_features[i] << " ";
            }
            if (nan_features.size() > 10) std::cout << "...";
            std::cout << "\n";
        }
        
        // Report constant features
        if (!constant_features.empty()) {
            has_issues = true;
            std::cout << "\n⚠️  Constant features (zero variance): " << constant_features.size() << "\n";
            std::cout << "   Feature indices: ";
            for (size_t i = 0; i < std::min(size_t(10), constant_features.size()); ++i) {
                size_t idx = constant_features[i];
                std::cout << idx << "(" << std::fixed << std::setprecision(2) 
                         << feature_stats[idx].mean << ") ";
            }
            if (constant_features.size() > 10) std::cout << "...";
            std::cout << "\n";
        }
        
        // Report extreme features
        if (!extreme_features.empty()) {
            has_issues = true;
            std::cout << "\n⚠️  Features with extreme values: " << extreme_features.size() << "\n";
            for (size_t idx : extreme_features) {
                auto& stats = feature_stats[idx];
                std::cout << "   Feature " << idx << ": mean=" << std::scientific 
                         << stats.mean << ", std=" << stats.std << "\n";
            }
        }
        
        // Show sample of normal features
        std::cout << "\n✓ Sample of normal features:\n";
        int shown = 0;
        for (size_t i = 0; i < num_features && shown < 5; ++i) {
            auto& stats = feature_stats[i];
            if (!stats.is_constant && stats.nan_count == 0 && 
                std::abs(stats.mean) < 1e6f && stats.std > 1e-6f) {
                std::cout << "   Feature " << i << ": "
                         << "mean=" << std::fixed << std::setprecision(3) << stats.mean
                         << ", std=" << stats.std
                         << ", range=[" << stats.min_val << ", " << stats.max_val << "]\n";
                shown++;
            }
        }
        
        if (!has_issues) {
            std::cout << "\n✓ No major issues detected in the data!\n";
        }
    }
    
    void suggest_fixes() {
        std::cout << "\n=== RECOMMENDED FIXES ===\n";
        
        if (!nan_features.empty()) {
            std::cout << "\nFor NaN features:\n";
            std::cout << "  1. Check sensor connections and data fusion pipeline\n";
            std::cout << "  2. Verify topic subscription and message parsing\n";
            std::cout << "  3. Consider removing features that are always NaN\n";
        }
        
        if (!constant_features.empty()) {
            std::cout << "\nFor constant features:\n";
            std::cout << "  1. These features provide no information for anomaly detection\n";
            std::cout << "  2. Check if sensors are active during data collection\n";
            std::cout << "  3. For idle robot, some features (like velocity) may naturally be zero\n";
            std::cout << "  4. Consider:\n";
            std::cout << "     - Removing constant features from training\n";
            std::cout << "     - Setting their std to 1.0 to avoid division by zero\n";
            std::cout << "     - Collecting more diverse data with robot movement\n";
        }
        
        if (!extreme_features.empty()) {
            std::cout << "\nFor extreme values:\n";
            std::cout << "  1. Check unit conversions (e.g., radians vs degrees)\n";
            std::cout << "  2. Verify sensor calibration\n";
            std::cout << "  3. Consider feature scaling or clipping\n";
        }
        
        std::cout << "\n=== TRAINING RECOMMENDATIONS ===\n";
        std::cout << "  1. Use the fixed implementation with safe normalization\n";
        std::cout << "  2. Start with a smaller learning rate (e.g., 1e-4)\n";
        std::cout << "  3. Monitor loss values closely in first epoch\n";
        std::cout << "  4. Consider collecting data with more variation:\n";
        std::cout << "     - Include some movement (stand/sit cycles)\n";
        std::cout << "     - Add controlled disturbances\n";
        std::cout << "     - Ensure all sensors are active\n";
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <binary_log> [--window=N] [--stride=N]\n";
        return 1;
    }
    
    std::string log_path = argv[1];
    size_t window_size = 10;
    size_t stride = 5;
    
    // Parse optional arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--window=") == 0) {
            window_size = std::stoul(arg.substr(9));
        } else if (arg.find("--stride=") == 0) {
            stride = std::stoul(arg.substr(9));
        }
    }
    
    DataDiagnostic diagnostic;
    diagnostic.analyze_log(log_path, window_size, stride);
    
    return 0;
}