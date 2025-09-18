// ============================================================================
// apps/coca_train.cpp - Main COCA training application
// ============================================================================
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "../src/coca_model.hpp"
#include "../src/io/binary_log.hpp"
#include "../src/data/window_maker.hpp"
#include "../src/utils/config_parser.hpp"
#include "../src/utils/model_io.hpp"

using namespace coca;
using namespace roca;

// ============================================================================
// Feature filter for handling valid features only
// ============================================================================
class FeatureFilter {
private:
    std::vector<size_t> valid_indices;
    size_t filtered_dimension = 0;
    
public:
    bool load_from_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Warning: Cannot load feature filter from " << filename << "\n";
            return false;
        }
        
        valid_indices.clear();
        std::string line;
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            
            size_t idx;
            std::istringstream iss(line);
            if (iss >> idx) {
                valid_indices.push_back(idx);
            }
        }
        
        filtered_dimension = valid_indices.size();
        std::cout << "Loaded " << filtered_dimension << " valid feature indices\n";
        return !valid_indices.empty();
    }
    
    void auto_detect(const std::vector<std::vector<float>>& frames, size_t num_features) {
        valid_indices.clear();
        
        std::vector<size_t> nan_counts(num_features, 0);
        std::vector<size_t> inf_counts(num_features, 0);
        
        for (const auto& frame : frames) {
            for (size_t i = 0; i < num_features; ++i) {
                if (std::isnan(frame[i])) nan_counts[i]++;
                if (std::isinf(frame[i])) inf_counts[i]++;
            }
        }
        
        for (size_t i = 0; i < num_features; ++i) {
            float nan_ratio = static_cast<float>(nan_counts[i]) / frames.size();
            if (nan_ratio < 0.1f && inf_counts[i] == 0) {
                valid_indices.push_back(i);
            }
        }
        
        filtered_dimension = valid_indices.size();
        std::cout << "Auto-detected " << filtered_dimension << " valid features\n";
        
        // Save detected indices
        std::ofstream out("valid_features_auto.txt");
        out << "# Valid feature indices (auto-detected)\n";
        out << "# Total: " << valid_indices.size() << "\n";
        for (size_t idx : valid_indices) {
            out << idx << "\n";
        }
        out.close();
    }
    
    std::vector<float> filter_window(const std::vector<float>& full_window, size_t T, size_t D_full) {
        std::vector<float> filtered;
        
        for (size_t t = 0; t < T; ++t) {
            for (size_t idx : valid_indices) {
                filtered.push_back(full_window[t * D_full + idx]);
            }
        }
        
        return filtered;
    }
    
    size_t get_dimension() const { return filtered_dimension; }
    bool is_active() const { return !valid_indices.empty(); }
};

// ============================================================================
// Main application
// ============================================================================
int main(int argc, char** argv) {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║        COCA Training Application       ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    // Parse command line arguments
    std::string data_file = "";
    std::string config_file = "coca_config.yaml";
    std::string filter_file = "";
    std::string output_model = "trained_model.coca";
    bool use_filter = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_file = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--filter" && i + 1 < argc) {
            filter_file = argv[++i];
            use_filter = true;
        } else if (arg == "--output" && i + 1 < argc) {
            output_model = argv[++i];
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " --data <file> [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --data <file>    Input binary log file (required)\n";
            std::cout << "  --config <file>  Configuration file (default: coca_config.yaml)\n";
            std::cout << "  --filter <file>  Valid features file (optional)\n";
            std::cout << "  --output <file>  Output model file (default: trained_model.coca)\n";
            return 0;
        }
    }
    
    if (data_file.empty()) {
        std::cerr << "Error: Please specify data file with --data\n";
        return 1;
    }
    
    // Load configuration
    COCAConfig config;
    if (!ConfigParser::load_config(config_file, config)) {
        std::cout << "Using default configuration\n";
    }
    
    std::cout << "Configuration:\n";
    std::cout << "  Window size: " << config.T << "\n";
    std::cout << "  Latent dim: " << config.C << "\n";
    std::cout << "  Projection dim: " << config.K << "\n";
    std::cout << "  λ_rec: " << config.lambda_rec << "\n";
    std::cout << "  λ_inv: " << config.lambda_inv << "\n";
    std::cout << "  λ_var: " << config.lambda_var << "\n";
    std::cout << "  ζ: " << config.zeta << "\n";
    std::cout << "  Score mix: " << config.score_mix << "\n";
    std::cout << "  Threshold mode: " << config.threshold_mode << "\n";
    std::cout << "  Seed: " << config.seed << "\n\n";
    
    // Load data
    std::cout << "Loading data from: " << data_file << "\n";
    
    std::ifstream file(data_file, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file\n";
        return 1;
    }
    
    BinaryFileHeader header;
    if (!read_header(file, header)) {
        std::cerr << "Error: Invalid header\n";
        return 1;
    }
    
    size_t original_features = header.feature_count;
    std::cout << "Original feature count: " << original_features << "\n";
    
    // Load all frames
    std::vector<std::vector<float>> raw_frames;
    AutoencoderFrame frame;
    
    while (read_frame(file, frame)) {
        std::vector<float> features(original_features);
        for (size_t i = 0; i < original_features; ++i) {
            features[i] = frame.features[i];
        }
        raw_frames.push_back(features);
    }
    
    std::cout << "Loaded " << raw_frames.size() << " frames\n";
    
    if (raw_frames.empty()) {
        std::cerr << "Error: No frames loaded\n";
        return 1;
    }
    
    // Feature filtering
    FeatureFilter filter;
    size_t actual_features = original_features;
    
    if (use_filter && !filter_file.empty()) {
        if (filter.load_from_file(filter_file)) {
            actual_features = filter.get_dimension();
        } else {
            std::cout << "Falling back to auto-detection\n";
            filter.auto_detect(raw_frames, original_features);
            actual_features = filter.get_dimension();
        }
    } else if (use_filter) {
        std::cout << "Auto-detecting valid features...\n";
        filter.auto_detect(raw_frames, original_features);
        actual_features = filter.get_dimension();
    }
    
    // Update config dimension
    config.D = actual_features;
    
    std::cout << "\nFeatures: " << config.D << " (from " << original_features << ")\n";
    
    // Create windows
    std::cout << "\nCreating windows...\n";
    
    WindowConfig window_cfg;
    window_cfg.T = config.T;
    window_cfg.stride = 5;
    window_cfg.D = original_features;
    
    WindowMaker maker(window_cfg);
    std::vector<std::vector<float>> windows;
    
    for (const auto& frame : raw_frames) {
        maker.push(frame);
        
        if (maker.ready()) {
            auto full_window = maker.get_window();
            
            if (filter.is_active()) {
                auto filtered_window = filter.filter_window(full_window, config.T, original_features);
                windows.push_back(filtered_window);
            } else {
                windows.push_back(full_window);
            }
        }
    }
    
    std::cout << "Created " << windows.size() << " windows\n";
    
    if (windows.size() < 100) {
        std::cerr << "Warning: Very few training windows\n";
    }
    
    // Train model
    std::cout << "\n--- Starting Training ---\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    COCAModel model(config);
    train_coca_model(model, windows, config);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\nTraining time: " << duration.count() << " seconds\n";
    
    // Save model
    std::cout << "\nSaving model to: " << output_model << "\n";
    
    ModelIO::save_model(model, output_model);
    
    // Save configuration summary
    std::ofstream summary("training_summary.txt");
    summary << "# COCA Training Summary\n";
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    summary << "# " << std::ctime(&time_t) << "\n";
    summary << "Data file: " << data_file << "\n";
    summary << "Original features: " << original_features << "\n";
    summary << "Filtered features: " << config.D << "\n";
    summary << "Training windows: " << windows.size() << "\n";
    summary << "Window size: " << config.T << "\n";
    summary << "Latent dim: " << config.C << "\n";
    summary << "Projection dim: " << config.K << "\n";
    summary << "Final threshold: " << model.anomaly_threshold << "\n";
    summary << "Training time: " << duration.count() << " seconds\n";
    summary.close();
    
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║         TRAINING COMPLETE              ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    std::cout << "Outputs:\n";
    std::cout << "  Model: " << output_model << "\n";
    std::cout << "  Log: training_log.csv\n";
    std::cout << "  Summary: training_summary.txt\n";
    
    if (filter.is_active()) {
        std::cout << "  Valid features: valid_features_auto.txt\n";
    }
    
    return 0;
}
