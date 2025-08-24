// ============================================================================
// apps/idle_poc_filtered.cpp - Train only on valid (non-NaN) features
// ============================================================================
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <algorithm>
#include <sstream>
#include <iomanip>

#include "../src/roca_idle_specialized.hpp"
#include "../src/io/binary_log.hpp"
#include "../src/data/window_maker.hpp"

using namespace roca;
using namespace roca_idle;

// ============================================================================
// Feature filtering utilities
// ============================================================================
class FeatureFilter {
private:
    std::vector<size_t> valid_indices;
    size_t original_features;
    size_t filtered_features;
    
public:
    // Automatically detect valid features from data
    void auto_detect_valid_features(const std::vector<std::vector<float>>& frames, 
                                   size_t num_features,
                                   float max_nan_ratio = 0.1f) {
        std::cout << "\nAuto-detecting valid features...\n";
        
        original_features = num_features;
        valid_indices.clear();
        
        // Count NaN occurrences per feature
        std::vector<size_t> nan_counts(num_features, 0);
        std::vector<size_t> inf_counts(num_features, 0);
        
        for (const auto& frame : frames) {
            for (size_t i = 0; i < num_features; ++i) {
                if (std::isnan(frame[i])) {
                    nan_counts[i]++;
                }
                if (std::isinf(frame[i])) {
                    inf_counts[i]++;
                }
            }
        }
        
        // Select features with acceptable NaN ratio and no infinities
        for (size_t i = 0; i < num_features; ++i) {
            float nan_ratio = static_cast<float>(nan_counts[i]) / frames.size();
            
            if (nan_ratio <= max_nan_ratio && inf_counts[i] == 0) {
                valid_indices.push_back(i);
            }
        }
        
        filtered_features = valid_indices.size();
        
        // Report findings
        std::cout << "  Original features: " << original_features << "\n";
        std::cout << "  Valid features: " << filtered_features << "\n";
        std::cout << "  Removed: " << (original_features - filtered_features) << " features\n";
        
        if (filtered_features < 30) {
            std::cerr << "\n⚠️  WARNING: Only " << filtered_features 
                     << " valid features found!\n";
            std::cerr << "  This may be too few for effective training.\n";
        }
        
        // Show which features were kept
        std::cout << "\nValid feature ranges:\n";
        size_t start = valid_indices[0];
        size_t end = start;
        
        for (size_t i = 1; i < valid_indices.size(); ++i) {
            if (valid_indices[i] == end + 1) {
                end = valid_indices[i];
            } else {
                std::cout << "  [" << start << "-" << end << "]";
                if (end - start + 1 == 1) {
                    std::cout << " (1 feature)\n";
                } else {
                    std::cout << " (" << (end - start + 1) << " features)\n";
                }
                start = valid_indices[i];
                end = start;
            }
        }
        std::cout << "  [" << start << "-" << end << "] (" 
                  << (end - start + 1) << " features)\n";
        
        // Save to file
        save_valid_indices("valid_features_auto.txt");
    }
    
    // Load valid indices from file (from enhanced diagnostic)
    bool load_valid_indices(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Warning: Cannot load " << filename << "\n";
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
        
        filtered_features = valid_indices.size();
        std::cout << "Loaded " << filtered_features << " valid feature indices from " 
                  << filename << "\n";
        
        return !valid_indices.empty();
    }
    
    // Save valid indices to file
    void save_valid_indices(const std::string& filename) {
        std::ofstream file(filename);
        file << "# Valid feature indices (auto-detected)\n";
        file << "# Total: " << valid_indices.size() << "\n";
        for (size_t idx : valid_indices) {
            file << idx << "\n";
        }
        std::cout << "  Saved valid indices to: " << filename << "\n";
    }
    
    // Filter a single frame to only valid features
    std::vector<float> filter_frame(const std::vector<float>& full_frame) {
        std::vector<float> filtered(filtered_features);
        for (size_t i = 0; i < valid_indices.size(); ++i) {
            filtered[i] = full_frame[valid_indices[i]];
        }
        return filtered;
    }
    
    // Filter a window (T frames concatenated)
    std::vector<float> filter_window(const std::vector<float>& full_window, size_t T) {
        size_t D_full = full_window.size() / T;
        std::vector<float> filtered_window;
        
        for (size_t t = 0; t < T; ++t) {
            for (size_t i = 0; i < valid_indices.size(); ++i) {
                size_t full_idx = t * D_full + valid_indices[i];
                filtered_window.push_back(full_window[full_idx]);
            }
        }
        
        return filtered_window;
    }
    
    size_t get_filtered_dimension() const { return filtered_features; }
    const std::vector<size_t>& get_valid_indices() const { return valid_indices; }
};

// ============================================================================
// Main training application with filtering
// ============================================================================
int main(int argc, char** argv) {
    std::cout << "\n╔═══════════════════════════════════════════╗\n";
    std::cout << "║  Filtered Idle Training (Valid Features)  ║\n";
    std::cout << "╚═══════════════════════════════════════════╝\n\n";
    
    // Parse arguments
    std::string data_file = "";
    bool use_auto_detect = true;
    std::string valid_features_file = "valid_features.txt";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--file" && i + 1 < argc) {
            data_file = argv[++i];
        } else if (arg == "--valid-features" && i + 1 < argc) {
            valid_features_file = argv[++i];
            use_auto_detect = false;
        } else if (arg == "--auto") {
            use_auto_detect = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " --file <data.bin> [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --auto                Auto-detect valid features (default)\n";
            std::cout << "  --valid-features <f>  Use feature list from file\n";
            return 0;
        }
    }
    
    if (data_file.empty()) {
        std::cerr << "Error: Please specify data file with --file\n";
        return 1;
    }
    
    // Step 1: Load raw data
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
    
    // Step 2: Filter features
    FeatureFilter filter;
    
    if (use_auto_detect) {
        filter.auto_detect_valid_features(raw_frames, original_features, 0.1f);
    } else {
        if (!filter.load_valid_indices(valid_features_file)) {
            std::cout << "Falling back to auto-detection\n";
            filter.auto_detect_valid_features(raw_frames, original_features, 0.1f);
        }
    }
    
    size_t filtered_D = filter.get_filtered_dimension();
    
    if (filtered_D < 10) {
        std::cerr << "\nError: Too few valid features (" << filtered_D 
                  << ") for training!\n";
        std::cerr << "Please check your data collection pipeline.\n";
        return 1;
    }
    
    // Step 3: Create filtered windows
    std::cout << "\nCreating filtered training windows...\n";
    
    IdleRoCAConfig config;
    config.T = 10;
    config.D = filtered_D;  // Use filtered dimension
    config.C = std::min(32, (int)filtered_D);  // Adjust latent size if needed
    config.K = std::min(16, (int)config.C / 2);
    config.epochs = 50;
    config.batch_size = 32;
    
    WindowConfig window_cfg;
    window_cfg.T = config.T;
    window_cfg.stride = 5;
    window_cfg.D = original_features;  // Original dimension for window maker
    
    WindowMaker maker(window_cfg);
    std::vector<std::vector<float>> filtered_windows;
    
    for (const auto& frame : raw_frames) {
        maker.push(frame);
        
        if (maker.ready()) {
            auto full_window = maker.get_window();
            auto filtered_window = filter.filter_window(full_window, config.T);
            filtered_windows.push_back(filtered_window);
        }
    }
    
    std::cout << "Created " << filtered_windows.size() << " filtered windows\n";
    std::cout << "Window dimensions: T=" << config.T << ", D=" << config.D 
              << " (filtered from " << original_features << ")\n";
    
    if (filtered_windows.size() < 100) {
        std::cerr << "\nWarning: Very few training windows. Results may be poor.\n";
    }
    
    // Step 4: Train model on filtered data
    std::cout << "\n--- Training on Filtered Features ---\n";
    
    IdleRoCAModel model(config);
    train_idle_model(model, filtered_windows, config);
    
    // Step 5: Validation test
    std::cout << "\n--- Validation Test ---\n";
    
    // Test on last 10% of data
    size_t test_start = filtered_windows.size() * 0.9;
    int anomalies = 0;
    
    std::vector<float> scores;
    for (size_t i = test_start; i < filtered_windows.size(); ++i) {
        float score = model.score_window(filtered_windows[i]);
        scores.push_back(score);
        if (score > model.anomaly_threshold) {
            anomalies++;
        }
    }
    
    size_t test_count = filtered_windows.size() - test_start;
    std::cout << "Test results: " << anomalies << "/" << test_count 
              << " flagged as anomalous (";
    std::cout << std::fixed << std::setprecision(1) 
              << (100.0f * anomalies / test_count) << "% rate)\n";
    
    // Show score distribution
    if (!scores.empty()) {
        std::sort(scores.begin(), scores.end());
        float min_score = scores.front();
        float max_score = scores.back();
        float median = scores[scores.size() / 2];
        
        std::cout << "\nScore distribution:\n";
        std::cout << "  Min: " << std::fixed << std::setprecision(6) << min_score << "\n";
        std::cout << "  Median: " << median << "\n";
        std::cout << "  Max: " << max_score << "\n";
        std::cout << "  Threshold: " << model.anomaly_threshold << "\n";
    }
    
    // Step 6: Save configuration
    std::cout << "\n--- Saving Configuration ---\n";
    
    std::ofstream config_file("training_config.txt");
    config_file << "# Training configuration for filtered model\n";
    config_file << "original_features: " << original_features << "\n";
    config_file << "filtered_features: " << filtered_D << "\n";
    config_file << "window_size: " << config.T << "\n";
    config_file << "latent_dim: " << config.C << "\n";
    config_file << "projection_dim: " << config.K << "\n";
    config_file << "training_windows: " << filtered_windows.size() << "\n";
    config_file << "anomaly_threshold: " << model.anomaly_threshold << "\n";
    config_file.close();
    
    std::cout << "Configuration saved to: training_config.txt\n";
    std::cout << "Valid feature indices saved to: valid_features_auto.txt\n";
    
    // Final summary
    std::cout << "\n╔═══════════════════════════════════════════╗\n";
    std::cout << "║              TRAINING COMPLETE             ║\n";
    std::cout << "╚═══════════════════════════════════════════╝\n\n";
    
    std::cout << "Summary:\n";
    std::cout << "  ✅ Successfully trained on " << filtered_D << " valid features\n";
    std::cout << "  ✅ Filtered out " << (original_features - filtered_D) 
              << " problematic features\n";
    std::cout << "  ✅ Model ready for anomaly detection\n";
    
    std::cout << "\nNext steps:\n";
    std::cout << "1. Test with movement data using the same feature filter\n";
    std::cout << "2. Deploy with feature indices from valid_features_auto.txt\n";
    std::cout << "3. Ensure inference also filters features the same way\n";



   // After training completes, save the model
#include "../src/model_io_complete.hpp"

// ... training code ...

// At the end of train_idle_model function:
std::cout << "\nSaving trained model...\n";

// Extract weights from your model
std::vector<std::vector<float>> encoder_weights;
std::vector<std::vector<float>> decoder_weights;

// Get encoder weights (you need to expose these from your model)
for (const auto& layer : model.encoder_layers) {
    encoder_weights.push_back(layer.W.data);
}

// Get decoder weights
for (const auto& layer : model.decoder_layers) {
    decoder_weights.push_back(layer.W.data);
}

// Save complete model
ModelSerializer::save_model(
    "trained_model.roca",
    processor.get_mean(),      // Training statistics
    processor.get_std(),        // Training statistics
    valid_indices,              // Which features are valid
    encoder_weights,            // Trained encoder weights
    decoder_weights,            // Trained decoder weights
    model.Ce,                   // Center vector
    model.anomaly_threshold     // Calibrated threshold
);
    
    return 0;
}