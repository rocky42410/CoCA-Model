#include <iostream>
#include <vector>
#include <set>
#include "../src/roca_one_class.hpp"
#include "../src/io/binary_log.hpp"

using namespace roca;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data.bin>\n";
        return 1;
    }
    
    BinaryLogReader reader;
    reader.open(argv[1]);
    auto raw_windows = reader.read_all_windows(10, 5);
    reader.close();
    
    std::cout << "Loaded " << raw_windows.size() << " raw windows\n";
    
    // First pass: identify valid features (non-NaN, non-corrupt)
    std::vector<bool> feature_valid(256, true);
    std::vector<int> nan_counts(256, 0);
    std::vector<float> feature_min(256, 1e9);
    std::vector<float> feature_max(256, -1e9);
    
    for (const auto& window : raw_windows) {
        for (size_t t = 0; t < 10; ++t) {
            for (size_t f = 0; f < 256; ++f) {
                float val = window[t * 256 + f];
                
                if (std::isnan(val) || std::isinf(val)) {
                    nan_counts[f]++;
                } else {
                    feature_min[f] = std::min(feature_min[f], val);
                    feature_max[f] = std::max(feature_max[f], val);
                }
            }
        }
    }
    
    // Mark features as invalid if:
    // 1. More than 5% NaN
    // 2. Extreme values (> 1000 or < -1000)
    // 3. Known corrupt features (36-39 for power)
    std::vector<size_t> valid_features;
    int total_samples = raw_windows.size() * 10;
    
    for (size_t f = 0; f < 256; ++f) {
        bool valid = true;
        
        // Check NaN ratio
        float nan_ratio = (float)nan_counts[f] / total_samples;
        if (nan_ratio > 0.05) {
            valid = false;
            std::cout << "Feature " << f << ": too many NaN (" 
                     << (nan_ratio * 100) << "%)\n";
        }
        // Check extreme values
        else if (std::abs(feature_min[f]) > 1000 || std::abs(feature_max[f]) > 1000) {
            valid = false;
            std::cout << "Feature " << f << ": extreme values [" 
                     << feature_min[f] << ", " << feature_max[f] << "]\n";
        }
        // Known corrupt power features
        else if (f >= 36 && f <= 39) {
            valid = false;
            std::cout << "Feature " << f << ": excluded (power feature)\n";
        }
        // All zeros (constant)
        else if (feature_min[f] == 0 && feature_max[f] == 0) {
            valid = false;
            std::cout << "Feature " << f << ": all zeros\n";
        }
        
        if (valid) {
            valid_features.push_back(f);
        }
    }
    
    std::cout << "\nValid features: " << valid_features.size() << "/256\n";
    
    if (valid_features.size() < 10) {
        std::cerr << "Too few valid features to train!\n";
        return 1;
    }
    
    // Extract only valid features
    std::vector<std::vector<float>> clean_windows;
    for (const auto& window : raw_windows) {
        std::vector<float> clean;
        bool window_valid = true;
        
        for (size_t t = 0; t < 10; ++t) {
            for (size_t idx : valid_features) {
                float val = window[t * 256 + idx];
                
                // Final safety check
                if (std::isnan(val) || std::isinf(val)) {
                    val = 0;  // Replace with 0
                    window_valid = false;
                }
                
                clean.push_back(val);
            }
        }
        
        // Only add windows that are completely valid
        if (window_valid) {
            clean_windows.push_back(clean);
        }
    }
    
    std::cout << "Clean windows: " << clean_windows.size() << "\n";
    
    if (clean_windows.size() < 100) {
        std::cerr << "Too few clean windows for training!\n";
        return 1;
    }
    
    // Normalize
    size_t feat_dim = valid_features.size();
    size_t total_dim = 10 * feat_dim;
    std::vector<float> mean(total_dim, 0);
    std::vector<float> std(total_dim, 1);
    
    // Calculate mean
    for (const auto& w : clean_windows) {
        for (size_t i = 0; i < total_dim; ++i) {
            mean[i] += w[i];
        }
    }
    for (auto& m : mean) {
        m /= clean_windows.size();
    }
    
    // Calculate std
    for (const auto& w : clean_windows) {
        for (size_t i = 0; i < total_dim; ++i) {
            float diff = w[i] - mean[i];
            std[i] += diff * diff;
        }
    }
    for (auto& s : std) {
        s = std::sqrt(s / clean_windows.size());
        if (s < 1e-6) s = 1.0f;
    }
    
    // Normalize and clip
    for (auto& w : clean_windows) {
        for (size_t i = 0; i < total_dim; ++i) {
            w[i] = (w[i] - mean[i]) / std[i];
            w[i] = std::max(-3.0f, std::min(3.0f, w[i]));
        }
    }
    
    // Check final data quality
    float max_val = -1e9, min_val = 1e9;
    for (const auto& w : clean_windows) {
        for (float v : w) {
            max_val = std::max(max_val, v);
            min_val = std::min(min_val, v);
        }
    }
    
    std::cout << "\nNormalized data range: [" << min_val << ", " << max_val << "]\n";
    std::cout << "Expected range: [-3, 3]\n";
    
    // Save valid feature indices for later use
    std::ofstream feat_file("valid_features.txt");
    feat_file << "# Valid feature indices (" << valid_features.size() << " total)\n";
    for (size_t idx : valid_features) {
        feat_file << idx << "\n";
    }
    feat_file.close();
    
    // Train model
    OneClassConfig config;
    config.T = 10;
    config.D = feat_dim;
    config.C = 32;  
    config.K = 16;
    config.epochs = 100;
    config.batch_size = 32;
    config.lr = 1e-3;
    config.lambda_rec = 1.0f;
    config.lambda_oc = 0.5f;
    config.lambda_aug = 0.3f;
    
    std::cout << "\nStarting training with " << feat_dim << " features...\n";
    
    OneClassRoCA model(config);
    train_one_class_model(model, clean_windows, config);
    
    // Save model
    model.save_model("model_clean.roca");
    std::cout << "\nModel saved to model_clean.roca\n";
    
    return 0;
}