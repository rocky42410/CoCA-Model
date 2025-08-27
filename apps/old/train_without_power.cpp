#include <iostream>
#include "../src/roca_one_class.hpp"
#include "../src/io/binary_log.hpp"

using namespace roca;

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    
    BinaryLogReader reader;
    reader.open(argv[1]);
    auto raw_windows = reader.read_all_windows(10, 5);
    reader.close();
    
    // Features to use (excluding power features 36-39 and other problematic ones)
    std::vector<size_t> good_features;
    for (size_t i = 0; i < 36; ++i) {
        good_features.push_back(i);  // IMU, position, etc
    }
    // Skip 36-39 (power features)
    for (size_t i = 40; i < 256; ++i) {
        if (i >= 77 && i <= 83) {  // Skip other potentially corrupt features
            continue;
        }
        good_features.push_back(i);
    }
    
    std::cout << "Using " << good_features.size() << " features (excluding power)\n";
    
    // Extract clean features
    std::vector<std::vector<float>> clean_windows;
    for (const auto& window : raw_windows) {
        std::vector<float> clean;
        for (size_t t = 0; t < 10; ++t) {
            for (size_t idx : good_features) {
                float val = window[t * 256 + idx];
                
                // Skip if NaN or extreme
                if (std::isnan(val) || std::abs(val) > 100) {
                    val = 0;
                }
                clean.push_back(val);
            }
        }
        
        // Only keep if reasonable size
        if (clean.size() == 10 * good_features.size()) {
            clean_windows.push_back(clean);
        }
    }
    
    std::cout << "Created " << clean_windows.size() << " clean windows\n";
    
    // Check for remaining NaN/extreme values
    int extreme_count = 0;
    for (const auto& w : clean_windows) {
        for (float v : w) {
            if (std::abs(v) > 50) extreme_count++;
        }
    }
    
    if (extreme_count > 0) {
        std::cout << "Warning: " << extreme_count << " extreme values remain\n";
    }
    
    // Normalize
    size_t feat_dim = good_features.size();
    std::vector<float> mean(10 * feat_dim, 0);
    std::vector<float> std(10 * feat_dim, 1);
    
    for (const auto& w : clean_windows) {
        for (size_t i = 0; i < w.size(); ++i) {
            mean[i] += w[i];
        }
    }
    for (auto& m : mean) m /= clean_windows.size();
    
    for (const auto& w : clean_windows) {
        for (size_t i = 0; i < w.size(); ++i) {
            float diff = w[i] - mean[i];
            std[i] += diff * diff;
        }
    }
    for (auto& s : std) {
        s = std::sqrt(s / clean_windows.size());
        if (s < 1e-6) s = 1.0f;
    }
    
    for (auto& w : clean_windows) {
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = (w[i] - mean[i]) / std[i];
            w[i] = std::max(-3.0f, std::min(3.0f, w[i]));
        }
    }
    
    // Train
    OneClassConfig config;
    config.T = 10;
    config.D = feat_dim;
    config.C = 32;
    config.K = 16;
    config.epochs = 100;
    config.lr = 1e-3;
    
    OneClassRoCA model(config);
    train_one_class_model(model, clean_windows, config);
    
    return 0;
}