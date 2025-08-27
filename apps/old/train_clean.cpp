#include <iostream>
#include <vector>
#include <set>
#include "../src/roca_one_class.hpp"
#include "../src/io/binary_log.hpp"

using namespace roca;

class CleanFeatureFilter {
private:
    std::vector<size_t> valid_indices;
    
public:
    void fit(const std::vector<std::vector<float>>& windows, size_t window_size) {
        size_t num_features = windows[0].size() / window_size;
        
        // Identify problematic features
        std::set<size_t> exclude_features;
        
        for (size_t f = 0; f < num_features; ++f) {
            std::vector<float> values;
            for (const auto& window : windows) {
                for (size_t t = 0; t < window_size; ++t) {
                    float val = window[t * num_features + f];
                    if (!std::isnan(val) && !std::isinf(val)) {
                        values.push_back(val);
                    }
                }
            }
            
            if (values.empty()) {
                exclude_features.insert(f);
                continue;
            }
            
            // Exclude if:
            // 1. All zeros (constant)
            float max_val = *std::max_element(values.begin(), values.end());
            float min_val = *std::min_element(values.begin(), values.end());
            
            if (max_val == 0 && min_val == 0) {
                exclude_features.insert(f);
                std::cout << "Excluding feature " << f << ": all zeros\n";
            }
            // 2. Extreme values (>1000 or <-1000)
            else if (std::abs(max_val) > 1000 || std::abs(min_val) > 1000) {
                exclude_features.insert(f);
                std::cout << "Excluding feature " << f << ": extreme values [" 
                         << min_val << ", " << max_val << "]\n";
            }
            // 3. NaN/Inf present
            else if (values.size() < windows.size() * window_size * 0.95) {
                exclude_features.insert(f);
                std::cout << "Excluding feature " << f << ": too many NaN\n";
            }
        }
        
        // Build valid indices
        for (size_t f = 0; f < num_features; ++f) {
            if (exclude_features.find(f) == exclude_features.end()) {
                valid_indices.push_back(f);
            }
        }
        
        std::cout << "\nKept " << valid_indices.size() << " clean features\n";
    }
    
    std::vector<float> filter(const std::vector<float>& window, size_t window_size, size_t orig_features) const {
        std::vector<float> filtered;
        for (size_t t = 0; t < window_size; ++t) {
            for (size_t idx : valid_indices) {
                filtered.push_back(window[t * orig_features + idx]);
            }
        }
        return filtered;
    }
    
    size_t get_num_features() const { return valid_indices.size(); }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data.bin>\n";
        return 1;
    }
    
    // Load data
    BinaryLogReader reader;
    reader.open(argv[1]);
    auto raw_windows = reader.read_all_windows(10, 5);
    reader.close();
    
    // First filter: remove NaN features (256 -> 83)
    std::vector<std::vector<float>> step1_windows;
    for (const auto& window : raw_windows) {
        std::vector<float> filtered;
        for (size_t t = 0; t < 10; ++t) {
            for (size_t f = 0; f < 256; ++f) {
                float val = window[t * 256 + f];
                if (!std::isnan(val) && !std::isinf(val)) {
                    filtered.push_back(val);
                }
            }
        }
        // Pad if needed
        while (filtered.size() < 10 * 83) {
            filtered.push_back(0.0f);
        }
        step1_windows.push_back(filtered);
    }
    
    // Second filter: remove problematic features
    CleanFeatureFilter cleaner;
    cleaner.fit(step1_windows, 10);
    
    std::vector<std::vector<float>> clean_windows;
    for (const auto& window : step1_windows) {
        clean_windows.push_back(cleaner.filter(window, 10, 83));
    }
    
    // Simple normalization (mean=0, std=1)
    size_t num_features = cleaner.get_num_features();
    std::vector<float> feature_mean(num_features, 0);
    std::vector<float> feature_std(num_features, 0);
    
    // Calculate mean
    for (const auto& window : clean_windows) {
        for (size_t i = 0; i < window.size(); ++i) {
            feature_mean[i % num_features] += window[i];
        }
    }
    for (auto& m : feature_mean) {
        m /= (clean_windows.size() * 10);
    }
    
    // Calculate std
    for (const auto& window : clean_windows) {
        for (size_t i = 0; i < window.size(); ++i) {
            float diff = window[i] - feature_mean[i % num_features];
            feature_std[i % num_features] += diff * diff;
        }
    }
    for (auto& s : feature_std) {
        s = std::sqrt(s / (clean_windows.size() * 10));
        if (s < 1e-6) s = 1.0f; // Avoid division by zero
    }
    
    // Normalize
    std::vector<std::vector<float>> normalized_windows;
    for (const auto& window : clean_windows) {
        std::vector<float> normalized;
        for (size_t i = 0; i < window.size(); ++i) {
            float val = (window[i] - feature_mean[i % num_features]) / feature_std[i % num_features];
            normalized.push_back(std::max(-5.0f, std::min(5.0f, val))); // Clip
        }
        normalized_windows.push_back(normalized);
    }
    
    std::cout << "\nFinal data shape:\n";
    std::cout << "  Windows: " << normalized_windows.size() << "\n";
    std::cout << "  Features: " << num_features << "\n";
    
    // Train
    OneClassConfig config;
    config.T = 10;
    config.D = num_features;
    config.C = 32;
    config.K = 16;
    config.epochs = 100;
    config.lr = 1e-3;
    
    OneClassRoCA model(config);
    train_one_class_model(model, normalized_windows, config);
    
    return 0;
}