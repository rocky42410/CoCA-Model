#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "../src/io/binary_log.hpp"
#include "../src/feature_filter.hpp"
#include <cstdint>

using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data.bin>\n";
        return 1;
    }
    
    BinaryLogReader reader;
    if (!reader.open(argv[1])) return 1;
    
    auto raw_windows = reader.read_all_windows(10, 5);
    reader.close();
    
    // Filter to valid features
    FeatureFilter filter;
    filter.fit(raw_windows, 10);
    
    std::vector<std::vector<float>> filtered_windows;
    for (const auto& raw : raw_windows) {
        filtered_windows.push_back(filter.filter_window(raw, 10));
    }
    
    // Analyze the filtered data
    size_t num_features = filter.get_output_dim();
    std::cout << "\n=== Data Statistics (Filtered) ===\n";
    std::cout << "Windows: " << filtered_windows.size() << "\n";
    std::cout << "Features: " << num_features << "\n";
    
    // Compute statistics per feature
    std::vector<float> feat_min(num_features, 1e9);
    std::vector<float> feat_max(num_features, -1e9);
    std::vector<float> feat_mean(num_features, 0);
    std::vector<float> feat_std(num_features, 0);
    
    // First pass: min, max, mean
    for (const auto& window : filtered_windows) {
        for (size_t t = 0; t < 10; ++t) {
            for (size_t f = 0; f < num_features; ++f) {
                float val = window[t * num_features + f];
                feat_min[f] = std::min(feat_min[f], val);
                feat_max[f] = std::max(feat_max[f], val);
                feat_mean[f] += val;
            }
        }
    }
    
    for (size_t f = 0; f < num_features; ++f) {
        feat_mean[f] /= (filtered_windows.size() * 10);
    }
    
    // Second pass: std dev
    for (const auto& window : filtered_windows) {
        for (size_t t = 0; t < 10; ++t) {
            for (size_t f = 0; f < num_features; ++f) {
                float val = window[t * num_features + f];
                feat_std[f] += (val - feat_mean[f]) * (val - feat_mean[f]);
            }
        }
    }
    
    for (size_t f = 0; f < num_features; ++f) {
        feat_std[f] = std::sqrt(feat_std[f] / (filtered_windows.size() * 10));
    }
    
    // Identify problematic features
    std::cout << "\n=== Feature Analysis ===\n";
    
    int constant_features = 0;
    int high_variance_features = 0;
    int extreme_range_features = 0;
    
    const auto& valid_indices = filter.get_valid_indices();
    
    for (size_t f = 0; f < num_features; ++f) {
        bool problematic = false;
        
        // Check if constant (std < 1e-6)
        if (feat_std[f] < 1e-6) {
            constant_features++;
            problematic = true;
        }
        
        // Check if high variance (std > mean * 10)
        if (feat_std[f] > std::abs(feat_mean[f]) * 10 && std::abs(feat_mean[f]) > 0.01) {
            high_variance_features++;
            problematic = true;
        }
        
        // Check if extreme range
        float range = feat_max[f] - feat_min[f];
        if (range > 1000 || std::abs(feat_max[f]) > 1000 || std::abs(feat_min[f]) > 1000) {
            extreme_range_features++;
            problematic = true;
        }
        
        if (problematic) {
            std::cout << "Feature " << f << " (orig idx " << valid_indices[f] << "):\n";
            std::cout << "  Range: [" << feat_min[f] << ", " << feat_max[f] << "]\n";
            std::cout << "  Mean: " << feat_mean[f] << ", Std: " << feat_std[f] << "\n";
        }
    }
    
    std::cout << "\nSummary:\n";
    std::cout << "  Constant features: " << constant_features << "/" << num_features << "\n";
    std::cout << "  High variance features: " << high_variance_features << "/" << num_features << "\n";
    std::cout << "  Extreme range features: " << extreme_range_features << "/" << num_features << "\n";
    
    // Sample first window values
    std::cout << "\n=== First Window Sample (first 10 features) ===\n";
    for (size_t f = 0; f < std::min(size_t(10), num_features); ++f) {
        std::cout << "Feature " << f << ": ";
        for (size_t t = 0; t < 3; ++t) {  // First 3 timesteps
            std::cout << filtered_windows[0][t * num_features + f] << " ";
        }
        std::cout << "...\n";
    }
    
    return 0;
}