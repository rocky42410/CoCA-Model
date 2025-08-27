#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>

class FeatureFilter {
private:
    std::vector<size_t> valid_indices;
    size_t input_dim;
    size_t output_dim;
    
public:
    FeatureFilter() : input_dim(256), output_dim(0) {}
    
    // Analyze data to find valid features
    void fit(const std::vector<std::vector<float>>& raw_windows, size_t window_size) {
        valid_indices.clear();
        input_dim = raw_windows[0].size() / window_size;
        
        std::cout << "\n=== Analyzing Feature Validity ===\n";
        std::cout << "Total features: " << input_dim << "\n";
        
        std::vector<bool> is_valid(input_dim, false);
        std::vector<int> nan_counts(input_dim, 0);
        std::vector<int> valid_counts(input_dim, 0);
        
        // Check each feature across all samples
        for (const auto& window : raw_windows) {
            for (size_t t = 0; t < window_size; ++t) {
                for (size_t d = 0; d < input_dim; ++d) {
                    float val = window[t * input_dim + d];
                    if (std::isnan(val) || std::isinf(val)) {
                        nan_counts[d]++;
                    } else {
                        valid_counts[d]++;
                        is_valid[d] = true;
                    }
                }
            }
        }
        
        // Only keep features that are valid in >95% of samples
        for (size_t d = 0; d < input_dim; ++d) {
            int total = nan_counts[d] + valid_counts[d];
            float valid_ratio = (float)valid_counts[d] / total;
            
            if (valid_ratio > 0.95f) {
                valid_indices.push_back(d);
            }
        }
        
        output_dim = valid_indices.size();
        
        std::cout << "Valid features found: " << output_dim << "\n";
        std::cout << "Feature indices: ";
        for (size_t i = 0; i < std::min(size_t(10), valid_indices.size()); ++i) {
            std::cout << valid_indices[i] << " ";
        }
        if (valid_indices.size() > 10) std::cout << "...";
        std::cout << "\n";
        
        // Save valid indices
        save_indices("valid_features.txt");
    }
    
    // Filter window to only valid features
    std::vector<float> filter_window(const std::vector<float>& raw_window, size_t window_size) const {
        std::vector<float> filtered;
        filtered.reserve(window_size * output_dim);
        
        for (size_t t = 0; t < window_size; ++t) {
            for (size_t idx : valid_indices) {
                float val = raw_window[t * input_dim + idx];
                // Replace any remaining NaN/inf with 0
                if (std::isnan(val) || std::isinf(val)) {
                    val = 0.0f;
                }
                filtered.push_back(val);
            }
        }
        
        return filtered;
    }
    
    void save_indices(const std::string& filename) const {
        std::ofstream file(filename);
        file << "# Valid feature indices (" << valid_indices.size() << " total)\n";
        for (size_t idx : valid_indices) {
            file << idx << "\n";
        }
    }
    
    bool load_indices(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) return false;
        
        valid_indices.clear();
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            valid_indices.push_back(std::stoul(line));
        }
        
        output_dim = valid_indices.size();
        std::cout << "Loaded " << output_dim << " valid feature indices\n";
        return true;
    }
    
    size_t get_output_dim() const { return output_dim; }
    const std::vector<size_t>& get_valid_indices() const { return valid_indices; }
};