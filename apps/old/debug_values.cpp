#include <iostream>
#include <vector>
#include <cmath>
#include "../src/io/binary_log.hpp"

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    
    // Load valid features
    std::vector<size_t> valid_features;
    std::ifstream feat_file("valid_features.txt");
    std::string line;
    while (std::getline(feat_file, line)) {
        if (line[0] != '#') {
            valid_features.push_back(std::stoul(line));
        }
    }
    
    BinaryLogReader reader;
    reader.open(argv[1]);
    
    // Get first window
    std::vector<AutoencoderFrame> frames;
    AutoencoderFrame frame;
    for (int i = 0; i < 10 && reader.read_frame(frame); ++i) {
        frames.push_back(frame);
    }
    
    std::cout << "=== First Window Raw Values ===\n";
    std::cout << "Showing first 10 valid features across 10 timesteps:\n\n";
    
    for (size_t f = 0; f < std::min(size_t(10), valid_features.size()); ++f) {
        std::cout << "Feature " << valid_features[f] << ": ";
        for (size_t t = 0; t < 10; ++t) {
            float val = frames[t].features[valid_features[f]];
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
    
    // Check for zeros
    int zero_count = 0;
    int nan_count = 0;
    int normal_count = 0;
    
    for (const auto& fr : frames) {
        for (size_t idx : valid_features) {
            float val = fr.features[idx];
            if (std::isnan(val)) nan_count++;
            else if (val == 0.0f) zero_count++;
            else normal_count++;
        }
    }
    
    std::cout << "\n=== Value Distribution ===\n";
    std::cout << "Zeros: " << zero_count << "/" << (10 * valid_features.size()) << "\n";
    std::cout << "NaNs: " << nan_count << "/" << (10 * valid_features.size()) << "\n";
    std::cout << "Non-zero values: " << normal_count << "/" << (10 * valid_features.size()) << "\n";
    
    // Calculate simple reconstruction error manually
    std::cout << "\n=== Manual Reconstruction Test ===\n";
    
    // Take mean of each feature as "reconstruction"
    std::vector<float> feature_means(valid_features.size(), 0);
    for (const auto& fr : frames) {
        for (size_t i = 0; i < valid_features.size(); ++i) {
            feature_means[i] += fr.features[valid_features[i]];
        }
    }
    for (auto& m : feature_means) m /= 10;
    
    // Calculate MSE
    float mse = 0;
    for (const auto& fr : frames) {
        for (size_t i = 0; i < valid_features.size(); ++i) {
            float diff = fr.features[valid_features[i]] - feature_means[i];
            mse += diff * diff;
        }
    }
    mse /= (10 * valid_features.size());
    
    std::cout << "MSE if reconstructing with mean: " << mse << "\n";
    std::cout << "This is the baseline - model should beat this\n";
    
    return 0;
}