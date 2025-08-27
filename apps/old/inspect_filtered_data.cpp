#include <iostream>
#include <vector>
#include <cmath>
#include "../src/io/binary_log.hpp"

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    
    // Load valid features from previous run
    std::vector<size_t> valid_features;
    std::ifstream feat_file("valid_features.txt");
    std::string line;
    while (std::getline(feat_file, line)) {
        if (line[0] != '#') {
            valid_features.push_back(std::stoul(line));
        }
    }
    
    std::cout << "Using " << valid_features.size() << " valid features\n\n";
    
    BinaryLogReader reader;
    reader.open(argv[1]);
    
    // Read first 1000 frames to analyze
    std::vector<std::vector<float>> frames;
    AutoencoderFrame frame;
    
    for (int i = 0; i < 1000 && reader.read_frame(frame); ++i) {
        std::vector<float> valid_only;
        for (size_t idx : valid_features) {
            valid_only.push_back(frame.features[idx]);
        }
        frames.push_back(valid_only);
    }
    
    // Analyze variance of each feature
    std::cout << "=== Feature Variance Analysis (for IDLE data) ===\n";
    std::cout << "High variance in idle data indicates either:\n";
    std::cout << "1. The robot is not actually idle\n";
    std::cout << "2. Sensor noise/drift\n\n";
    
    int high_variance_count = 0;
    
    for (size_t f = 0; f < valid_features.size(); ++f) {
        std::vector<float> values;
        for (const auto& frame : frames) {
            values.push_back(frame[f]);
        }
        
        float mean = 0;
        for (float v : values) mean += v;
        mean /= values.size();
        
        float variance = 0;
        for (float v : values) {
            variance += (v - mean) * (v - mean);
        }
        variance /= values.size();
        float std_dev = std::sqrt(variance);
        
        // For idle robot, std should be very small
        if (std_dev > 0.1 && std::abs(mean) < 100) {  // Exclude already corrupt features
            high_variance_count++;
            std::cout << "Feature " << valid_features[f] << " (idx " << f << "):\n";
            std::cout << "  Mean: " << mean << ", Std: " << std_dev << "\n";
            
            // Check if it's oscillating
            int direction_changes = 0;
            for (size_t i = 1; i < values.size() - 1; ++i) {
                bool increasing_before = values[i] > values[i-1];
                bool increasing_after = values[i+1] > values[i];
                if (increasing_before != increasing_after) {
                    direction_changes++;
                }
            }
            
            if (direction_changes > values.size() * 0.3) {
                std::cout << "  WARNING: High frequency oscillation detected!\n";
            }
        }
    }
    
    std::cout << "\nSummary: " << high_variance_count << "/" << valid_features.size() 
              << " features have high variance\n";
    
    if (high_variance_count > valid_features.size() * 0.3) {
        std::cout << "\n⚠️  This doesn't look like idle data!\n";
        std::cout << "Possible issues:\n";
        std::cout << "1. Robot was moving/vibrating during capture\n";
        std::cout << "2. Sensor noise is very high\n";
        std::cout << "3. Data includes both idle and active periods\n";
    }
    
    // Check raw reconstruction difficulty
    std::cout << "\n=== Reconstruction Difficulty Test ===\n";
    
    // Create a simple window
    std::vector<float> test_window;
    for (int t = 0; t < 10; ++t) {
        for (size_t f = 0; f < valid_features.size(); ++f) {
            test_window.push_back(frames[t][f]);
        }
    }
    
    // Calculate how different each timestep is from the mean
    std::vector<float> timestep_means(10, 0);
    for (int t = 0; t < 10; ++t) {
        for (size_t f = 0; f < valid_features.size(); ++f) {
            timestep_means[t] += test_window[t * valid_features.size() + f];
        }
        timestep_means[t] /= valid_features.size();
    }
    
    float temporal_variance = 0;
    float mean_of_means = 0;
    for (float m : timestep_means) mean_of_means += m;
    mean_of_means /= 10;
    
    for (float m : timestep_means) {
        temporal_variance += (m - mean_of_means) * (m - mean_of_means);
    }
    
    std::cout << "Temporal variance across window: " << temporal_variance << "\n";
    std::cout << "For idle data, this should be near 0\n";
    
    return 0;
}