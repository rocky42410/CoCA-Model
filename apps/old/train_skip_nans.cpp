#include <iostream>
#include "../src/roca_one_class.hpp"
#include "../src/io/binary_log.hpp"

using namespace roca;

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    
    BinaryLogReader reader;
    reader.open(argv[1]);
    
    // Skip first 100 frames to ensure sensors are initialized
    AutoencoderFrame frame;
    for (int i = 0; i < 100; ++i) {
        reader.read_frame(frame);
    }
    
    // Now read actual data
    std::vector<AutoencoderFrame> frames;
    while (reader.read_frame(frame) && frames.size() < 5000) {
        // Check if frame has valid data
        bool has_nan = false;
        for (int i = 0; i < 87; ++i) {  // Check first 87 features
            if (std::isnan(frame.features[i])) {
                has_nan = true;
                break;
            }
        }
        
        if (!has_nan) {
            frames.push_back(frame);
        }
    }
    reader.close();
    
    std::cout << "Collected " << frames.size() << " valid frames\n";
    
    // Use only features 0-86 (excluding all the always-NaN features)
    size_t num_features = 87;
    
    // Create windows WITHOUT NaN
    std::vector<std::vector<float>> windows;
    for (size_t i = 0; i + 10 <= frames.size(); i += 5) {
        std::vector<float> window;
        bool valid = true;
        
        for (size_t t = 0; t < 10; ++t) {
            for (size_t f = 0; f < num_features; ++f) {
                float val = frames[i+t].features[f];
                if (std::isnan(val) || std::abs(val) > 1000) {
                    valid = false;
                    break;
                }
                window.push_back(val);
            }
            if (!valid) break;
        }
        
        if (valid && window.size() == 10 * num_features) {
            windows.push_back(window);
        }
    }
    
    std::cout << "Created " << windows.size() << " clean windows\n";
    
    // Simple normalization
    std::vector<float> mean(10 * num_features, 0);
    std::vector<float> std(10 * num_features, 1);
    
    for (const auto& w : windows) {
        for (size_t i = 0; i < w.size(); ++i) {
            mean[i] += w[i];
        }
    }
    for (auto& m : mean) m /= windows.size();
    
    for (const auto& w : windows) {
        for (size_t i = 0; i < w.size(); ++i) {
            float diff = w[i] - mean[i];
            std[i] += diff * diff;
        }
    }
    for (auto& s : std) {
        s = std::sqrt(s / windows.size());
        if (s < 1e-6) s = 1.0f;
    }
    
    for (auto& w : windows) {
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = (w[i] - mean[i]) / std[i];
        }
    }
    
    // Train
    OneClassConfig config;
    config.T = 10;
    config.D = num_features;
    config.C = 32;
    config.K = 16;
    config.epochs = 100;
    config.lr = 1e-3;
    
    OneClassRoCA model(config);
    train_one_class_model(model, windows, config);
    
    return 0;
}