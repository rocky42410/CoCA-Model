#include <iostream>
#include "../src/roca_one_class.hpp"
#include "../src/io/binary_log.hpp"

using namespace roca;

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    
    // Load valid features list
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
    
    // Get only first 5000 frames (100 seconds) to ensure consistency
    std::vector<std::vector<float>> windows;
    AutoencoderFrame frame;
    std::vector<AutoencoderFrame> frames;
    
    for (int i = 0; i < 5000 && reader.read_frame(frame); ++i) {
        frames.push_back(frame);
    }
    reader.close();
    
    std::cout << "Using first " << frames.size() << " frames only\n";
    
    // Create windows with valid features only
    for (size_t i = 0; i + 10 <= frames.size(); i += 5) {
        std::vector<float> window;
        for (size_t t = 0; t < 10; ++t) {
            for (size_t idx : valid_features) {
                window.push_back(frames[i+t].features[idx]);
            }
        }
        windows.push_back(window);
    }
    
    std::cout << "Created " << windows.size() << " windows\n";
    
    // Robust normalization with percentiles
    size_t total_features = valid_features.size() * 10;
    std::vector<float> medians(total_features);
    std::vector<float> scales(total_features);
    
    for (size_t i = 0; i < total_features; ++i) {
        std::vector<float> values;
        for (const auto& w : windows) {
            values.push_back(w[i]);
        }
        
        std::sort(values.begin(), values.end());
        medians[i] = values[values.size() / 2];
        
        // Use IQR for scale
        float q1 = values[values.size() / 4];
        float q3 = values[3 * values.size() / 4];
        scales[i] = q3 - q1;
        
        if (scales[i] < 1e-6) scales[i] = 1.0f;
    }
    
    // Normalize
    for (auto& w : windows) {
        for (size_t i = 0; i < total_features; ++i) {
            w[i] = (w[i] - medians[i]) / scales[i];
            w[i] = std::max(-2.0f, std::min(2.0f, w[i]));
        }
    }
    
    // Use MUCH simpler model
    OneClassConfig config;
    config.T = 10;
    config.D = valid_features.size();
    
    // Smaller architecture
    config.C = 16;  // Smaller latent space
    config.K = 8;   // Smaller projection
    
    // Adjusted hyperparameters
    config.epochs = 50;
    config.lr = 1e-4;  // Much smaller learning rate
    config.lambda_rec = 1.0f;
    config.lambda_oc = 0.01f;  // Much less one-class loss
    config.lambda_aug = 0.0f;   // No augmentation for now
    config.lambda_var = 0.0f;   // No variance reg
    
    std::cout << "\nTraining simplified model...\n";
    
    OneClassRoCA model(config);
    train_one_class_model(model, windows, config);
    
    return 0;
}