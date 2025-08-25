// ============================================================================
// train_one_class.cpp - Train RoCA model on NORMAL data only
// 
// This implements self-supervised one-class anomaly detection.
// The model is trained ONLY on normal (idle) robot data.
// ============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>

#include "../src/roca_one_class.hpp"
#include "../src/model_io_complete.hpp"
#include "../src/io/binary_log.hpp"
#include "../src/data/window_maker.hpp"

using namespace roca;

int main(int argc, char** argv) {
    std::cout << "\n╔══════════════════════════════════════════════════╗\n";
    std::cout << "║   RoCA Self-Supervised One-Class Training        ║\n";
    std::cout << "║                                                  ║\n";
    std::cout << "║   Training on NORMAL data only                  ║\n";
    std::cout << "║   No anomalies required for training!           ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n\n";
    
    // Parse arguments
    std::string normal_data_file = "";
    std::string output_model = "one_class_model.roca";
    bool verbose = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--normal-data" && i + 1 < argc) {
            normal_data_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_model = argv[++i];
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " --normal-data <file> [options]\n\n";
            std::cout << "This trains a self-supervised one-class anomaly detector.\n";
            std::cout << "IMPORTANT: Only provide NORMAL data for training!\n\n";
            std::cout << "Options:\n";
            std::cout << "  --normal-data <file>  Binary log of NORMAL behavior only\n";
            std::cout << "  --output <file>       Output model file (default: one_class_model.roca)\n";
            std::cout << "  --verbose            Show detailed training progress\n";
            return 0;
        }
    }
    
    if (normal_data_file.empty()) {
        std::cerr << "Error: Please specify normal training data with --normal-data\n";
        std::cerr << "Remember: Only use NORMAL data (e.g., idle robot logs)\n";
        return 1;
    }
    
    // Verify this is normal data
    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "⚠️  ONE-CLASS TRAINING REQUIREMENT\n";
    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "You are training a one-class model.\n";
    std::cout << "The file '" << normal_data_file << "' should contain:\n";
    std::cout << "  ✓ ONLY normal/idle robot behavior\n";
    std::cout << "  ✗ NO anomalies, faults, or unusual behaviors\n";
    std::cout << "═══════════════════════════════════════════\n\n";
    
    // Load NORMAL data
    std::cout << "Loading normal training data...\n";
    std::ifstream file(normal_data_file, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file\n";
        return 1;
    }
    
    BinaryFileHeader header;
    if (!read_header(file, header)) {
        std::cerr << "Error: Invalid header\n";
        return 1;
    }
    
    std::cout << "Data info:\n";
    std::cout << "  Features: " << header.feature_count << "\n";
    std::cout << "  Fusion rate: " << header.fusion_rate_hz << " Hz\n";
    
    // Create windows from NORMAL data
    std::vector<std::vector<float>> normal_windows;
    WindowConfig window_cfg;
    window_cfg.T = 10;  // 10 frames per window
    window_cfg.stride = 5;
    window_cfg.D = header.feature_count;
    
    WindowMaker maker(window_cfg);
    AutoencoderFrame frame;
    size_t frame_count = 0;
    
    while (read_frame(file, frame)) {
        frame_count++;
        std::vector<float> features(header.feature_count);
        for (size_t i = 0; i < header.feature_count; ++i) {
            features[i] = frame.features[i];
        }
        
        maker.push(features);
        if (maker.ready()) {
            normal_windows.push_back(maker.get_window());
        }
    }
    
    std::cout << "Loaded " << frame_count << " frames\n";
    std::cout << "Created " << normal_windows.size() << " normal training windows\n";
    
    if (normal_windows.size() < 100) {
        std::cerr << "\n⚠️  Warning: Very few training windows.\n";
        std::cerr << "   Consider collecting more normal data for better results.\n";
    }
    
    // Configure one-class model
    OneClassConfig config;
    config.T = window_cfg.T;
    config.D = header.feature_count;
    config.C = 32;  // Latent dimension
    config.K = 16;  // One-class projection dimension
    config.epochs = 100;
    config.batch_size = 32;
    
    // Adjust for data size
    if (normal_windows.size() < 1000) {
        config.epochs = 50;  // Fewer epochs for small datasets
    }
    
    std::cout << "\nModel configuration:\n";
    std::cout << "  Input: " << config.T << " × " << config.D << "\n";
    std::cout << "  Latent dimension: " << config.C << "\n";
    std::cout << "  One-class space: " << config.K << "\n";
    std::cout << "  Training mode: Self-supervised\n";
    std::cout << "  Losses: Reconstruction + One-class + Augmentation\n";
    
    // Train one-class model
    OneClassRoCA model(config);
    train_one_class_model(model, normal_windows, config);
    
    // Test on training data to verify low false positives
    std::cout << "\n=== Verifying Model on Normal Data ===\n";
    
    int false_positives = 0;
    std::vector<float> normal_scores;
    
    for (size_t i = 0; i < std::min(size_t(100), normal_windows.size()); ++i) {
        float score = model.anomaly_score(normal_windows[i]);
        normal_scores.push_back(score);
        if (score > model.anomaly_threshold) {
            false_positives++;
        }
    }
    
    float fp_rate = 100.0f * false_positives / normal_scores.size();
    std::cout << "False positive rate on normal data: " << fp_rate << "%\n";
    
    if (fp_rate > 10.0f) {
        std::cout << "⚠️  High false positive rate. Consider:\n";
        std::cout << "   - Collecting more diverse normal data\n";
        std::cout << "   - Adjusting the threshold multiplier\n";
    } else {
        std::cout << "✅ Low false positive rate - model learned normal behavior well!\n";
    }
    
    // Save model
    std::cout << "\n=== Saving One-Class Model ===\n";
    if (model.save_model(output_model)) {
        std::cout << "✅ Model saved to: " << output_model << "\n";
        std::cout << "   File size: " << std::filesystem::file_size(output_model) / 1024 
                  << " KB\n";
    } else {
        std::cerr << "❌ Failed to save model\n";
        return 1;
    }
    
    std::cout << "\n╔══════════════════════════════════════════════════╗\n";
    std::cout << "║         ONE-CLASS TRAINING COMPLETE              ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n\n";
    
    std::cout << "The model has learned the NORMAL behavior distribution.\n";
    std::cout << "It will detect ANY deviation from normal as anomalous:\n";
    std::cout << "  • Movement (walking, standing, turning)\n";
    std::cout << "  • Sensor failures\n";
    std::cout << "  • Unexpected states\n";
    std::cout << "  • Any behavior not seen during training\n\n";
    
    std::cout << "To test the model:\n";
    std::cout << "  ./one_class_test --model " << output_model << " --test anomaly_data.bin\n";
    
    return 0;
}