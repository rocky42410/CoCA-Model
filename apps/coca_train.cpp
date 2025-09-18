// ============================================================================
// apps/coca_train.cpp - Main COCA training application with CSV support
// ============================================================================
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "../src/coca_model.hpp"
#include "../src/io/csv_reader.hpp"
#include "../src/utils/config_parser.hpp"
#include "../src/utils/model_io.hpp"

using namespace coca;

// ============================================================================
// Main application
// ============================================================================
int main(int argc, char** argv) {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║        COCA Training Application       ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    // Parse command line arguments
    std::string data_file = "";
    std::string config_file = "coca_config.yaml";
    std::string output_model = "trained_model.coca";
    size_t window_size = 10;
    size_t window_stride = 5;
    bool skip_header = true;
    bool skip_timestamp = true;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--data" || arg == "--csv") && i + 1 < argc) {
            data_file = argv[++i];
        } else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_model = argv[++i];
        } else if (arg == "--window" && i + 1 < argc) {
            window_size = std::stoi(argv[++i]);
        } else if (arg == "--stride" && i + 1 < argc) {
            window_stride = std::stoi(argv[++i]);
        } else if (arg == "--no-header") {
            skip_header = false;
        } else if (arg == "--no-timestamp") {
            skip_timestamp = false;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " --csv <file> [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --csv <file>      Input CSV file (required)\n";
            std::cout << "  --config <file>   Configuration file (default: coca_config.yaml)\n";
            std::cout << "  --output <file>   Output model file (default: trained_model.coca)\n";
            std::cout << "  --window <size>   Window size (default: 10)\n";
            std::cout << "  --stride <size>   Window stride (default: 5)\n";
            std::cout << "  --no-header       CSV has no header row\n";
            std::cout << "  --no-timestamp    Don't skip first column\n";
            return 0;
        }
    }
    
    if (data_file.empty()) {
        std::cerr << "Error: Please specify data file with --csv or --data\n";
        return 1;
    }
    
    // Load configuration
    COCAConfig config;
    if (!ConfigParser::load_config(config_file, config)) {
        std::cout << "Using default configuration\n";
    }
    
    // Update window size from command line
    config.T = window_size;
    
    // Load CSV data
    std::cout << "Loading CSV data from: " << data_file << "\n";
    std::cout << "  Skip header: " << (skip_header ? "yes" : "no") << "\n";
    std::cout << "  Skip timestamp: " << (skip_timestamp ? "yes" : "no") << "\n\n";
    
    CSVReader reader;
    if (!reader.load(data_file, skip_header, skip_timestamp, true)) {
        std::cerr << "Error: Failed to load CSV file\n";
        return 1;
    }
    
    // Update config with actual feature count
    config.D = reader.get_feature_count();
    
    std::cout << "\nConfiguration:\n";
    std::cout << "  Window size: " << config.T << "\n";
    std::cout << "  Window stride: " << window_stride << "\n";
    std::cout << "  Features: " << config.D << "\n";
    std::cout << "  Latent dim: " << config.C << "\n";
    std::cout << "  Projection dim: " << config.K << "\n";
    std::cout << "  λ_rec: " << config.lambda_rec << "\n";
    std::cout << "  λ_inv: " << config.lambda_inv << "\n";
    std::cout << "  λ_var: " << config.lambda_var << "\n";
    std::cout << "  ζ: " << config.zeta << "\n";
    std::cout << "  Score mix: " << config.score_mix << "\n";
    std::cout << "  Threshold mode: " << config.threshold_mode << "\n";
    std::cout << "  Seed: " << config.seed << "\n\n";
    
    // Create windows from CSV data
    std::cout << "Creating training windows...\n";
    std::vector<std::vector<float>> windows = reader.get_windows(config.T, window_stride, true);
    
    if (windows.empty()) {
        std::cerr << "Error: No windows created from data\n";
        return 1;
    }
    
    if (windows.size() < 100) {
        std::cerr << "Warning: Very few training windows (" << windows.size() << ")\n";
        std::cerr << "Consider using smaller stride or longer recordings\n";
    }
    
    // Train model
    std::cout << "\n--- Starting Training ---\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    COCAModel model(config);
    train_coca_model(model, windows, config);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\nTraining time: " << duration.count() << " seconds\n";
    
    // Save model
    std::cout << "\nSaving model to: " << output_model << "\n";
    
    ModelIO::save_model(model, output_model);
    
    // Save configuration summary
    std::ofstream summary("training_summary.txt");
    summary << "# COCA Training Summary\n";
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    summary << "# " << std::ctime(&time_t) << "\n";
    summary << "Data file: " << data_file << "\n";
    summary << "Samples: " << reader.get_sample_count() << "\n";
    summary << "Features: " << config.D << "\n";
    summary << "Training windows: " << windows.size() << "\n";
    summary << "Window size: " << config.T << "\n";
    summary << "Window stride: " << window_stride << "\n";
    summary << "Latent dim: " << config.C << "\n";
    summary << "Projection dim: " << config.K << "\n";
    summary << "Final threshold: " << model.anomaly_threshold << "\n";
    summary << "Training time: " << duration.count() << " seconds\n";
    summary.close();
    
    // Save configuration used
    ConfigParser::save_config("training_config_used.yaml", config);
    
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║         TRAINING COMPLETE              ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    std::cout << "Outputs:\n";
    std::cout << "  Model: " << output_model << "\n";
    std::cout << "  Log: training_log.csv\n";
    std::cout << "  Summary: training_summary.txt\n";
    std::cout << "  Config used: training_config_used.yaml\n";
    
    return 0;
}

