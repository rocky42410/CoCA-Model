#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <filesystem>
#include "../src/roca_one_class.hpp"

using namespace roca;
namespace fs = std::filesystem;

struct CSVData {
    std::vector<std::string> headers;
    std::vector<std::vector<float>> rows;
    std::vector<size_t> feature_indices;
};

CSVData load_csv(const std::string& filename) {
    CSVData data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + filename);
    }
    
    std::string line;
    
    // Read header
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            data.headers.push_back(cell);
            
            // Map headers to feature indices (same as before)
            if (cell.find("feature_") == 0) {
                size_t idx = std::stoul(cell.substr(8));
                data.feature_indices.push_back(idx);
            } else if (cell != "timestamp_ms") {
                // Known feature name mappings
                if (cell == "accel_x_mean") data.feature_indices.push_back(0);
                else if (cell == "accel_x_std") data.feature_indices.push_back(1);
                // ... (rest of mappings as before)
                else data.feature_indices.push_back(999);
            }
        }
    }
    
    // Read data rows
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            if (cell.empty()) {
                row.push_back(0.0f);
            } else {
                try {
                    row.push_back(std::stof(cell));
                } catch (...) {
                    row.push_back(0.0f);
                }
            }
        }
        
        if (row.size() == data.headers.size()) {
            data.rows.push_back(row);
        }
    }
    
    return data;
}

// Load multiple CSV files and concatenate
std::vector<std::vector<float>> load_multiple_csvs(const std::vector<std::string>& csv_files,
                                                   size_t& num_features,
                                                   std::vector<size_t>& feature_indices) {
    std::vector<std::vector<float>> all_windows;
    
    // Verify all CSVs have same structure
    std::vector<std::string> reference_headers;
    bool first_file = true;
    
    for (const auto& csv_file : csv_files) {
        std::cout << "Loading: " << csv_file << "\n";
        
        CSVData data = load_csv(csv_file);
        
        if (first_file) {
            reference_headers = data.headers;
            feature_indices = data.feature_indices;
            num_features = data.feature_indices.size();
            first_file = false;
        } else {
            // Verify headers match
            if (data.headers != reference_headers) {
                std::cerr << "Warning: Headers don't match in " << csv_file << "\n";
                std::cerr << "Expected " << reference_headers.size() << " columns, got " 
                         << data.headers.size() << "\n";
                continue;
            }
        }
        
        std::cout << "  - Loaded " << data.rows.size() << " rows\n";
        
        // Create windows from this CSV
        size_t window_size = 10;
        size_t stride = 5;
        
        for (size_t i = 0; i + window_size <= data.rows.size(); i += stride) {
            std::vector<float> window;
            window.reserve(window_size * num_features);
            
            for (size_t t = 0; t < window_size; ++t) {
                for (size_t f = 0; f < num_features; ++f) {
                    float value = data.rows[i + t][f + 1]; // +1 to skip timestamp
                    
                    if (std::isnan(value) || std::isinf(value)) {
                        value = 0.0f;
                    }
                    
                    window.push_back(value);
                }
            }
            
            all_windows.push_back(window);
        }
    }
    
    return all_windows;
}

// Parse command line to get CSV files
std::vector<std::string> parse_csv_inputs(int argc, char** argv) {
    std::vector<std::string> csv_files;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        // Check if it's a CSV file
        if (arg.size() > 4 && arg.substr(arg.size() - 4) == ".csv") {
            csv_files.push_back(arg);
        }
        // Check if it's a directory
        else if (fs::is_directory(arg)) {
            std::cout << "Scanning directory: " << arg << "\n";
            for (const auto& entry : fs::directory_iterator(arg)) {
                if (entry.path().extension() == ".csv") {
                    csv_files.push_back(entry.path().string());
                }
            }
        }
        // Check if it's a wildcard pattern
        else if (arg.find('*') != std::string::npos) {
            fs::path parent_path = fs::path(arg).parent_path();
            std::string pattern = fs::path(arg).filename().string();
            
            // Simple wildcard matching (just * support)
            std::string prefix = pattern.substr(0, pattern.find('*'));
            std::string suffix = pattern.substr(pattern.find('*') + 1);
            
            for (const auto& entry : fs::directory_iterator(parent_path)) {
                std::string filename = entry.path().filename().string();
                if (filename.substr(0, prefix.size()) == prefix &&
                    filename.size() >= suffix.size() &&
                    filename.substr(filename.size() - suffix.size()) == suffix) {
                    csv_files.push_back(entry.path().string());
                }
            }
        }
    }
    
    return csv_files;
}

void analyze_data_quality(const std::vector<std::vector<float>>& windows, 
                          size_t num_features) {
    // Same implementation as before
    std::cout << "\n=== Data Quality Analysis ===\n";
    
    size_t window_size = windows[0].size() / num_features;
    
    std::vector<bool> is_constant(num_features, true);
    std::vector<float> first_values(num_features);
    
    if (!windows.empty()) {
        for (size_t f = 0; f < num_features; ++f) {
            first_values[f] = windows[0][f];
        }
        
        for (const auto& window : windows) {
            for (size_t f = 0; f < num_features; ++f) {
                if (std::abs(window[f] - first_values[f]) > 1e-6) {
                    is_constant[f] = false;
                }
            }
        }
    }
    
    int constant_count = 0;
    for (size_t f = 0; f < num_features; ++f) {
        if (is_constant[f]) {
            constant_count++;
        }
    }
    
    std::cout << "Constant features: " << constant_count << "/" << num_features << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <csv_files...> [--output model.roca]\n";
        std::cerr << "Examples:\n";
        std::cerr << "  " << argv[0] << " data1.csv data2.csv data3.csv\n";
        std::cerr << "  " << argv[0] << " /path/to/csv_dir/\n";
        std::cerr << "  " << argv[0] << " 'data/*.csv' --output combined_model.roca\n";
        return 1;
    }
    
    // Parse arguments
    std::vector<std::string> csv_files = parse_csv_inputs(argc, argv);
    std::string model_file = "model.roca";
    
    // Look for --output flag
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--output") {
            model_file = argv[i + 1];
            break;
        }
    }
    
    if (csv_files.empty()) {
        std::cerr << "No CSV files found!\n";
        return 1;
    }
    
    std::cout << "Found " << csv_files.size() << " CSV files to process:\n";
    for (const auto& f : csv_files) {
        std::cout << "  - " << f << "\n";
    }
    std::cout << "\n";
    
    try {
        // Load all CSV files
        size_t num_features;
        std::vector<size_t> feature_indices;
        auto windows = load_multiple_csvs(csv_files, num_features, feature_indices);
        
        std::cout << "\nTotal windows created: " << windows.size() << "\n";
        
        if (windows.size() < 100) {
            std::cerr << "Not enough windows for training (need at least 100)\n";
            return 1;
        }
        
        // Analyze combined data quality
        analyze_data_quality(windows, num_features);
        
        // Normalize data
        std::cout << "\nNormalizing data...\n";
        size_t total_dim = windows[0].size();
        std::vector<float> mean(total_dim, 0);
        std::vector<float> std(total_dim, 1);
        
        // Calculate mean
        for (const auto& w : windows) {
            for (size_t i = 0; i < total_dim; ++i) {
                mean[i] += w[i];
            }
        }
        for (auto& m : mean) m /= windows.size();
        
        // Calculate std
        for (const auto& w : windows) {
            for (size_t i = 0; i < total_dim; ++i) {
                float diff = w[i] - mean[i];
                std[i] += diff * diff;
            }
        }
        for (auto& s : std) {
            s = std::sqrt(s / windows.size());
            if (s < 1e-6) s = 1.0f;
        }
        
        // Normalize
        for (auto& w : windows) {
            for (size_t i = 0; i < total_dim; ++i) {
                w[i] = (w[i] - mean[i]) / std[i];
                w[i] = std::max(-5.0f, std::min(5.0f, w[i]));
            }
        }
        
        // Configure and train model
        OneClassConfig config;
        config.T = 10;
        config.D = num_features;
        config.C = 32;
        config.K = 16;
        config.epochs = 100;
        config.batch_size = 32;
        config.lr = 1e-3;
        
        std::cout << "\n=== Training Configuration ===\n";
        std::cout << "Features: " << num_features << "\n";
        std::cout << "Window size: " << config.T << "\n";
        std::cout << "Training samples: " << windows.size() << "\n";
        std::cout << "Input files: " << csv_files.size() << "\n";
        
        // Train model
        OneClassRoCA model(config);
        train_one_class_model(model, windows, config);
        
        // Save model and metadata
        model.save_model(model_file);
        
        // Save normalization parameters
        std::string norm_file = model_file + ".norm";
        std::ofstream norm_out(norm_file);
        if (norm_out.is_open()) {
            norm_out << num_features << "\n";
            norm_out << total_dim << "\n";
            
            for (size_t idx : feature_indices) {
                norm_out << idx << " ";
            }
            norm_out << "\n";
            
            for (float m : mean) norm_out << m << " ";
            norm_out << "\n";
            for (float s : std) norm_out << s << " ";
            norm_out << "\n";
            
            norm_out.close();
        }
        
        std::cout << "\nModel saved to " << model_file << "\n";
        std::cout << "Training complete!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}