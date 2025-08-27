#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "../src/roca_one_class.hpp"

using namespace roca;

struct CSVData {
    std::vector<std::string> headers;
    std::vector<std::vector<float>> rows;
    std::vector<size_t> feature_indices;  // Maps CSV columns to feature indices
};

// Parse CSV file
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
            
            // Extract feature index from header name
            if (cell.find("feature_") == 0) {
                size_t idx = std::stoul(cell.substr(8));
                data.feature_indices.push_back(idx);
            } else if (cell != "timestamp_ms") {
                // Try to match known feature names to indices
                // This mapping should match feature_config.csv
                if (cell == "accel_x_mean") data.feature_indices.push_back(0);
                else if (cell == "accel_x_std") data.feature_indices.push_back(1);
                else if (cell == "accel_y_mean") data.feature_indices.push_back(3);
                else if (cell == "accel_y_std") data.feature_indices.push_back(4);
                else if (cell == "accel_z_mean") data.feature_indices.push_back(6);
                else if (cell == "accel_z_std") data.feature_indices.push_back(7);
                else if (cell == "gyro_x_mean") data.feature_indices.push_back(9);
                else if (cell == "gyro_y_mean") data.feature_indices.push_back(12);
                else if (cell == "gyro_z_mean") data.feature_indices.push_back(15);
                else if (cell == "pos_x") data.feature_indices.push_back(18);
                else if (cell == "pos_y") data.feature_indices.push_back(19);
                else if (cell == "pos_z") data.feature_indices.push_back(20);
                else if (cell == "vel_x") data.feature_indices.push_back(21);
                else if (cell == "vel_y") data.feature_indices.push_back(22);
                else if (cell == "vel_z") data.feature_indices.push_back(23);
                else if (cell == "roll") data.feature_indices.push_back(24);
                else if (cell == "pitch") data.feature_indices.push_back(25);
                else if (cell == "yaw") data.feature_indices.push_back(26);
                else if (cell == "yaw_speed") data.feature_indices.push_back(27);
                else if (cell == "foot_force_0") data.feature_indices.push_back(28);
                else if (cell == "foot_force_1") data.feature_indices.push_back(29);
                else if (cell == "foot_force_2") data.feature_indices.push_back(30);
                else if (cell == "foot_force_3") data.feature_indices.push_back(31);
                else if (cell == "body_height") data.feature_indices.push_back(32);
                else if (cell == "mode") data.feature_indices.push_back(33);
                else if (cell == "gait_type") data.feature_indices.push_back(34);
                else if (cell == "error_code") data.feature_indices.push_back(35);
                else if (cell == "power_v_mean") data.feature_indices.push_back(36);
                else if (cell == "power_a_mean") data.feature_indices.push_back(38);
                else if (cell == "bit_flag") data.feature_indices.push_back(76);
                else if (cell == "uwb_distance") data.feature_indices.push_back(77);
                else if (cell == "range_front") data.feature_indices.push_back(81);
                else if (cell == "range_left") data.feature_indices.push_back(82);
                else if (cell == "range_right") data.feature_indices.push_back(83);
                else data.feature_indices.push_back(999); // Unknown feature
            }
        }
    }
    
    std::cout << "CSV has " << data.headers.size() << " columns\n";
    std::cout << "Found " << data.feature_indices.size() << " feature columns\n";
    
    // Read data rows
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;
        
        while (std::getline(ss, cell, ',')) {
            if (cell.empty()) {
                row.push_back(0.0f);  // Replace empty cells with 0
            } else {
                try {
                    row.push_back(std::stof(cell));
                } catch (...) {
                    row.push_back(0.0f);  // Invalid values become 0
                }
            }
        }
        
        if (row.size() == data.headers.size()) {
            data.rows.push_back(row);
        }
    }
    
    std::cout << "Loaded " << data.rows.size() << " data rows\n";
    
    return data;
}

// Create training windows from CSV data
std::vector<std::vector<float>> create_windows(const CSVData& data, 
                                               size_t window_size = 10, 
                                               size_t stride = 5) {
    std::vector<std::vector<float>> windows;
    
    // Skip timestamp column (index 0)
    size_t num_features = data.feature_indices.size();
    
    for (size_t i = 0; i + window_size <= data.rows.size(); i += stride) {
        std::vector<float> window;
        window.reserve(window_size * num_features);
        
        // For each timestep in the window
        for (size_t t = 0; t < window_size; ++t) {
            // For each feature (skip timestamp at column 0)
            for (size_t f = 0; f < num_features; ++f) {
                float value = data.rows[i + t][f + 1];  // +1 to skip timestamp
                
                // Handle NaN and inf
                if (std::isnan(value) || std::isinf(value)) {
                    value = 0.0f;
                }
                
                window.push_back(value);
            }
        }
        
        windows.push_back(window);
    }
    
    return windows;
}

// Data quality check
void analyze_data_quality(const std::vector<std::vector<float>>& windows, 
                          size_t num_features) {
    std::cout << "\n=== Data Quality Analysis ===\n";
    
    size_t window_size = windows[0].size() / num_features;
    
    // Check for constant features
    std::vector<bool> is_constant(num_features, true);
    std::vector<float> first_values(num_features);
    
    if (!windows.empty()) {
        for (size_t f = 0; f < num_features; ++f) {
            first_values[f] = windows[0][f];
        }
        
        for (const auto& window : windows) {
            for (size_t f = 0; f < num_features; ++f) {
                // Check first timestep of each feature
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
            std::cout << "Feature " << f << " is constant (value: " 
                     << first_values[f] << ")\n";
        }
    }
    
    std::cout << "Constant features: " << constant_count << "/" << num_features << "\n";
    
    // Check value ranges
    std::vector<float> min_vals(num_features, 1e9);
    std::vector<float> max_vals(num_features, -1e9);
    
    for (const auto& window : windows) {
        for (size_t t = 0; t < window_size; ++t) {
            for (size_t f = 0; f < num_features; ++f) {
                float val = window[t * num_features + f];
                min_vals[f] = std::min(min_vals[f], val);
                max_vals[f] = std::max(max_vals[f], val);
            }
        }
    }
    
    std::cout << "\nFeature ranges:\n";
    for (size_t f = 0; f < std::min(size_t(10), num_features); ++f) {
        std::cout << "  Feature " << f << ": [" << min_vals[f] 
                 << ", " << max_vals[f] << "]\n";
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data.csv> [output_model.roca]\n";
        return 1;
    }
    
    std::string csv_file = argv[1];
    std::string model_file = argc > 2 ? argv[2] : "model.roca";
    
    try {
        // Load CSV data
        std::cout << "Loading CSV file: " << csv_file << "\n";
        CSVData csv_data = load_csv(csv_file);
        
        if (csv_data.rows.size() < 100) {
            std::cerr << "Not enough data rows for training (need at least 100)\n";
            return 1;
        }
        
        // Create training windows
        std::cout << "\nCreating training windows...\n";
        auto windows = create_windows(csv_data, 10, 5);
        std::cout << "Created " << windows.size() << " windows\n";
        
        if (windows.size() < 100) {
            std::cerr << "Not enough windows for training\n";
            return 1;
        }
        
        size_t num_features = csv_data.feature_indices.size();
        
        // Analyze data quality
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
            if (s < 1e-6) s = 1.0f;  // Prevent division by zero
        }
        
        // Normalize
        for (auto& w : windows) {
            for (size_t i = 0; i < total_dim; ++i) {
                w[i] = (w[i] - mean[i]) / std[i];
                // Clip to reasonable range
                w[i] = std::max(-5.0f, std::min(5.0f, w[i]));
            }
        }
        
        // Check normalized data
        float max_val = -1e9, min_val = 1e9;
        for (const auto& w : windows) {
            for (float v : w) {
                max_val = std::max(max_val, v);
                min_val = std::min(min_val, v);
            }
        }
        std::cout << "Normalized data range: [" << min_val << ", " << max_val << "]\n";
        
        // Configure and train model
        OneClassConfig config;
        config.T = 10;  // Window size
        config.D = num_features;
        config.C = 32;  // Latent dimension
        config.K = 16;  // Projection dimension
        config.epochs = 100;
        config.batch_size = 32;
        config.lr = 1e-3;
        config.lambda_rec = 1.0f;
        config.lambda_oc = 0.5f;
        config.lambda_aug = 0.3f;
        config.lambda_var = 0.1f;
        
        std::cout << "\n=== Training Configuration ===\n";
        std::cout << "Features: " << num_features << "\n";
        std::cout << "Window size: " << config.T << "\n";
        std::cout << "Latent dim: " << config.C << "\n";
        std::cout << "Training samples: " << windows.size() << "\n";
        
        // Train model
        OneClassRoCA model(config);
        train_one_class_model(model, windows, config);
        
        // Save model with metadata
        model.save_model(model_file);
        
        // Save normalization parameters
        std::string norm_file = model_file + ".norm";
        std::ofstream norm_out(norm_file);
        if (norm_out.is_open()) {
            norm_out << num_features << "\n";
            norm_out << total_dim << "\n";
            
            // Save feature indices mapping
            for (size_t idx : csv_data.feature_indices) {
                norm_out << idx << " ";
            }
            norm_out << "\n";
            
            // Save mean and std
            for (float m : mean) norm_out << m << " ";
            norm_out << "\n";
            for (float s : std) norm_out << s << " ";
            norm_out << "\n";
            
            norm_out.close();
            std::cout << "Saved normalization parameters to " << norm_file << "\n";
        }
        
        std::cout << "\nModel saved to " << model_file << "\n";
        std::cout << "Training complete!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}