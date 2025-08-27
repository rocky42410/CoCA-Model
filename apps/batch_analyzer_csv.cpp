#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include "../src/roca_one_class.hpp"

using namespace roca;

struct CSVData {
    std::vector<std::string> headers;
    std::vector<std::vector<float>> rows;
};

CSVData load_csv(const std::string& filename) {
    CSVData data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV: " + filename);
    }
    
    std::string line;
    
    // Read header
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            data.headers.push_back(cell);
        }
    }
    
    // Read data
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

std::vector<std::vector<float>> create_windows(const CSVData& data, 
                                               size_t window_size = 10,
                                               size_t stride = 5) {
    std::vector<std::vector<float>> windows;
    size_t num_features = data.headers.size() - 1; // Exclude timestamp
    
    for (size_t i = 0; i + window_size <= data.rows.size(); i += stride) {
        std::vector<float> window;
        
        for (size_t t = 0; t < window_size; ++t) {
            for (size_t f = 1; f < data.headers.size(); ++f) { // Skip timestamp
                float value = data.rows[i + t][f];
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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.roca> <data.csv> [threshold_multiplier]\n";
        return 1;
    }
    
    std::string model_file = argv[1];
    std::string csv_file = argv[2];
    float threshold_mult = argc > 3 ? std::stof(argv[3]) : 1.0f;
    
    try {
        // Create config for model loading
        OneClassConfig config;
        
        // Load model - need to create with config first
        OneClassRoCA model(config);
        if (!model.load_model(model_file)) {
            std::cerr << "Failed to load model\n";
            return 1;
        }
        
        // Get actual config after loading
        config = model.get_config();
        
        // Load normalization parameters
        std::string norm_file = model_file + ".norm";
        std::ifstream norm_in(norm_file);
        
        size_t num_features = config.D;
        size_t total_dim = config.T * config.D;
        std::vector<float> mean, std_dev;
        
        if (norm_in.is_open()) {
            size_t stored_features, stored_dim;
            norm_in >> stored_features >> stored_dim;
            
            // Skip feature indices
            std::string indices_line;
            std::getline(norm_in, indices_line);
            std::getline(norm_in, indices_line);
            
            mean.resize(stored_dim);
            std_dev.resize(stored_dim);
            
            for (size_t i = 0; i < stored_dim; ++i) {
                norm_in >> mean[i];
            }
            for (size_t i = 0; i < stored_dim; ++i) {
                norm_in >> std_dev[i];
            }
            
            norm_in.close();
            
            // Update dimensions if norm file has different size
            if (stored_dim > 0) {
                total_dim = stored_dim;
                num_features = stored_dim / config.T;
            }
        }
        
        // Load CSV data
        std::cout << "Loading CSV: " << csv_file << "\n";
        CSVData csv_data = load_csv(csv_file);
        std::cout << "Loaded " << csv_data.rows.size() << " rows\n";
        
        // Create windows
        auto windows = create_windows(csv_data, config.T, 5);
        std::cout << "Created " << windows.size() << " windows\n\n";
        
        // Normalize if parameters available
        if (!mean.empty() && mean.size() == windows[0].size()) {
            for (auto& w : windows) {
                for (size_t i = 0; i < w.size(); ++i) {
                    w[i] = (w[i] - mean[i]) / std_dev[i];
                    w[i] = std::max(-5.0f, std::min(5.0f, w[i]));
                }
            }
        }
        
        // Analyze windows using anomaly_score method
        float threshold = model.anomaly_threshold * threshold_mult;
        std::vector<float> scores;
        int anomaly_count = 0;
        
        std::cout << "=== Anomaly Detection Results ===\n";
        std::cout << "Threshold: " << threshold << "\n\n";
        
        for (size_t i = 0; i < windows.size(); ++i) {
            float score = model.anomaly_score(windows[i]);
            scores.push_back(score);
            
            bool is_anomaly = score > threshold;
            if (is_anomaly) {
                anomaly_count++;
                
                // Calculate timestamp from window start
                float timestamp_ms = csv_data.rows[i * 5][0]; // stride = 5
                
                std::cout << "ANOMALY at " << timestamp_ms << "ms";
                std::cout << " - Score: " << score;
                std::cout << " (" << (score / threshold * 100) << "% of threshold)\n";
            }
        }
        
        // Statistics
        float mean_score = std::accumulate(scores.begin(), scores.end(), 0.0f) / scores.size();
        float max_score = *std::max_element(scores.begin(), scores.end());
        float min_score = *std::min_element(scores.begin(), scores.end());
        
        std::cout << "\n=== Summary Statistics ===\n";
        std::cout << "Total windows: " << windows.size() << "\n";
        std::cout << "Anomalies detected: " << anomaly_count;
        std::cout << " (" << (100.0f * anomaly_count / windows.size()) << "%)\n";
        std::cout << "Score range: [" << min_score << ", " << max_score << "]\n";
        std::cout << "Mean score: " << mean_score << "\n";
        std::cout << "Threshold: " << threshold << "\n";
        
        // Save results
        std::string output_file = csv_file + ".analysis.csv";
        std::ofstream out(output_file);
        out << "window_idx,timestamp_ms,score,is_anomaly\n";
        
        for (size_t i = 0; i < scores.size(); ++i) {
            float timestamp_ms = csv_data.rows[i * 5][0];
            out << i << "," << timestamp_ms << "," << scores[i] << ",";
            out << (scores[i] > threshold ? "1" : "0") << "\n";
        }
        out.close();
        
        std::cout << "\nResults saved to: " << output_file << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}