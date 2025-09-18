// ============================================================================
// io/csv_reader.hpp - Direct CSV reading for COCA
// ============================================================================
#pragma once
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace coca {

struct CSVConfig {
    bool has_header = true;
    bool has_timestamp = true;  // First column is timestamp
    char delimiter = ',';
    float default_value = 0.0f;  // Value for missing/invalid data
    float sample_rate = 50.0f;   // Hz
    size_t max_features = 256;   // Maximum features to use
};

class CSVReader {
private:
    CSVConfig config;
    std::vector<std::string> header;
    std::vector<std::vector<float>> data;
    size_t num_features = 0;
    size_t num_rows = 0;
    
public:
    CSVReader(const CSVConfig& cfg = CSVConfig()) : config(cfg) {}
    
    // Load entire CSV into memory with configurable options
    bool load(const std::string& filename, 
              bool skip_header = true, 
              bool skip_timestamp = true,
              bool verbose = true) {
        
        config.has_header = skip_header;
        config.has_timestamp = skip_timestamp;
        
        std::ifstream file(filename);
        if (!file) {
            if (verbose) std::cerr << "Error: Cannot open CSV file: " << filename << "\n";
            return false;
        }
        
        data.clear();
        std::string line;
        size_t line_num = 0;
        
        // Process header if present
        if (config.has_header && std::getline(file, line)) {
            parse_header(line);
            line_num++;
            if (verbose) std::cout << "Reading CSV header...\n";
        }
        
        // Read all data rows
        while (std::getline(file, line)) {
            line_num++;
            if (line.empty()) continue;
            
            auto row = parse_row(line, line_num);
            if (!row.empty()) {
                data.push_back(row);
            }
        }
        
        num_rows = data.size();
        
        if (verbose) {
            std::cout << "Loaded " << num_rows << " samples with " 
                      << num_features << " features each\n";
            print_stats(data);
        }
        
        return !data.empty();
    }
    
    // Get data as windows for training
    std::vector<std::vector<float>> get_windows(size_t window_size, 
                                                size_t stride = 1,
                                                bool verbose = false) const {
        std::vector<std::vector<float>> windows;
        
        if (num_rows < window_size) {
            if (verbose) {
                std::cerr << "Warning: Not enough samples (" << num_rows 
                          << ") for window size " << window_size << "\n";
            }
            return windows;
        }
        
        for (size_t i = 0; i <= num_rows - window_size; i += stride) {
            std::vector<float> window;
            window.reserve(window_size * num_features);
            
            for (size_t t = 0; t < window_size; ++t) {
                const auto& frame = data[i + t];
                window.insert(window.end(), frame.begin(), frame.end());
            }
            
            windows.push_back(window);
        }
        
        if (verbose) {
            std::cout << "Created " << windows.size() << " windows of size " 
                      << window_size << " with stride " << stride << "\n";
        }
        
        return windows;
    }
    
    // Compatibility with existing load method
    std::vector<std::vector<float>> load(const std::string& filename) {
        if (load(filename, config.has_header, config.has_timestamp, true)) {
            return data;
        }
        return std::vector<std::vector<float>>();
    }
    
    // Stream CSV data (for large files)
    class CSVIterator {
    private:
        std::ifstream file;
        CSVReader* reader;
        std::string current_line;
        size_t line_num = 0;
        
    public:
        CSVIterator(const std::string& filename, CSVReader* r) 
            : file(filename), reader(r) {
            if (!file) {
                throw std::runtime_error("Cannot open CSV file: " + filename);
            }
            
            // Skip header if present
            if (reader->config.has_header && std::getline(file, current_line)) {
                reader->parse_header(current_line);
                line_num++;
            }
            
            // Read first data line
            advance();
        }
        
        bool has_next() const {
            return file.good() && !current_line.empty();
        }
        
        std::vector<float> next() {
            auto row = reader->parse_row(current_line, line_num);
            advance();
            return row;
        }
        
    private:
        void advance() {
            if (std::getline(file, current_line)) {
                line_num++;
            } else {
                current_line.clear();
            }
        }
    };
    
    // Get iterator for streaming
    CSVIterator get_iterator(const std::string& filename) {
        return CSVIterator(filename, this);
    }
    
    // Get metadata - updated names to match what apps expect
    size_t get_feature_count() const { return num_features; }
    size_t get_sample_count() const { return num_rows; }
    size_t get_num_features() const { return num_features; }
    size_t get_num_rows() const { return num_rows; }
    const std::vector<std::string>& get_header() const { return header; }
    const std::vector<std::vector<float>>& get_data() const { return data; }
    
private:
    void parse_header(const std::string& line) {
        header.clear();
        std::stringstream ss(line);
        std::string col;
        
        while (std::getline(ss, col, config.delimiter)) {
            // Trim whitespace
            col.erase(0, col.find_first_not_of(" \t\r\n"));
            col.erase(col.find_last_not_of(" \t\r\n") + 1);
            header.push_back(col);
        }
        
        // Determine feature count (skip timestamp if present)
        size_t start_col = config.has_timestamp ? 1 : 0;
        num_features = std::min(header.size() - start_col, config.max_features);
        
        std::cout << "CSV Header: " << header.size() << " columns, " 
                  << num_features << " features\n";
    }
    
    std::vector<float> parse_row(const std::string& line, size_t line_num) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;
        size_t col = 0;
        
        while (std::getline(ss, value, config.delimiter)) {
            // Skip timestamp column if configured
            if (config.has_timestamp && col == 0) {
                col++;
                continue;
            }
            
            // Stop at max features
            if (row.size() >= config.max_features) {
                break;
            }
            
            // Parse value
            float val = parse_value(value, line_num, col);
            row.push_back(val);
            col++;
        }
        
        // Set initial feature count from first row
        if (num_features == 0 && !row.empty()) {
            num_features = row.size();
            std::cout << "Detected " << num_features << " features from data\n";
        }
        
        // Pad or warn if inconsistent size
        if (row.size() != num_features) {
            if (row.size() < num_features) {
                // Pad with default values
                row.resize(num_features, config.default_value);
            } else {
                // Truncate
                row.resize(num_features);
            }
        }
        
        return row;
    }
    
    float parse_value(const std::string& str, size_t line_num, size_t col) {
        // Trim whitespace and quotes
        std::string value = str;
        value.erase(0, value.find_first_not_of(" \t\r\n\""));
        value.erase(value.find_last_not_of(" \t\r\n\"") + 1);
        
        // Handle empty values
        if (value.empty()) {
            return config.default_value;
        }
        
        // Parse float
        try {
            float val = std::stof(value);
            
            // Handle special values
            if (std::isnan(val) || std::isinf(val)) {
                return config.default_value;
            }
            
            return val;
        } catch (const std::exception& e) {
            // Invalid value, use default
            if (line_num <= 10) {  // Only warn for first few lines
                std::cerr << "Warning: Invalid value '" << value 
                          << "' at line " << line_num << ", col " << col 
                          << " - using " << config.default_value << "\n";
            }
            return config.default_value;
        }
    }
    
    void print_stats(const std::vector<std::vector<float>>& data) {
        if (data.empty()) return;
        
        std::cout << "\nCSV Statistics:\n";
        std::cout << "  Rows: " << data.size() << "\n";
        std::cout << "  Features: " << num_features << "\n";
        
        if (config.has_timestamp) {
            std::cout << "  Timestamp: skipped (first column)\n";
        }
        
        // Analyze feature ranges
        std::vector<float> min_vals(num_features, 1e10f);
        std::vector<float> max_vals(num_features, -1e10f);
        std::vector<size_t> zero_counts(num_features, 0);
        
        for (const auto& row : data) {
            for (size_t i = 0; i < std::min(row.size(), num_features); ++i) {
                float val = row[i];
                min_vals[i] = std::min(min_vals[i], val);
                max_vals[i] = std::max(max_vals[i], val);
                if (std::abs(val) < 1e-10f) zero_counts[i]++;
            }
        }
        
        // Count constant and variable features
        size_t constant_features = 0;
        size_t zero_features = 0;
        
        for (size_t i = 0; i < num_features; ++i) {
            if (std::abs(max_vals[i] - min_vals[i]) < 1e-6f) {
                constant_features++;
            }
            if (zero_counts[i] == data.size()) {
                zero_features++;
            }
        }
        
        std::cout << "  Constant features: " << constant_features << "\n";
        std::cout << "  All-zero features: " << zero_features << "\n";
        std::cout << "  Variable features: " 
                  << (num_features - constant_features) << "\n";
        
        // Show sample of feature names if header exists
        if (!header.empty() && config.has_header) {
            std::cout << "\nSample features:\n";
            size_t start = config.has_timestamp ? 1 : 0;
            for (size_t i = 0; i < std::min(size_t(5), num_features); ++i) {
                if (start + i < header.size()) {
                    std::cout << "  [" << i << "] " << header[start + i] 
                              << ": [" << std::fixed << std::setprecision(3) 
                              << min_vals[i] << ", " << max_vals[i] << "]\n";
                }
            }
        }
    }
};

} // namespace coca
