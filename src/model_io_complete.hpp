// ============================================================================
// model_io_complete.hpp - Complete model serialization/deserialization
// ============================================================================
#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

namespace roca {

struct ModelMetadata {
    uint32_t magic = 0x524F4341;  // "ROCA"
    uint32_t version = 1;
    size_t T;
    size_t D;
    size_t C;
    size_t K;
    float anomaly_threshold;
    size_t num_valid_features;
};

class ModelSerializer {
public:
    static bool save_model(const std::string& filepath,
                          const ModelMetadata& metadata,
                          const std::vector<float>& feature_mean,
                          const std::vector<float>& feature_std,
                          const std::vector<bool>& is_constant,
                          const std::vector<size_t>& valid_indices,
                          const std::vector<std::vector<float>>& encoder_weights,
                          const std::vector<std::vector<float>>& encoder_biases,
                          const std::vector<std::vector<float>>& decoder_weights,
                          const std::vector<std::vector<float>>& decoder_biases,
                          const std::vector<float>& projector_weights,
                          const std::vector<float>& projector_bias,
                          const std::vector<float>& Ce) {
        
        std::ofstream file(filepath, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot create file " << filepath << std::endl;
            return false;
        }
        
        // Write metadata
        file.write(reinterpret_cast<const char*>(&metadata), sizeof(metadata));
        
        // Write feature statistics
        write_vector(file, feature_mean);
        write_vector(file, feature_std);
        write_bool_vector(file, is_constant);
        write_size_vector(file, valid_indices);
        
        // Write encoder weights and biases (3 layers)
        for (const auto& weights : encoder_weights) {
            write_vector(file, weights);
        }
        for (const auto& biases : encoder_biases) {
            write_vector(file, biases);
        }
        
        // Write decoder weights and biases
        for (const auto& weights : decoder_weights) {
            write_vector(file, weights);
        }
        for (const auto& biases : decoder_biases) {
            write_vector(file, biases);
        }
        
        // Write projector weights and bias
        write_vector(file, projector_weights);
        write_vector(file, projector_bias);
        
        // Write center vector
        write_vector(file, Ce);
        
        file.close();
        
        std::cout << "Model saved to: " << filepath << std::endl;
        std::cout << "  File size: " << std::filesystem::file_size(filepath) / 1024 
                  << " KB" << std::endl;
        
        return true;
    }
    
    static bool load_model(const std::string& filepath,
                          ModelMetadata& metadata,
                          std::vector<float>& feature_mean,
                          std::vector<float>& feature_std,
                          std::vector<bool>& is_constant,
                          std::vector<size_t>& valid_indices,
                          std::vector<std::vector<float>>& encoder_weights,
                          std::vector<std::vector<float>>& encoder_biases,
                          std::vector<std::vector<float>>& decoder_weights,
                          std::vector<std::vector<float>>& decoder_biases,
                          std::vector<float>& projector_weights,
                          std::vector<float>& projector_bias,
                          std::vector<float>& Ce) {
        
        std::ifstream file(filepath, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot open file " << filepath << std::endl;
            return false;
        }
        
        // Read metadata
        file.read(reinterpret_cast<char*>(&metadata), sizeof(metadata));
        
        if (metadata.magic != 0x524F4341) {
            std::cerr << "Error: Invalid model file format" << std::endl;
            return false;
        }
        
        // Read feature statistics
        read_vector(file, feature_mean);
        read_vector(file, feature_std);
        read_bool_vector(file, is_constant);
        read_size_vector(file, valid_indices);
        
        // Read encoder weights and biases
        encoder_weights.resize(3);
        encoder_biases.resize(3);
        for (auto& weights : encoder_weights) {
            read_vector(file, weights);
        }
        for (auto& biases : encoder_biases) {
            read_vector(file, biases);
        }
        
        // Read decoder weights and biases
        decoder_weights.resize(1);
        decoder_biases.resize(1);
        for (auto& weights : decoder_weights) {
            read_vector(file, weights);
        }
        for (auto& biases : decoder_biases) {
            read_vector(file, biases);
        }
        
        // Read projector weights and bias
        read_vector(file, projector_weights);
        read_vector(file, projector_bias);
        
        // Read center vector
        read_vector(file, Ce);
        
        file.close();
        
        std::cout << "Model loaded from: " << filepath << std::endl;
        std::cout << "  T=" << metadata.T << ", D=" << metadata.D 
                  << ", C=" << metadata.C << ", K=" << metadata.K << std::endl;
        std::cout << "  Threshold: " << metadata.anomaly_threshold << std::endl;
        
        return true;
    }
    
private:
    static void write_vector(std::ofstream& file, const std::vector<float>& vec) {
        size_t size = vec.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
    }
    
    static void write_bool_vector(std::ofstream& file, const std::vector<bool>& vec) {
        size_t size = vec.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        for (bool val : vec) {
            char b = val ? 1 : 0;
            file.write(&b, 1);
        }
    }
    
    static void write_size_vector(std::ofstream& file, const std::vector<size_t>& vec) {
        size_t size = vec.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(size_t));
    }
    
    static void read_vector(std::ifstream& file, std::vector<float>& vec) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        vec.resize(size);
        file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
    }
    
    static void read_bool_vector(std::ifstream& file, std::vector<bool>& vec) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        vec.clear();
        vec.reserve(size);
        for (size_t i = 0; i < size; ++i) {
            char b;
            file.read(&b, 1);
            vec.push_back(b != 0);
        }
    }
    
    static void read_size_vector(std::ifstream& file, std::vector<size_t>& vec) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        vec.resize(size);
        file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(size_t));
    }
};

} // namespace roca