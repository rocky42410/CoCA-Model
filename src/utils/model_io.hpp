// ============================================================================
// utils/model_io.hpp - Model serialization and deserialization
// ============================================================================
#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include "../coca_model.hpp"

namespace coca {

class ModelIO {
public:
    // Save model to binary file
    static bool save_model(const COCAModel& model, const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            return false;
        }
        
        // Write magic number and version
        uint32_t magic = 0x434F4341; // "COCA"
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // Write configuration
        write_config(file, model.config);
        
        // Write processor statistics
        write_vector(file, model.processor.get_mean());
        write_vector(file, model.processor.get_std());
        write_bool_vector(file, model.processor.get_constant_mask());
        
        // Write encoder layers
        size_t num_encoder = model.encoder_layers.size();
        file.write(reinterpret_cast<const char*>(&num_encoder), sizeof(num_encoder));
        for (const auto& layer : model.encoder_layers) {
            write_layer(file, layer);
        }
        
        // Write decoder layers
        size_t num_decoder = model.decoder_layers.size();
        file.write(reinterpret_cast<const char*>(&num_decoder), sizeof(num_decoder));
        for (const auto& layer : model.decoder_layers) {
            write_layer(file, layer);
        }
        
        // Write projector
        write_layer(file, model.projector);
        
        // Write center
        write_vector(file, model.Ce);
        
        // Write threshold
        file.write(reinterpret_cast<const char*>(&model.anomaly_threshold), sizeof(float));
        
        file.close();
        return true;
    }
    
    // Load model from binary file
    static bool load_model(COCAModel& model, const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            return false;
        }
        
        // Check magic number and version
        uint32_t magic, version;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        
        if (magic != 0x434F4341 || version != 1) {
            std::cerr << "Error: Invalid model file format (magic: " << std::hex << magic 
                      << ", version: " << version << ")\n";
            return false;
        }
        
        // Read configuration
        COCAConfig config;
        read_config(file, config);
        
        // Normalize config strings after loading
        normalize_config_strings(config);
        
        model = COCAModel(config);
        
        // Read processor statistics and restore them
        std::vector<float> mean, std_dev;
        std::vector<bool> const_mask;
        read_vector(file, mean);
        read_vector(file, std_dev);
        read_bool_vector(file, const_mask);
        
        // CRITICAL: Restore the processor statistics
        model.processor.set_stats(mean, std_dev, const_mask);
        
        // Read encoder layers
        size_t num_encoder;
        file.read(reinterpret_cast<char*>(&num_encoder), sizeof(num_encoder));
        model.encoder_layers.clear();
        for (size_t i = 0; i < num_encoder; ++i) {
            // Create dummy layer just for reading
            DenseLayer layer(1, 1);
            read_layer(file, layer);
            model.encoder_layers.push_back(layer);
        }
        
        // Read decoder layers
        size_t num_decoder;
        file.read(reinterpret_cast<char*>(&num_decoder), sizeof(num_decoder));
        model.decoder_layers.clear();
        for (size_t i = 0; i < num_decoder; ++i) {
            // Create dummy layer just for reading
            DenseLayer layer(1, 1);
            read_layer(file, layer);
            model.decoder_layers.push_back(layer);
        }
        
        // Read projector
        read_layer(file, model.projector);
        
        // Read center
        read_vector(file, model.Ce);
        
        // Read threshold
        file.read(reinterpret_cast<char*>(&model.anomaly_threshold), sizeof(float));
        
        file.close();
        return true;
    }
    
private:
    // Normalize config strings by removing inline comments and trimming
    static void normalize_config_strings(COCAConfig& config) {
        auto strip_inline_comment = [](std::string& s) {
            // Remove everything after '#'
            if (auto p = s.find('#'); p != std::string::npos) s.erase(p);
            // Trim spaces/tabs
            auto l = s.find_first_not_of(" \t\r\n");
            auto r = s.find_last_not_of(" \t\r\n");
            s = (l == std::string::npos) ? "" : s.substr(l, r - l + 1);
        };
        
        strip_inline_comment(config.score_mix);
        strip_inline_comment(config.threshold_mode);
        
        // Canonicalize values
        if (config.score_mix != "inv_only" && config.score_mix != "inv_plus_rec") {
            std::cerr << "Warning: Invalid score_mix '" << config.score_mix << "', defaulting to 'inv_only'\n";
            config.score_mix = "inv_only";
        }
        if (config.threshold_mode != "quantile" && config.threshold_mode != "zscore") {
            std::cerr << "Warning: Invalid threshold_mode '" << config.threshold_mode << "', defaulting to 'quantile'\n";
            config.threshold_mode = "quantile";
        }
    }
    
    static void write_config(std::ofstream& file, const COCAConfig& config) {
        file.write(reinterpret_cast<const char*>(&config.T), sizeof(config.T));
        file.write(reinterpret_cast<const char*>(&config.D), sizeof(config.D));
        file.write(reinterpret_cast<const char*>(&config.C), sizeof(config.C));
        file.write(reinterpret_cast<const char*>(&config.K), sizeof(config.K));
        file.write(reinterpret_cast<const char*>(&config.lambda_inv), sizeof(config.lambda_inv));
        file.write(reinterpret_cast<const char*>(&config.lambda_var), sizeof(config.lambda_var));
        file.write(reinterpret_cast<const char*>(&config.lambda_rec), sizeof(config.lambda_rec));
        file.write(reinterpret_cast<const char*>(&config.zeta), sizeof(config.zeta));
        file.write(reinterpret_cast<const char*>(&config.variance_epsilon), sizeof(config.variance_epsilon));
        
        // Write string fields
        size_t len = config.score_mix.length();
        file.write(reinterpret_cast<const char*>(&len), sizeof(len));
        file.write(config.score_mix.c_str(), len);
        
        len = config.threshold_mode.length();
        file.write(reinterpret_cast<const char*>(&len), sizeof(len));
        file.write(config.threshold_mode.c_str(), len);
    }
    
    static void read_config(std::ifstream& file, COCAConfig& config) {
        file.read(reinterpret_cast<char*>(&config.T), sizeof(config.T));
        file.read(reinterpret_cast<char*>(&config.D), sizeof(config.D));
        file.read(reinterpret_cast<char*>(&config.C), sizeof(config.C));
        file.read(reinterpret_cast<char*>(&config.K), sizeof(config.K));
        file.read(reinterpret_cast<char*>(&config.lambda_inv), sizeof(config.lambda_inv));
        file.read(reinterpret_cast<char*>(&config.lambda_var), sizeof(config.lambda_var));
        file.read(reinterpret_cast<char*>(&config.lambda_rec), sizeof(config.lambda_rec));
        file.read(reinterpret_cast<char*>(&config.zeta), sizeof(config.zeta));
        file.read(reinterpret_cast<char*>(&config.variance_epsilon), sizeof(config.variance_epsilon));
        
        // Read string fields
        size_t len;
        file.read(reinterpret_cast<char*>(&len), sizeof(len));
        config.score_mix.resize(len);
        file.read(&config.score_mix[0], len);
        
        file.read(reinterpret_cast<char*>(&len), sizeof(len));
        config.threshold_mode.resize(len);
        file.read(&config.threshold_mode[0], len);
    }
    
    static void write_vector(std::ofstream& file, const std::vector<float>& vec) {
        size_t size = vec.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
    }
    
    static void read_vector(std::ifstream& file, std::vector<float>& vec) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        vec.resize(size);
        file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
    }
    
    static void write_bool_vector(std::ofstream& file, const std::vector<bool>& vec) {
        size_t size = vec.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        for (bool b : vec) {
            uint8_t byte = b ? 1 : 0;
            file.write(reinterpret_cast<const char*>(&byte), 1);
        }
    }
    
    static void read_bool_vector(std::ifstream& file, std::vector<bool>& vec) {
        size_t size;
        file.read(reinterpret_cast<char*>(&size), sizeof(size));
        vec.resize(size);
        for (size_t i = 0; i < size; ++i) {
            uint8_t byte;
            file.read(reinterpret_cast<char*>(&byte), 1);
            vec[i] = (byte != 0);
        }
    }
    
    static void write_layer(std::ofstream& file, const DenseLayer& layer) {
        file.write(reinterpret_cast<const char*>(&layer.in_dim), sizeof(layer.in_dim));
        file.write(reinterpret_cast<const char*>(&layer.out_dim), sizeof(layer.out_dim));
        file.write(reinterpret_cast<const char*>(&layer.use_relu), sizeof(layer.use_relu));
        file.write(reinterpret_cast<const char*>(&layer.dropout_rate), sizeof(layer.dropout_rate));
        
        write_vector(file, layer.W);
        write_vector(file, layer.b);
        
        // Adam state
        file.write(reinterpret_cast<const char*>(&layer.adam_t), sizeof(layer.adam_t));
        write_vector(file, layer.mW);
        write_vector(file, layer.vW);
        write_vector(file, layer.mb);
        write_vector(file, layer.vb);
    }
    
    // CRITICAL FIX: read_layer must match write_layer exactly!
    static void read_layer(std::ifstream& file, DenseLayer& layer) {
        // First read the header fields (THIS WAS MISSING!)
        file.read(reinterpret_cast<char*>(&layer.in_dim), sizeof(layer.in_dim));
        file.read(reinterpret_cast<char*>(&layer.out_dim), sizeof(layer.out_dim));
        file.read(reinterpret_cast<char*>(&layer.use_relu), sizeof(layer.use_relu));
        file.read(reinterpret_cast<char*>(&layer.dropout_rate), sizeof(layer.dropout_rate));
        
        // Then read the vectors
        read_vector(file, layer.W);
        read_vector(file, layer.b);
        
        // Adam state
        file.read(reinterpret_cast<char*>(&layer.adam_t), sizeof(layer.adam_t));
        read_vector(file, layer.mW);
        read_vector(file, layer.vW);
        read_vector(file, layer.mb);
        read_vector(file, layer.vb);
    }
};

} // namespace coca
