// ============================================================================
// utils/config_parser.hpp - YAML-style configuration parser for COCA
// ============================================================================
#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include "../coca_model.hpp"

namespace coca {

class ConfigParser {
public:
    static bool load_config(const std::string& filename, COCAConfig& config) {
        std::ifstream file(filename);
        if (!file) {
            return false;
        }
        
        std::map<std::string, std::string> params;
        std::string line;
        
        while (std::getline(file, line)) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') continue;
            
            // Parse key: value
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string key = trim(line.substr(0, colon_pos));
                std::string value = trim(line.substr(colon_pos + 1));
                params[key] = value;
            }
        }
        
        // Model architecture
        if (params.count("T")) config.T = std::stoi(params["T"]);
        if (params.count("D")) config.D = std::stoi(params["D"]);
        if (params.count("C")) config.C = std::stoi(params["C"]);
        if (params.count("K")) config.K = std::stoi(params["K"]);
        
        // Loss weights
        if (params.count("lambda_inv")) config.lambda_inv = std::stof(params["lambda_inv"]);
        if (params.count("lambda_var")) config.lambda_var = std::stof(params["lambda_var"]);
        if (params.count("lambda_rec")) config.lambda_rec = std::stof(params["lambda_rec"]);
        
        // Variance parameters
        if (params.count("zeta")) config.zeta = std::stof(params["zeta"]);
        if (params.count("variance_epsilon")) config.variance_epsilon = std::stof(params["variance_epsilon"]);
        
        // Center parameters
        if (params.count("center_warmup_epochs")) config.center_warmup_epochs = std::stoi(params["center_warmup_epochs"]);
        if (params.count("inv_ramp_epochs")) config.inv_ramp_epochs = std::stoi(params["inv_ramp_epochs"]);
        
        // Training parameters
        if (params.count("learning_rate")) config.learning_rate = std::stof(params["learning_rate"]);
        if (params.count("batch_size")) config.batch_size = std::stoi(params["batch_size"]);
        if (params.count("epochs")) config.epochs = std::stoi(params["epochs"]);
        if (params.count("val_split")) config.val_split = std::stof(params["val_split"]);
        
        // Score and threshold
        if (params.count("score_mix")) config.score_mix = params["score_mix"];
        if (params.count("threshold_mode")) config.threshold_mode = params["threshold_mode"];
        if (params.count("threshold_quantile")) config.threshold_quantile = std::stof(params["threshold_quantile"]);
        if (params.count("threshold_zscore_k")) config.threshold_zscore_k = std::stof(params["threshold_zscore_k"]);
        
        // Misc
        if (params.count("min_std")) config.min_std = std::stof(params["min_std"]);
        if (params.count("dropout_rate")) config.dropout_rate = std::stof(params["dropout_rate"]);
        if (params.count("seed")) config.seed = std::stoi(params["seed"]);
        
        // Early stopping
        if (params.count("early_stop_patience")) config.early_stop_patience = std::stoi(params["early_stop_patience"]);
        if (params.count("early_stop_min_delta")) config.early_stop_min_delta = std::stof(params["early_stop_min_delta"]);
        
        return true;
    }
    
    static void save_config(const std::string& filename, const COCAConfig& config) {
        std::ofstream file(filename);
        
        file << "# COCA Configuration\n";
        file << "# Model architecture\n";
        file << "T: " << config.T << "\n";
        file << "D: " << config.D << "\n";
        file << "C: " << config.C << "\n";
        file << "K: " << config.K << "\n";
        file << "\n";
        
        file << "# Loss weights\n";
        file << "lambda_inv: " << config.lambda_inv << "\n";
        file << "lambda_var: " << config.lambda_var << "\n";
        file << "lambda_rec: " << config.lambda_rec << "\n";
        file << "\n";
        
        file << "# Variance parameters\n";
        file << "zeta: " << config.zeta << "\n";
        file << "variance_epsilon: " << config.variance_epsilon << "\n";
        file << "\n";
        
        file << "# Center parameters\n";
        file << "center_warmup_epochs: " << config.center_warmup_epochs << "\n";
        file << "inv_ramp_epochs: " << config.inv_ramp_epochs << "\n";
        file << "\n";
        
        file << "# Training parameters\n";
        file << "learning_rate: " << config.learning_rate << "\n";
        file << "batch_size: " << config.batch_size << "\n";
        file << "epochs: " << config.epochs << "\n";
        file << "val_split: " << config.val_split << "\n";
        file << "\n";
        
        file << "# Score and threshold\n";
        file << "score_mix: " << config.score_mix << "\n";
        file << "threshold_mode: " << config.threshold_mode << "\n";
        file << "threshold_quantile: " << config.threshold_quantile << "\n";
        file << "threshold_zscore_k: " << config.threshold_zscore_k << "\n";
        file << "\n";
        
        file << "# Misc\n";
        file << "min_std: " << config.min_std << "\n";
        file << "dropout_rate: " << config.dropout_rate << "\n";
        file << "seed: " << config.seed << "\n";
        file << "\n";
        
        file << "# Early stopping\n";
        file << "early_stop_patience: " << config.early_stop_patience << "\n";
        file << "early_stop_min_delta: " << config.early_stop_min_delta << "\n";
    }
    
private:
    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\n\r");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\n\r");
        return str.substr(first, (last - first + 1));
    }
};

} // namespace coca
