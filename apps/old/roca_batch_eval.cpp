// ============================================================================
// apps/roca_batch_eval.cpp - Batch evaluation using saved model
// ============================================================================
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <iomanip>
#include <filesystem>

#include "../src/roca_idle_specialized.hpp"
#include "../src/model_io_complete.hpp"
#include "../src/io/binary_log.hpp"
#include "../src/data/window_maker.hpp"

using namespace roca;
using namespace roca_idle;

struct DatasetInfo {
    std::string filepath;
    std::string label;
    std::string expected_class;  // "normal" or "anomaly"
};

class BatchEvaluator {
private:
    IdleRoCAModel model;
    std::vector<DatasetInfo> datasets;
    std::map<std::string, std::vector<float>> results;
    
public:
    BatchEvaluator(const IdleRoCAConfig& config) : model(config) {}
    
    bool load_model(const std::string& model_path) {
        std::cout << "Loading model from: " << model_path << "\n";
        if (!model.load_model(model_path)) {
            std::cerr << "Failed to load model!\n";
            return false;
        }
        std::cout << "✅ Model loaded successfully\n\n";
        return true;
    }
    
    void add_dataset(const std::string& filepath, const std::string& label, 
                     const std::string& expected_class) {
        datasets.push_back({filepath, label, expected_class});
    }
    
    void evaluate_all() {
        std::cout << "╔════════════════════════════════════════╗\n";
        std::cout << "║       BATCH EVALUATION RESULTS        ║\n";
        std::cout << "╚════════════════════════════════════════╝\n\n";
        
        for (const auto& dataset : datasets) {
            evaluate_dataset(dataset);
        }
        
        print_summary();
    }
    
private:
    void evaluate_dataset(const DatasetInfo& info) {
        std::cout << "Evaluating: " << info.label << "\n";
        std::cout << "  File: " << info.filepath << "\n";
        std::cout << "  Expected: " << info.expected_class << "\n";
        
        // Load data
        std::ifstream file(info.filepath, std::ios::binary);
        if (!file) {
            std::cerr << "  ❌ Cannot open file\n\n";
            return;
        }
        
        BinaryFileHeader header;
        if (!read_header(file, header)) {
            std::cerr << "  ❌ Invalid header\n\n";
            return;
        }
        
        // Create windows
        WindowConfig window_cfg;
        window_cfg.T = model.get_config().T;
        window_cfg.stride = 5;
        window_cfg.D = header.feature_count;
        
        WindowMaker maker(window_cfg);
        std::vector<float> scores;
        
        AutoencoderFrame frame;
        while (read_frame(file, frame)) {
            std::vector<float> features(header.feature_count);
            for (size_t i = 0; i < header.feature_count; ++i) {
                features[i] = frame.features[i];
            }
            
            maker.push(features);
            
            if (maker.ready()) {
                auto window = maker.get_window();
                
                // Filter to valid features
                std::vector<float> filtered_window;
                const auto& valid_indices = model.get_valid_indices();
                
                for (size_t t = 0; t < window_cfg.T; ++t) {
                    for (size_t idx : valid_indices) {
                        filtered_window.push_back(window[t * header.feature_count + idx]);
                    }
                }
                
                float score = model.score_window(filtered_window);
                scores.push_back(score);
            }
        }
        
        if (scores.empty()) {
            std::cerr << "  ❌ No windows created\n\n";
            return;
        }
        
        // Analyze scores
        std::sort(scores.begin(), scores.end());
        float min_score = scores.front();
        float max_score = scores.back();
        float median = scores[scores.size() / 2];
        float mean = std::accumulate(scores.begin(), scores.end(), 0.0f) / scores.size();
        
        // Count anomalies
        int anomalies = 0;
        for (float score : scores) {
            if (score > model.anomaly_threshold) {
                anomalies++;
            }
        }
        
        float detection_rate = 100.0f * anomalies / scores.size();
        
        // Store results
        results[info.label] = scores;
        
        // Print results
        std::cout << "  Windows analyzed: " << scores.size() << "\n";
        std::cout << "  Score range: [" << std::fixed << std::setprecision(6) 
                  << min_score << ", " << max_score << "]\n";
        std::cout << "  Mean score: " << mean << "\n";
        std::cout << "  Median score: " << median << "\n";
        std::cout << "  Anomalies detected: " << anomalies << "/" << scores.size() 
                  << " (" << std::fixed << std::setprecision(1) << detection_rate << "%)\n";
        
        // Check if detection matches expectation
        bool is_correct = false;
        if (info.expected_class == "normal" && detection_rate < 20.0f) {
            std::cout << "  ✅ Correctly identified as NORMAL\n";
            is_correct = true;
        } else if (info.expected_class == "anomaly" && detection_rate > 60.0f) {
            std::cout << "  ✅ Correctly identified as ANOMALY\n";
            is_correct = true;
        } else {
            std::cout << "  ❌ Misclassified (expected " << info.expected_class << ")\n";
        }
        
        std::cout << "\n";
    }
    
    void print_summary() {
        std::cout << "╔════════════════════════════════════════╗\n";
        std::cout << "║            SUMMARY REPORT              ║\n";
        std::cout << "╚════════════════════════════════════════╝\n\n";
        
        std::cout << "Threshold: " << model.anomaly_threshold << "\n\n";
        
        int correct_normal = 0, total_normal = 0;
        int correct_anomaly = 0, total_anomaly = 0;
        
        for (const auto& dataset : datasets) {
            if (results.find(dataset.label) == results.end()) continue;
            
            const auto& scores = results[dataset.label];
            int anomalies = std::count_if(scores.begin(), scores.end(),
                [this](float s) { return s > model.anomaly_threshold; });
            float rate = 100.0f * anomalies / scores.size();
            
            if (dataset.expected_class == "normal") {
                total_normal++;
                if (rate < 20.0f) correct_normal++;
            } else {
                total_anomaly++;
                if (rate > 60.0f) correct_anomaly++;
            }
        }
        
        std::cout << "Normal datasets: " << correct_normal << "/" << total_normal 
                  << " correctly identified\n";
        std::cout << "Anomaly datasets: " << correct_anomaly << "/" << total_anomaly 
                  << " correctly identified\n";
        
        float accuracy = 100.0f * (correct_normal + correct_anomaly) / 
                        (total_normal + total_anomaly);
        std::cout << "Overall accuracy: " << std::fixed << std::setprecision(1) 
                  << accuracy << "%\n";
    }
};

int main(int argc, char** argv) {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║      RoCA Batch Evaluation Tool       ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    std::string model_file = "trained_model.roca";
    std::vector<DatasetInfo> datasets;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--model" && i + 1 < argc) {
            model_file = argv[++i];
        } else if (arg == "--add" && i + 3 < argc) {
            std::string filepath = argv[++i];
            std::string label = argv[++i];
            std::string expected = argv[++i];
            
            if (expected != "normal" && expected != "anomaly") {
                std::cerr << "Error: Expected class must be 'normal' or 'anomaly'\n";
                return 1;
            }
            
            datasets.push_back({filepath, label, expected});
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --model <file>   Model file to load (default: trained_model.roca)\n";
            std::cout << "  --add <file> <label> <expected>  Add dataset to evaluate\n";
            std::cout << "       <file>: Path to binary log\n";
            std::cout << "       <label>: Display name\n";
            std::cout << "       <expected>: 'normal' or 'anomaly'\n";
            std::cout << "\nExample:\n";
            std::cout << "  " << argv[0] << " --model my_model.roca \\\n";
            std::cout << "    --add idle_30min.bin \"30min idle\" normal \\\n";
            std::cout << "    --add walking_5min.bin \"Walking\" anomaly \\\n";
            std::cout << "    --add standing_2min.bin \"Standing up\" anomaly\n";
            return 0;
        }
    }
    
    if (datasets.empty()) {
        std::cerr << "Error: No datasets specified. Use --add to add datasets.\n";
        std::cerr << "Run with --help for usage information.\n";
        return 1;
    }
    
    // Initialize with dummy config (will be overwritten by load)
    IdleRoCAConfig config;
    BatchEvaluator evaluator(config);
    
    // Load model
    if (!evaluator.load_model(model_file)) {
        return 1;
    }
    
    // Add datasets
    for (const auto& dataset : datasets) {
        evaluator.add_dataset(dataset.filepath, dataset.label, dataset.expected_class);
    }
    
    // Run evaluation
    evaluator.evaluate_all();
    
    return 0;
}