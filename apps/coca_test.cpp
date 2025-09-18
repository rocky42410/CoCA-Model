// ============================================================================
// apps/coca_test.cpp - Test trained COCA model on new data
// ============================================================================
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include "../src/coca_model.hpp"
#include "../src/io/binary_log.hpp"
#include "../src/data/window_maker.hpp"
#include "../src/utils/model_io.hpp"

using namespace coca;
using namespace roca;

// ============================================================================
// Test metrics computation
// ============================================================================
struct TestMetrics {
    size_t true_positives = 0;
    size_t false_positives = 0;
    size_t true_negatives = 0;
    size_t false_negatives = 0;
    
    std::vector<float> scores;
    std::vector<bool> labels;  // true = anomaly
    
    float get_accuracy() const {
        size_t total = true_positives + false_positives + true_negatives + false_negatives;
        if (total == 0) return 0.0f;
        return static_cast<float>(true_positives + true_negatives) / total;
    }
    
    float get_precision() const {
        if (true_positives + false_positives == 0) return 0.0f;
        return static_cast<float>(true_positives) / (true_positives + false_positives);
    }
    
    float get_recall() const {
        if (true_positives + false_negatives == 0) return 0.0f;
        return static_cast<float>(true_positives) / (true_positives + false_negatives);
    }
    
    float get_f1_score() const {
        float precision = get_precision();
        float recall = get_recall();
        if (precision + recall == 0) return 0.0f;
        return 2 * precision * recall / (precision + recall);
    }
    
    float get_tpr_at_fpr(float target_fpr) const {
        if (scores.empty()) return 0.0f;
        
        // Sort scores with labels
        std::vector<std::pair<float, bool>> scored_labels;
        for (size_t i = 0; i < scores.size(); ++i) {
            scored_labels.push_back({scores[i], labels[i]});
        }
        std::sort(scored_labels.begin(), scored_labels.end());
        
        size_t n_negatives = std::count(labels.begin(), labels.end(), false);
        size_t n_positives = labels.size() - n_negatives;
        
        if (n_negatives == 0 || n_positives == 0) return 0.0f;
        
        size_t max_fp = static_cast<size_t>(target_fpr * n_negatives);
        
        // Find threshold that gives us this FPR
        size_t fp = 0;
        size_t tp = 0;
        
        for (auto it = scored_labels.rbegin(); it != scored_labels.rend(); ++it) {
            if (it->second) {  // True anomaly
                tp++;
            } else {  // Normal
                fp++;
                if (fp > max_fp) break;
            }
        }
        
        return static_cast<float>(tp) / n_positives;
    }
    
    void print_summary() const {
        std::cout << "\nTest Metrics:\n";
        std::cout << "─────────────────────────────\n";
        std::cout << "True Positives:  " << true_positives << "\n";
        std::cout << "False Positives: " << false_positives << "\n";
        std::cout << "True Negatives:  " << true_negatives << "\n";
        std::cout << "False Negatives: " << false_negatives << "\n";
        std::cout << "\n";
        std::cout << "Accuracy:  " << std::fixed << std::setprecision(3) << get_accuracy() << "\n";
        std::cout << "Precision: " << get_precision() << "\n";
        std::cout << "Recall:    " << get_recall() << "\n";
        std::cout << "F1 Score:  " << get_f1_score() << "\n";
        std::cout << "\n";
        std::cout << "TPR @ 1% FPR:  " << get_tpr_at_fpr(0.01f) << "\n";
        std::cout << "TPR @ 5% FPR:  " << get_tpr_at_fpr(0.05f) << "\n";
        std::cout << "TPR @ 10% FPR: " << get_tpr_at_fpr(0.10f) << "\n";
    }
};

// ============================================================================
// Main test application
// ============================================================================
int main(int argc, char** argv) {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║         COCA Test Application          ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    // Parse arguments
    std::string model_file = "trained_model.coca";
    std::string test_file = "";
    std::string label_file = "";  // Optional: file with true labels
    bool verbose = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_file = argv[++i];
        } else if (arg == "--test" && i + 1 < argc) {
            test_file = argv[++i];
        } else if (arg == "--labels" && i + 1 < argc) {
            label_file = argv[++i];
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " --test <file> [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --test <file>    Test data file (required)\n";
            std::cout << "  --model <file>   Model file (default: trained_model.coca)\n";
            std::cout << "  --labels <file>  True labels file (optional)\n";
            std::cout << "  --verbose        Show detailed results\n";
            return 0;
        }
    }
    
    if (test_file.empty()) {
        std::cerr << "Error: Please specify test data with --test\n";
        return 1;
    }
    
    // Load model
    std::cout << "Loading model from: " << model_file << "\n";
    
    COCAConfig config;
    config.D = 256;  // Will be updated from model
    COCAModel model(config);
    
    if (!ModelIO::load_model(model, model_file)) {
        std::cerr << "Error: Cannot load model\n";
        return 1;
    }
    
    std::cout << "Model loaded:\n";
    std::cout << "  Window size: " << model.config.T << "\n";
    std::cout << "  Feature dim: " << model.config.D << "\n";
    std::cout << "  Threshold: " << model.anomaly_threshold << "\n\n";
    
    // Load test data
    std::cout << "Loading test data from: " << test_file << "\n";
    
    std::ifstream file(test_file, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open test file\n";
        return 1;
    }
    
    BinaryFileHeader header;
    if (!read_header(file, header)) {
        std::cerr << "Error: Invalid header\n";
        return 1;
    }
    
    // Load frames
    std::vector<std::vector<float>> test_frames;
    AutoencoderFrame frame;
    
    while (read_frame(file, frame)) {
        std::vector<float> features(header.feature_count);
        for (size_t i = 0; i < header.feature_count; ++i) {
            features[i] = frame.features[i];
        }
        test_frames.push_back(features);
    }
    
    std::cout << "Loaded " << test_frames.size() << " test frames\n";
    
    // Create windows
    WindowConfig window_cfg;
    window_cfg.T = model.config.T;
    window_cfg.stride = 5;
    window_cfg.D = header.feature_count;
    
    WindowMaker maker(window_cfg);
    std::vector<std::vector<float>> test_windows;
    
    for (const auto& frame_data : test_frames) {
        maker.push(frame_data);
        if (maker.ready()) {
            test_windows.push_back(maker.get_window());
        }
    }
    
    std::cout << "Created " << test_windows.size() << " test windows\n\n";
    
    // Load labels if provided
    std::vector<bool> true_labels;
    if (!label_file.empty()) {
        std::ifstream labels(label_file);
        if (labels) {
            int label;
            while (labels >> label) {
                true_labels.push_back(label != 0);
            }
            std::cout << "Loaded " << true_labels.size() << " labels\n";
        }
    }
    
    // Test model
    std::cout << "--- Testing Model ---\n";
    
    TestMetrics metrics;
    std::vector<float> all_scores;
    int anomaly_count = 0;
    
    for (size_t i = 0; i < test_windows.size(); ++i) {
        float score = model.score_window(test_windows[i]);
        all_scores.push_back(score);
        
        bool is_anomaly = score > model.anomaly_threshold;
        if (is_anomaly) {
            anomaly_count++;
        }
        
        // If we have true labels, compute metrics
        if (i < true_labels.size()) {
            metrics.scores.push_back(score);
            metrics.labels.push_back(true_labels[i]);
            
            if (is_anomaly && true_labels[i]) {
                metrics.true_positives++;
            } else if (is_anomaly && !true_labels[i]) {
                metrics.false_positives++;
            } else if (!is_anomaly && true_labels[i]) {
                metrics.false_negatives++;
            } else {
                metrics.true_negatives++;
            }
        }
        
        if (verbose && is_anomaly) {
            std::cout << "Window " << i << ": ANOMALY (score=" 
                     << std::fixed << std::setprecision(6) << score << ")\n";
        }
    }
    
    // Compute statistics
    std::sort(all_scores.begin(), all_scores.end());
    float min_score = all_scores.front();
    float max_score = all_scores.back();
    float median_score = all_scores[all_scores.size() / 2];
    float mean_score = std::accumulate(all_scores.begin(), all_scores.end(), 0.0f) / all_scores.size();
    
    // Results
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║            TEST RESULTS                ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    std::cout << "Anomalies detected: " << anomaly_count << "/" << test_windows.size();
    std::cout << " (" << std::fixed << std::setprecision(1) 
              << (100.0f * anomaly_count / test_windows.size()) << "%)\n\n";
    
    std::cout << "Score Statistics:\n";
    std::cout << "─────────────────────────────\n";
    std::cout << "Min:       " << std::fixed << std::setprecision(6) << min_score << "\n";
    std::cout << "Median:    " << median_score << "\n";
    std::cout << "Mean:      " << mean_score << "\n";
    std::cout << "Max:       " << max_score << "\n";
    std::cout << "Threshold: " << model.anomaly_threshold << "\n";
    
    // If we have true labels, show metrics
    if (!true_labels.empty()) {
        metrics.print_summary();
    }
    
    // Show score distribution
    if (verbose) {
        std::cout << "\nScore Distribution:\n";
        std::cout << "─────────────────────────────\n";
        
        int n_bins = 10;
        float bin_width = (max_score - min_score) / n_bins;
        
        for (int i = 0; i < n_bins; ++i) {
            float bin_start = min_score + i * bin_width;
            float bin_end = bin_start + bin_width;
            
            int count = 0;
            for (float score : all_scores) {
                if (score >= bin_start && score < bin_end) {
                    count++;
                }
            }
            
            std::cout << "[" << std::fixed << std::setprecision(4) << bin_start 
                     << "-" << bin_end << "]: ";
            
            int bar_length = (50 * count) / test_windows.size();
            for (int j = 0; j < bar_length; ++j) std::cout << "█";
            std::cout << " " << count << "\n";
        }
    }
    
    return 0;
}
