// ============================================================================
// apps/coca_test.cpp - Test trained COCA model on CSV data
// ============================================================================
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cstring>  // For strcmp

#include "../src/coca_model.hpp"
#include "../src/io/csv_reader.hpp"
#include "../src/utils/model_io.hpp"

using namespace coca;

// Feature alignment modes
enum class AlignmentMode {
    ERROR,     // Hard fail on mismatch (default)
    TRUNCATE,  // Truncate to model's feature count
    PAD        // Pad with zeros to model's feature count
};

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

// Align features to match model expectations
std::vector<std::vector<float>> align_features(
    const std::vector<std::vector<float>>& windows,
    size_t csv_features,
    size_t model_features,
    AlignmentMode mode) {
    
    if (csv_features == model_features) {
        return windows;  // No alignment needed
    }
    
    std::vector<std::vector<float>> aligned_windows;
    size_t window_size = windows[0].size() / csv_features;
    
    for (const auto& window : windows) {
        std::vector<float> aligned_window;
        
        for (size_t t = 0; t < window_size; ++t) {
            size_t offset = t * csv_features;
            
            if (mode == AlignmentMode::TRUNCATE) {
                // Take first model_features from each timestep
                for (size_t f = 0; f < std::min(csv_features, model_features); ++f) {
                    aligned_window.push_back(window[offset + f]);
                }
                // Pad if necessary
                for (size_t f = csv_features; f < model_features; ++f) {
                    aligned_window.push_back(0.0f);
                }
            } else if (mode == AlignmentMode::PAD) {
                // Copy all available features
                for (size_t f = 0; f < std::min(csv_features, model_features); ++f) {
                    aligned_window.push_back(window[offset + f]);
                }
                // Pad remaining with zeros
                for (size_t f = csv_features; f < model_features; ++f) {
                    aligned_window.push_back(0.0f);
                }
            }
        }
        
        aligned_windows.push_back(aligned_window);
    }
    
    return aligned_windows;
}

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
    size_t window_size = 10;
    size_t window_stride = 5;
    bool skip_header = true;
    bool skip_timestamp = true;
    bool verbose = false;
    AlignmentMode alignment_mode = AlignmentMode::ERROR;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_file = argv[++i];
        } else if ((arg == "--test" || arg == "--csv") && i + 1 < argc) {
            test_file = argv[++i];
        } else if (arg == "--labels" && i + 1 < argc) {
            label_file = argv[++i];
        } else if (arg == "--window" && i + 1 < argc) {
            window_size = std::stoi(argv[++i]);
        } else if (arg == "--stride" && i + 1 < argc) {
            window_stride = std::stoi(argv[++i]);
        } else if (arg == "--align-features" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "truncate") {
                alignment_mode = AlignmentMode::TRUNCATE;
            } else if (mode == "pad") {
                alignment_mode = AlignmentMode::PAD;
            } else if (mode == "error") {
                alignment_mode = AlignmentMode::ERROR;
            } else {
                std::cerr << "Error: Invalid alignment mode: " << mode << "\n";
                std::cerr << "Valid modes: error, truncate, pad\n";
                return 1;
            }
        } else if (arg == "--no-header") {
            skip_header = false;
        } else if (arg == "--no-timestamp") {
            skip_timestamp = false;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " --csv <file> [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --csv <file>         Test CSV file (required)\n";
            std::cout << "  --model <file>       Model file (default: trained_model.coca)\n";
            std::cout << "  --labels <file>      True labels file (optional)\n";
            std::cout << "  --window <size>      Window size (default: 10)\n";
            std::cout << "  --stride <size>      Window stride (default: 5)\n";
            std::cout << "  --align-features <mode> Feature alignment mode:\n";
            std::cout << "                       error: hard fail on mismatch (default)\n";
            std::cout << "                       truncate: use first D features\n";
            std::cout << "                       pad: pad with zeros to D features\n";
            std::cout << "  --no-header          CSV has no header row\n";
            std::cout << "  --no-timestamp       Don't skip first column\n";
            std::cout << "  --verbose            Show detailed results\n";
            return 0;
        }
    }
    
    if (test_file.empty()) {
        std::cerr << "Error: Please specify test data with --csv or --test\n";
        return 1;
    }
    
    // Load model
    std::cout << "Loading model from: " << model_file << "\n";
    
    COCAConfig config;
    COCAModel model(config);
    
    if (!ModelIO::load_model(model, model_file)) {
        std::cerr << "Error: Cannot load model\n";
        return 1;
    }
    
    std::cout << "Model loaded:\n";
    std::cout << "  Window size: " << model.config.T << "\n";
    std::cout << "  Feature dim: " << model.config.D << "\n";
    std::cout << "  Threshold: " << model.anomaly_threshold << "\n";
    std::cout << "  Score mode: " << model.config.score_mix << "\n";
    std::cout << "  Threshold mode: " << model.config.threshold_mode << "\n\n";
    
    // Override window size from model if not specified
    if (window_size == 10 && model.config.T != 10) {
        window_size = model.config.T;
        std::cout << "Using model's window size: " << window_size << "\n";
    }
    
    // Load test data from CSV
    std::cout << "Loading test data from: " << test_file << "\n";
    std::cout << "  Skip header: " << (skip_header ? "yes" : "no") << "\n";
    std::cout << "  Skip timestamp: " << (skip_timestamp ? "yes" : "no") << "\n\n";
    
    CSVReader reader;
    if (!reader.load(test_file, skip_header, skip_timestamp, verbose)) {
        std::cerr << "Error: Failed to load test CSV file\n";
        return 1;
    }
    
    // Check feature count and handle mismatch
    size_t csv_features = reader.get_feature_count();
    size_t model_features = model.config.D;
    
    if (csv_features != model_features) {
        std::cout << "\nFeature count mismatch detected!\n";
        std::cout << "  CSV has: " << csv_features << " features\n";
        std::cout << "  Model expects: " << model_features << " features\n";
        
        if (alignment_mode == AlignmentMode::ERROR) {
            std::cerr << "\nError: Feature dimension mismatch!\n";
            std::cerr << "Use --align-features <truncate|pad> to handle the mismatch\n";
            return 1;
        } else {
            std::cout << "  Alignment mode: " 
                     << (alignment_mode == AlignmentMode::TRUNCATE ? "truncate" : "pad") << "\n";
        }
    }
    
    // Create test windows
    std::cout << "\nCreating test windows...\n";
    std::vector<std::vector<float>> test_windows = reader.get_windows(window_size, window_stride, verbose);
    
    if (test_windows.empty()) {
        std::cerr << "Error: No test windows created\n";
        return 1;
    }
    
    // Align features if needed
    if (csv_features != model_features && alignment_mode != AlignmentMode::ERROR) {
        std::cout << "Aligning features from " << csv_features << " to " << model_features << "...\n";
        test_windows = align_features(test_windows, csv_features, model_features, alignment_mode);
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
    
    // Save results to CSV
    std::ofstream results("test_results.csv");
    results << "window_index,score,detected_anomaly";
    if (!true_labels.empty()) {
        results << ",true_label,correct";
    }
    results << "\n";
    
    for (size_t i = 0; i < all_scores.size(); ++i) {
        results << i << "," << all_scores[i] << "," << (all_scores[i] > model.anomaly_threshold ? 1 : 0);
        if (i < true_labels.size()) {
            bool is_correct = (all_scores[i] > model.anomaly_threshold) == true_labels[i];
            results << "," << true_labels[i] << "," << is_correct;
        }
        results << "\n";
    }
    results.close();
    
    std::cout << "\nResults saved to: test_results.csv\n";
    
    return 0;
}
