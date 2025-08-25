// ============================================================================
// apps/roca_live_filtered.cpp - Live inference using saved model
// ============================================================================
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>
#include <deque>

#include "../src/roca_idle_specialized.hpp"
#include "../src/model_io_complete.hpp"
#include "../src/io/binary_log.hpp"
#include "../src/data/window_maker.hpp"

using namespace roca;
using namespace roca_idle;

class LiveAnalyzer {
private:
    IdleRoCAModel model;
    WindowMaker window_maker;
    std::deque<float> score_history;
    size_t history_size = 50;
    size_t frame_count = 0;
    size_t anomaly_count = 0;
    
public:
    LiveAnalyzer(const IdleRoCAConfig& config) 
        : model(config), 
          window_maker({config.T, 5, 256}) {}  // Will be updated after model load
    
    bool load_model(const std::string& model_path) {
        if (!model.load_model(model_path)) {
            return false;
        }
        
        // Update window maker with correct dimensions
        WindowConfig cfg;
        cfg.T = model.get_config().T;
        cfg.stride = 5;
        cfg.D = 256;  // Original feature dimension
        window_maker = WindowMaker(cfg);
        
        return true;
    }
    
    void process_frame(const std::vector<float>& frame) {
        frame_count++;
        
        // Add frame to window maker
        window_maker.push(frame);
        
        // Check if we have a complete window
        if (!window_maker.ready()) {
            return;
        }
        
        // Get window and filter features
        auto full_window = window_maker.get_window();
        std::vector<float> filtered_window;
        const auto& valid_indices = model.get_valid_indices();
        
        for (size_t t = 0; t < model.get_config().T; ++t) {
            for (size_t idx : valid_indices) {
                filtered_window.push_back(full_window[t * 256 + idx]);
            }
        }
        
        // Score the window
        float score = model.score_window(filtered_window);
        
        // Update history
        score_history.push_back(score);
        if (score_history.size() > history_size) {
            score_history.pop_front();
        }
        
        // Check for anomaly
        bool is_anomaly = score > model.anomaly_threshold;
        if (is_anomaly) {
            anomaly_count++;
        }
        
        // Print status
        print_status(score, is_anomaly);
    }
    
    void process_stream(std::istream& stream) {
        BinaryFileHeader header;
        if (!read_header(stream, header)) {
            std::cerr << "Error: Invalid stream header\n";
            return;
        }
        
        std::cout << "\n╔════════════════════════════════════════╗\n";
        std::cout << "║         LIVE ANALYSIS STARTED          ║\n";
        std::cout << "╚════════════════════════════════════════╝\n\n";
        std::cout << "Threshold: " << model.anomaly_threshold << "\n";
        std::cout << "Processing frames...\n\n";
        
        AutoencoderFrame frame;
        auto start_time = std::chrono::steady_clock::now();
        
        while (read_frame(stream, frame)) {
            // Convert to vector
            std::vector<float> features(header.feature_count);
            for (size_t i = 0; i < header.feature_count; ++i) {
                features[i] = frame.features[i];
            }
            
            // Process
            process_frame(features);
            
            // Simulate real-time processing
            std::this_thread::sleep_for(std::chrono::milliseconds(20));  // 50Hz
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
            end_time - start_time).count();
        
        print_summary(duration);
    }
    
private:
    void print_status(float score, bool is_anomaly) {
        // Calculate moving average
        float avg_score = 0.0f;
        if (!score_history.empty()) {
            avg_score = std::accumulate(score_history.begin(), score_history.end(), 0.0f) 
                       / score_history.size();
        }
        
        // Create visual bar
        int bar_length = 40;
        float normalized_score = std::min(1.0f, score / (model.anomaly_threshold * 2));
        int filled = static_cast<int>(normalized_score * bar_length);
        
        std::cout << "\r[";
        for (int i = 0; i < bar_length; ++i) {
            if (i < filled) {
                std::cout << (is_anomaly ? "█" : "▓");
            } else {
                std::cout << "░";
            }
        }
        std::cout << "] ";
        
        // Print score and status
        std::cout << "Score: " << std::fixed << std::setprecision(4) << score;
        std::cout << " | Avg: " << avg_score;
        std::cout << " | Status: ";
        
        if (is_anomaly) {
            std::cout << "\033[31mANOMALY\033[0m";  // Red
        } else {
            std::cout << "\033[32mNORMAL\033[0m";   // Green
        }
        
        std::cout << " | Frames: " << frame_count;
        std::cout << " | Anomalies: " << anomaly_count;
        std::cout << "    " << std::flush;
    }
    
    void print_summary(int duration_sec) {
        std::cout << "\n\n╔════════════════════════════════════════╗\n";
        std::cout << "║           ANALYSIS COMPLETE            ║\n";
        std::cout << "╚════════════════════════════════════════╝\n\n";
        
        std::cout << "Duration: " << duration_sec << " seconds\n";
        std::cout << "Total frames: " << frame_count << "\n";
        std::cout << "Windows analyzed: " << score_history.size() << "\n";
        std::cout << "Anomalies detected: " << anomaly_count << "\n";
        
        if (!score_history.empty()) {
            auto [min_it, max_it] = std::minmax_element(
                score_history.begin(), score_history.end());
            float avg = std::accumulate(score_history.begin(), score_history.end(), 0.0f) 
                       / score_history.size();
            
            std::cout << "\nScore statistics:\n";
            std::cout << "  Min: " << *min_it << "\n";
            std::cout << "  Max: " << *max_it << "\n";
            std::cout << "  Avg: " << avg << "\n";
            std::cout << "  Threshold: " << model.anomaly_threshold << "\n";
        }
    }
};

int main(int argc, char** argv) {
    std::string model_file = "trained_model.roca";
    std::string data_file = "";
    bool real_time = false;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--model" && i + 1 < argc) {
            model_file = argv[++i];
        } else if (arg == "--file" && i + 1 < argc) {
            data_file = argv[++i];
        } else if (arg == "--realtime") {
            real_time = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --model <file>  Model file to load (default: trained_model.roca)\n";
            std::cout << "  --file <file>   Binary log to analyze\n";
            std::cout << "  --realtime      Simulate real-time processing\n";
            return 0;
        }
    }
    
    if (data_file.empty()) {
        std::cerr << "Error: Please specify a data file with --file\n";
        return 1;
    }
    
    // Initialize analyzer
    IdleRoCAConfig config;  // Dummy config, will be overwritten
    LiveAnalyzer analyzer(config);
    
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║        RoCA Live Analysis Tool         ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    // Load model
    std::cout << "Loading model: " << model_file << "\n";
    if (!analyzer.load_model(model_file)) {
        std::cerr << "Failed to load model!\n";
        return 1;
    }
    std::cout << "✅ Model loaded successfully\n";
    
    // Open data file
    std::ifstream file(data_file, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open " << data_file << "\n";
        return 1;
    }
    
    // Process stream
    analyzer.process_stream(file);
    
    return 0;
}