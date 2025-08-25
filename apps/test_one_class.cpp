// ============================================================================
// test_one_class.cpp - Test one-class model on unseen data
// ============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

#include "../src/roca_one_class.hpp"
#include "../src/model_io_complete.hpp"
#include "../src/io/binary_log.hpp"
#include "../src/data/window_maker.hpp"

using namespace roca;

int main(int argc, char** argv) {
    std::cout << "\n╔══════════════════════════════════════════════════╗\n";
    std::cout << "║      One-Class Anomaly Detection Testing         ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n\n";
    
    std::string model_file = "";
    std::string test_file = "";
    std::string label = "unknown";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_file = argv[++i];
        } else if (arg == "--test" && i + 1 < argc) {
            test_file = argv[++i];
        } else if (arg == "--label" && i + 1 < argc) {
            label = argv[++i];
        }
    }
    
    if (model_file.empty() || test_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " --model <model.roca> --test <data.bin> [--label <name>]\n";
        return 1;
    }
    
    // Load one-class model
    OneClassConfig config;
    OneClassRoCA model(config);
    
    std::cout << "Loading one-class model: " << model_file << "\n";
    if (!model.load_model(model_file)) {
        std::cerr << "Failed to load model!\n";
        return 1;
    }
    
    std::cout << "✅ Model loaded (trained on NORMAL data only)\n";
    std::cout << "Anomaly threshold: " << model.anomaly_threshold << "\n\n";
    
    // Load test data
    std::cout << "Testing on: " << test_file << " (" << label << ")\n";
    
    std::ifstream file(test_file, std::ios::binary);
    BinaryFileHeader header;
    read_header(file, header);
    
    // Process test data
    WindowConfig window_cfg;
    window_cfg.T = model.get_config().T;
    window_cfg.stride = 5;
    window_cfg.D = header.feature_count;
    
    WindowMaker maker(window_cfg);
    std::vector<float> scores;
    int anomalies = 0;
    
    AutoencoderFrame frame;
    while (read_frame(file, frame)) {
        std::vector<float> features(header.feature_count);
        for (size_t i = 0; i < header.feature_count; ++i) {
            features[i] = frame.features[i];
        }
        
        maker.push(features);
        if (maker.ready()) {
            auto window = maker.get_window();
            
            // Filter features if needed
            // ... (feature filtering code) ...
            
            float score = model.anomaly_score(window);
            scores.push_back(score);
            
            if (score > model.anomaly_threshold) {
                anomalies++;
            }
        }
    }
    
    // Report results
    float detection_rate = 100.0f * anomalies / scores.size();
    
    std::cout << "\n═══════════════════════════════════════════\n";
    std::cout << "           DETECTION RESULTS\n";
    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "Windows analyzed: " << scores.size() << "\n";
    std::cout << "Anomalies detected: " << anomalies << " (" 
              << std::fixed << std::setprecision(1) << detection_rate << "%)\n";
    
    // Interpret results
    std::cout << "\nInterpretation:\n";
    if (detection_rate < 20.0f) {
        std::cout << "✅ NORMAL - Behavior matches training distribution\n";
    } else if (detection_rate > 60.0f) {
        std::cout << "🚨 ANOMALY - Significant deviation from normal\n";
    } else {
        std::cout << "⚠️  UNCERTAIN - Some deviations detected\n";
    }
    
    std::cout << "\nRemember: This model was trained on NORMAL data only.\n";
    std::cout << "It detects ANY deviation from that learned normal behavior.\n";
    
    return 0;
}