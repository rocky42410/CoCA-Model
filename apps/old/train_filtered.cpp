#include <iostream>
#include <vector>
#include <chrono>
#include "../src/roca_one_class.hpp"
#include "../src/feature_filter.hpp"
#include "../src/io/binary_log.hpp"

using namespace roca;

int main(int argc, char** argv) {
    std::string data_file = "";
    std::string output_model = "filtered_model.roca";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_model = argv[++i];
        }
    }
    
    if (data_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " --data <file.bin> [--output <model.roca>]\n";
        return 1;
    }
    
    std::cout << "\n╔══════════════════════════════════════════════════╗\n";
    std::cout << "║   Filtered RoCA Training (NaN-Aware)              ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n\n";
    
    // 1. Load binary data
    BinaryLogReader reader;
    if (!reader.open(data_file)) {
        return 1;
    }
    
    // 2. Analyze NaN patterns
    reader.analyze_nan_statistics();
    
    // 3. Read all windows (raw with NaN)
    size_t window_size = 10;
    size_t stride = 5;
    auto raw_windows = reader.read_all_windows(window_size, stride);
    reader.close();
    
    if (raw_windows.empty()) {
        std::cerr << "No windows created!\n";
        return 1;
    }
    
    // 4. Filter features
    FeatureFilter filter;
    filter.fit(raw_windows, window_size);
    
    if (filter.get_output_dim() == 0) {
        std::cerr << "No valid features found!\n";
        return 1;
    }
    
    // 5. Create filtered windows
    std::vector<std::vector<float>> filtered_windows;
    for (const auto& raw : raw_windows) {
        filtered_windows.push_back(filter.filter_window(raw, window_size));
    }
    
    std::cout << "\nFiltered data:\n";
    std::cout << "  Windows: " << filtered_windows.size() << "\n";
    std::cout << "  Features per timestep: " << filter.get_output_dim() << "\n";
    std::cout << "  Total dimension: " << window_size << " × " << filter.get_output_dim() 
              << " = " << (window_size * filter.get_output_dim()) << "\n";
    
    // 6. Configure model with ACTUAL dimensions
    OneClassConfig config;
    config.T = window_size;
    config.D = filter.get_output_dim();  // Use actual valid features, not 256!
    config.C = 32;
    config.K = 16;
    config.epochs = 100;
    config.batch_size = 32;
    config.lr = 1e-3;
    
    // 7. Train model
    OneClassRoCA model(config);
    train_one_class_model(model, filtered_windows, config);
    
    // 8. Verify on training data
    std::cout << "\n=== Verification ===\n";
    float total_score = 0;
    int high_scores = 0;
    
    for (size_t i = 0; i < std::min(size_t(100), filtered_windows.size()); ++i) {
        float score = model.anomaly_score(filtered_windows[i]);
        total_score += score;
        if (score > model.anomaly_threshold) {
            high_scores++;
        }
    }
    
    float avg_score = total_score / std::min(size_t(100), filtered_windows.size());
    std::cout << "Average score on training data: " << avg_score << "\n";
    std::cout << "Should be < 0.01 for good reconstruction\n";
    
    if (avg_score > 0.05) {
        std::cout << "\n⚠️  Warning: High reconstruction error on training data!\n";
        std::cout << "This suggests the model didn't learn properly.\n";
    } else {
        std::cout << "\n✅ Good reconstruction quality!\n";
    }
    
    // 9. Save model with filter info
    model.save_model(output_model);
    filter.save_indices("model_features.txt");
    
    std::cout << "\nModel saved to: " << output_model << "\n";
    std::cout << "Feature indices saved to: model_features.txt\n";
    
    return 0;
}