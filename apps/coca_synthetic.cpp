// ============================================================================
// apps/coca_synthetic.cpp - Generate synthetic data for COCA testing
// ============================================================================
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include "../src/coca_model.hpp"
#include "../src/io/binary_log.hpp"

using namespace coca;
using namespace roca;

// ============================================================================
// Synthetic data patterns
// ============================================================================
class SyntheticGenerator {
private:
    std::mt19937 rng;
    size_t num_features;
    
public:
    SyntheticGenerator(size_t features = 256, unsigned seed = 42)
        : rng(seed), num_features(features) {}
    
    // Generate normal (idle) data
    std::vector<float> generate_normal_frame(float time) {
        std::vector<float> frame(num_features);
        std::normal_distribution<float> noise(0.0f, 0.01f);
        
        for (size_t i = 0; i < num_features; ++i) {
            if (i < 50) {
                // Stable features with small noise
                frame[i] = 0.5f + noise(rng);
            } else if (i < 100) {
                // Slow sinusoidal variation
                frame[i] = 0.5f + 0.1f * std::sin(time * 0.1f + i) + noise(rng);
            } else if (i < 150) {
                // Constants
                frame[i] = 1.0f;
            } else {
                // Random walk
                static std::vector<float> walk_state(num_features, 0.5f);
                walk_state[i] += noise(rng) * 0.1f;
                walk_state[i] = std::max(0.0f, std::min(1.0f, walk_state[i]));
                frame[i] = walk_state[i];
            }
        }
        
        return frame;
    }
    
    // Generate anomalous data (different patterns)
    std::vector<float> generate_anomaly_frame(float time, const std::string& anomaly_type) {
        std::vector<float> frame(num_features);
        std::normal_distribution<float> noise(0.0f, 0.01f);
        
        if (anomaly_type == "spike") {
            // Sudden spikes in normally stable features
            frame = generate_normal_frame(time);
            for (size_t i = 0; i < 20; ++i) {
                size_t idx = rng() % num_features;
                frame[idx] += (rng() % 2 ? 1.0f : -1.0f) * 2.0f;
            }
            
        } else if (anomaly_type == "drift") {
            // Gradual drift from normal
            frame = generate_normal_frame(time);
            float drift = time * 0.01f;
            for (size_t i = 0; i < num_features; ++i) {
                frame[i] += drift;
            }
            
        } else if (anomaly_type == "oscillation") {
            // High-frequency oscillation
            for (size_t i = 0; i < num_features; ++i) {
                frame[i] = 0.5f + 0.5f * std::sin(time * 10.0f + i) + noise(rng);
            }
            
        } else if (anomaly_type == "correlation_break") {
            // Break normal correlations between features
            frame = generate_normal_frame(time);
            // Shuffle first 50 features
            std::shuffle(frame.begin(), frame.begin() + 50, rng);
            
        } else {
            // Random noise
            std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
            for (size_t i = 0; i < num_features; ++i) {
                frame[i] = uniform(rng);
            }
        }
        
        return frame;
    }
    
    // Create a binary log file
    void create_log(const std::string& filename,
                   const std::vector<std::vector<float>>& frames,
                   float rate_hz = 50.0f) {
        std::ofstream file(filename, std::ios::binary);
        
        // Write header
        BinaryFileHeader header;
        header.feature_count = num_features;
        header.fusion_rate_hz = static_cast<uint32_t>(rate_hz);
        write_header(file, header);
        
        // Write frames
        uint64_t timestamp_ns = 0;
        uint64_t dt_ns = static_cast<uint64_t>(1e9 / rate_hz);
        
        for (const auto& frame_data : frames) {
            AutoencoderFrame frame;
            frame.timestamp_ns = timestamp_ns;
            frame.modality_mask = 0x1F;
            
            for (int i = 0; i < 5; ++i) {
                frame.sample_counts[i] = 1;
                frame.staleness_ms[i] = 0.0f;
            }
            
            std::copy(frame_data.begin(), frame_data.end(), frame.features);
            
            write_frame(file, frame);
            timestamp_ns += dt_ns;
        }
        
        file.close();
    }
};

// ============================================================================
// Main application
// ============================================================================
int main(int argc, char** argv) {
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║    COCA Synthetic Data Generator       ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    // Parse arguments
    size_t num_features = 256;
    size_t normal_frames = 10000;
    size_t anomaly_frames = 1000;
    float sample_rate = 50.0f;
    std::string output_prefix = "synthetic";
    bool run_test = false;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--features" && i + 1 < argc) {
            num_features = std::stoi(argv[++i]);
        } else if (arg == "--normal" && i + 1 < argc) {
            normal_frames = std::stoi(argv[++i]);
        } else if (arg == "--anomaly" && i + 1 < argc) {
            anomaly_frames = std::stoi(argv[++i]);
        } else if (arg == "--rate" && i + 1 < argc) {
            sample_rate = std::stof(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_prefix = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        } else if (arg == "--test") {
            run_test = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --features N   Number of features (default: 256)\n";
            std::cout << "  --normal N     Normal frames to generate (default: 10000)\n";
            std::cout << "  --anomaly N    Anomaly frames to generate (default: 1000)\n";
            std::cout << "  --rate Hz      Sample rate (default: 50)\n";
            std::cout << "  --output name  Output file prefix (default: synthetic)\n";
            std::cout << "  --seed N       Random seed\n";
            std::cout << "  --test         Run quick test\n";
            return 0;
        }
    }
    
    SyntheticGenerator generator(num_features, seed);
    
    if (run_test) {
        // Quick test mode
        std::cout << "Running quick test...\n";
        
        // Generate small datasets
        std::vector<std::vector<float>> train_data;
        std::vector<std::vector<float>> test_normal;
        std::vector<std::vector<float>> test_anomaly;
        
        for (size_t i = 0; i < 500; ++i) {
            train_data.push_back(generator.generate_normal_frame(i * 0.02f));
            test_normal.push_back(generator.generate_normal_frame((500 + i) * 0.02f));
        }
        
        for (size_t i = 0; i < 100; ++i) {
            test_anomaly.push_back(generator.generate_anomaly_frame(i * 0.02f, "spike"));
        }
        
        // Quick train
        COCAConfig config;
        config.D = num_features;
        config.epochs = 10;
        config.batch_size = 16;
        
        COCAModel model(config);
        
        // Create windows
        std::vector<std::vector<float>> windows;
        for (size_t i = 0; i + config.T < train_data.size(); i += 5) {
            std::vector<float> window;
            for (size_t t = 0; t < config.T; ++t) {
                window.insert(window.end(), train_data[i+t].begin(), train_data[i+t].end());
            }
            windows.push_back(window);
        }
        
        std::cout << "Training on " << windows.size() << " windows...\n";
        train_coca_model(model, windows, config);
        
        // Test
        int normal_detected = 0, anomaly_detected = 0;
        
        for (size_t i = 0; i + config.T < test_normal.size(); i += config.T) {
            std::vector<float> window;
            for (size_t t = 0; t < config.T; ++t) {
                window.insert(window.end(), test_normal[i+t].begin(), test_normal[i+t].end());
            }
            float score = model.score_window(window);
            if (score > model.anomaly_threshold) normal_detected++;
        }
        
        for (size_t i = 0; i + config.T < test_anomaly.size(); i += config.T) {
            std::vector<float> window;
            for (size_t t = 0; t < config.T; ++t) {
                window.insert(window.end(), test_anomaly[i+t].begin(), test_anomaly[i+t].end());
            }
            float score = model.score_window(window);
            if (score > model.anomaly_threshold) anomaly_detected++;
        }
        
        size_t n_normal = test_normal.size() / config.T;
        size_t n_anomaly = test_anomaly.size() / config.T;
        
        std::cout << "\nTest Results:\n";
        std::cout << "  False positives: " << normal_detected << "/" << n_normal 
                  << " (" << (100.0f * normal_detected / n_normal) << "%)\n";
        std::cout << "  True positives: " << anomaly_detected << "/" << n_anomaly
                  << " (" << (100.0f * anomaly_detected / n_anomaly) << "%)\n";
        
        return 0;
    }
    
    // Normal mode - generate datasets
    std::cout << "Configuration:\n";
    std::cout << "  Features: " << num_features << "\n";
    std::cout << "  Normal frames: " << normal_frames << "\n";
    std::cout << "  Anomaly frames: " << anomaly_frames << "\n";
    std::cout << "  Sample rate: " << sample_rate << " Hz\n";
    std::cout << "  Seed: " << seed << "\n\n";
    
    // Generate training data (normal only)
    std::cout << "Generating training data...\n";
    std::vector<std::vector<float>> train_data;
    
    float dt = 1.0f / sample_rate;
    for (size_t i = 0; i < normal_frames; ++i) {
        train_data.push_back(generator.generate_normal_frame(i * dt));
        
        if ((i + 1) % 1000 == 0) {
            std::cout << "  Generated " << (i + 1) << "/" << normal_frames << " frames\r" << std::flush;
        }
    }
    std::cout << "\n";
    
    std::string train_file = output_prefix + "_train.bin";
    generator.create_log(train_file, train_data, sample_rate);
    std::cout << "Saved training data to: " << train_file << "\n";
    
    // Generate test data (mixed)
    std::cout << "\nGenerating test data...\n";
    std::vector<std::vector<float>> test_data;
    std::vector<int> test_labels;
    
    // Add normal frames
    for (size_t i = 0; i < normal_frames / 2; ++i) {
        test_data.push_back(generator.generate_normal_frame((normal_frames + i) * dt));
        test_labels.push_back(0);
    }
    
    // Add various anomaly types
    std::vector<std::string> anomaly_types = {"spike", "drift", "oscillation", "correlation_break", "random"};
    size_t frames_per_type = anomaly_frames / anomaly_types.size();
    
    for (const auto& anomaly_type : anomaly_types) {
        std::cout << "  Generating " << anomaly_type << " anomalies...\n";
        for (size_t i = 0; i < frames_per_type; ++i) {
            test_data.push_back(generator.generate_anomaly_frame(i * dt, anomaly_type));
            test_labels.push_back(1);
        }
    }
    
    // Shuffle test data
    std::mt19937 shuffle_rng(seed + 1);  // Different seed for shuffling
    std::vector<size_t> indices(test_data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), shuffle_rng);
    
    std::vector<std::vector<float>> shuffled_data;
    std::vector<int> shuffled_labels;
    
    for (size_t idx : indices) {
        shuffled_data.push_back(test_data[idx]);
        shuffled_labels.push_back(test_labels[idx]);
    }
    
    std::string test_file = output_prefix + "_test.bin";
    generator.create_log(test_file, shuffled_data, sample_rate);
    std::cout << "Saved test data to: " << test_file << "\n";
    
    // Save labels
    std::string label_file = output_prefix + "_labels.txt";
    std::ofstream labels(label_file);
    for (int label : shuffled_labels) {
        labels << label << "\n";
    }
    labels.close();
    std::cout << "Saved labels to: " << label_file << "\n";
    
    // Summary
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║         Generation Complete            ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    std::cout << "Generated files:\n";
    std::cout << "  Training: " << train_file << " (" << train_data.size() << " frames)\n";
    std::cout << "  Testing:  " << test_file << " (" << shuffled_data.size() << " frames)\n";
    std::cout << "  Labels:   " << label_file << "\n\n";
    
    std::cout << "Next steps:\n";
    std::cout << "1. Train model: ./coca_train --data " << train_file << "\n";
    std::cout << "2. Test model:  ./coca_test --test " << test_file << " --labels " << label_file << "\n";
    
    return 0;
}
