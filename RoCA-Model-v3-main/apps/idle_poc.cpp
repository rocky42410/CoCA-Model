// ============================================================================
// apps/idle_poc.cpp - Proof-of-concept for idle baseline anomaly detection
// ============================================================================
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>

// Include the specialized idle implementation
#include "../src/roca_idle_specialized.hpp"
#include "../src/io/binary_log.hpp"
#include "../src/data/window_maker.hpp"

using namespace roca;
using namespace roca_idle;

// ============================================================================
// Synthetic data generator for testing
// ============================================================================
class SyntheticDataGenerator {
private:
    std::mt19937 rng;
    size_t num_features;
    
public:
    SyntheticDataGenerator(size_t features = 256, unsigned seed = 42) 
        : rng(seed), num_features(features) {}
    
    // Generate idle robot data with realistic characteristics
    std::vector<float> generate_idle_frame(float time_sec) {
        std::vector<float> frame(num_features);
        
        std::normal_distribution<float> noise(0.0f, 0.001f);  // Small sensor noise
        
        for (size_t i = 0; i < num_features; ++i) {
            if (i < 36) {
                // Joint positions (12 joints × 3 values)
                // Idle positions are constant with tiny noise
                if (i % 3 == 0) {
                    // Position: fixed idle stance
                    frame[i] = 0.5f + noise(rng);
                } else if (i % 3 == 1) {
                    // Velocity: nearly zero
                    frame[i] = noise(rng);
                } else {
                    // Torque: gravity compensation + noise
                    frame[i] = 2.0f + noise(rng);
                }
            } else if (i >= 36 && i < 42) {
                // IMU (accel + gyro)
                if (i < 39) {
                    // Accelerometer: gravity vector + noise
                    if (i == 38) frame[i] = 9.81f + noise(rng);  // Z-axis
                    else frame[i] = noise(rng);
                } else {
                    // Gyroscope: tiny drift
                    frame[i] = noise(rng) * 0.1f;
                }
            } else if (i >= 42 && i < 45) {
                // Battery (SOC, current, voltage)
                if (i == 42) frame[i] = 0.85f + noise(rng) * 0.01f;  // SOC
                else if (i == 43) frame[i] = 0.5f + noise(rng);      // Current
                else frame[i] = 48.0f + noise(rng);                  // Voltage
            } else if (i >= 45 && i < 48) {
                // UWB position (stationary)
                frame[i] = 1.0f + noise(rng) * 0.01f;
            } else if (i >= 48 && i < 52) {
                // Foot forces (weight distribution)
                frame[i] = 20.0f + noise(rng);
            } else {
                // Other features: mostly zeros or constants
                if (i % 10 == 0) {
                    frame[i] = 1.0f + noise(rng);  // Some constant values
                } else {
                    frame[i] = noise(rng);          // Near zero
                }
            }
        }
        
        return frame;
    }
    
    // Generate movement data (walking, standing up, etc.)
    std::vector<float> generate_movement_frame(float time_sec, const std::string& movement_type) {
        std::vector<float> frame(num_features);
        
        std::normal_distribution<float> noise(0.0f, 0.01f);
        
        for (size_t i = 0; i < num_features; ++i) {
            if (i < 36) {
                // Joint positions with movement
                if (i % 3 == 0) {
                    // Position: sinusoidal movement
                    float freq = (i / 3 + 1) * 0.5f;
                    frame[i] = 0.5f + 0.3f * std::sin(time_sec * freq) + noise(rng);
                } else if (i % 3 == 1) {
                    // Velocity: derivative of position
                    float freq = (i / 3 + 1) * 0.5f;
                    frame[i] = 0.3f * freq * std::cos(time_sec * freq) + noise(rng);
                } else {
                    // Torque: varies with movement
                    frame[i] = 2.0f + 5.0f * std::sin(time_sec * 2.0f) + noise(rng);
                }
            } else if (i >= 36 && i < 42) {
                // IMU with movement
                if (i < 39) {
                    // Accelerometer: movement acceleration
                    frame[i] = (i == 38 ? 9.81f : 0.0f) + 
                              2.0f * std::sin(time_sec * 3.0f + i) + noise(rng);
                } else {
                    // Gyroscope: rotation during movement
                    frame[i] = 0.5f * std::sin(time_sec * 2.0f + i) + noise(rng);
                }
            } else if (i >= 48 && i < 52) {
                // Foot forces during movement
                float phase = (i - 48) * M_PI / 2;
                frame[i] = 20.0f + 15.0f * std::sin(time_sec * 4.0f + phase) + noise(rng);
            } else {
                // Other features similar to idle but with more variation
                frame[i] = generate_idle_frame(time_sec)[i] + 
                          0.1f * std::sin(time_sec + i) + noise(rng);
            }
        }
        
        return frame;
    }
    
    // Create a binary log file
    void create_log_file(const std::string& filename, 
                        const std::vector<std::vector<float>>& frames,
                        float fusion_rate = 50.0f) {
        std::ofstream file(filename, std::ios::binary);
        
        // Write header
        BinaryFileHeader header;
        header.feature_count = num_features;
        header.fusion_rate_hz = static_cast<uint32_t>(fusion_rate);
        write_header(file, header);
        
        // Write frames
        uint64_t timestamp_ns = 0;
        uint64_t dt_ns = static_cast<uint64_t>(1e9 / fusion_rate);
        
        for (const auto& frame_data : frames) {
            AutoencoderFrame frame;
            frame.timestamp_ns = timestamp_ns;
            frame.modality_mask = 0x1F;  // All modalities present
            
            for (int i = 0; i < 5; ++i) {
                frame.sample_counts[i] = 1;
                frame.staleness_ms[i] = 0.0f;  // Fresh data
            }
            
            std::copy(frame_data.begin(), frame_data.end(), frame.features);
            
            write_frame(file, frame);
            timestamp_ns += dt_ns;
        }
        
        file.close();
        std::cout << "Created log file: " << filename << " with " 
                  << frames.size() << " frames\n";
    }
};

// ============================================================================
// Main proof-of-concept application
// ============================================================================
int main(int argc, char** argv) {
    std::cout << "\n=========================================\n";
    std::cout << "  RoCA Idle Baseline Proof-of-Concept\n";
    std::cout << "=========================================\n\n";
    
    // Configuration
    IdleRoCAConfig config;
    config.T = 10;
    config.D = 256;
    config.C = 32;
    config.K = 16;
    config.epochs = 50;  // Fewer epochs for PoC
    config.batch_size = 32;
    
    bool use_synthetic = true;
    std::string data_file = "idle_data.bin";
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--file" && i + 1 < argc) {
            data_file = argv[++i];
            use_synthetic = false;
        } else if (arg == "--synthetic") {
            use_synthetic = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --file <path>    Use existing binary log file\n";
            std::cout << "  --synthetic      Generate synthetic data (default)\n";
            return 0;
        }
    }
    
    // Step 1: Generate or load training data
    std::vector<std::vector<float>> training_windows;
    
    if (use_synthetic) {
        std::cout << "Generating synthetic idle data...\n";
        
        SyntheticDataGenerator generator(config.D);
        std::vector<std::vector<float>> idle_frames;
        
        // Generate 30 minutes of idle data at 50Hz
        float duration_sec = 30 * 60;  // 30 minutes
        float dt = 1.0f / 50.0f;       // 50 Hz
        
        for (float t = 0; t < duration_sec; t += dt) {
            idle_frames.push_back(generator.generate_idle_frame(t));
        }
        
        std::cout << "Generated " << idle_frames.size() << " idle frames\n";
        
        // Create windows
        WindowConfig window_cfg;
        window_cfg.T = config.T;
        window_cfg.stride = 5;
        window_cfg.D = config.D;
        
        WindowMaker maker(window_cfg);
        
        for (const auto& frame : idle_frames) {
            maker.push(frame);
            if (maker.ready()) {
                training_windows.push_back(maker.get_window());
            }
        }
        
        std::cout << "Created " << training_windows.size() << " training windows\n";
        
        // Save synthetic data for inspection
        generator.create_log_file("synthetic_idle.bin", idle_frames);
        
    } else {
        std::cout << "Loading data from: " << data_file << "\n";
        
        std::ifstream file(data_file, std::ios::binary);
        if (!file) {
            std::cerr << "Error: Cannot open file " << data_file << "\n";
            return 1;
        }
        
        BinaryFileHeader header;
        if (!read_header(file, header)) {
            std::cerr << "Error: Invalid file header\n";
            return 1;
        }
        
        config.D = header.feature_count;
        
        WindowConfig window_cfg;
        window_cfg.T = config.T;
        window_cfg.stride = 5;
        window_cfg.D = config.D;
        
        WindowMaker maker(window_cfg);
        
        AutoencoderFrame frame;
        while (read_frame(file, frame)) {
            std::vector<float> features(config.D);
            for (size_t i = 0; i < config.D; ++i) {
                features[i] = std::isnan(frame.features[i]) ? 0.0f : frame.features[i];
            }
            
            maker.push(features);
            if (maker.ready()) {
                training_windows.push_back(maker.get_window());
            }
        }
        
        std::cout << "Loaded " << training_windows.size() << " windows from file\n";
    }
    
    if (training_windows.empty()) {
        std::cerr << "Error: No training windows created\n";
        return 1;
    }
    
    // Step 2: Train the idle model
    std::cout << "\n--- Training Phase ---\n";
    
    IdleRoCAModel model(config);
    train_idle_model(model, training_windows, config);
    
    // Step 3: Test on idle data (should be normal)
    std::cout << "\n--- Testing on Idle Data ---\n";
    
    SyntheticDataGenerator test_generator(config.D, 123);  // Different seed
    
    int idle_anomalies = 0;
    int idle_tests = 100;
    
    for (int test = 0; test < idle_tests; ++test) {
        std::vector<float> test_window;
        for (int t = 0; t < config.T; ++t) {
            auto frame = test_generator.generate_idle_frame(test * 0.02f + t * 0.02f);
            test_window.insert(test_window.end(), frame.begin(), frame.end());
        }
        
        float score = model.score_window(test_window);
        if (score > model.anomaly_threshold) {
            idle_anomalies++;
        }
    }
    
    std::cout << "Idle test results: " << idle_anomalies << "/" << idle_tests 
              << " flagged as anomalous (";
    std::cout << std::fixed << std::setprecision(1) 
              << (100.0f * idle_anomalies / idle_tests) << "% false positive rate)\n";
    
    // Step 4: Test on movement data (should be anomalous)
    std::cout << "\n--- Testing on Movement Data ---\n";
    
    std::vector<std::string> movements = {"walking", "standing_up", "turning"};
    
    for (const auto& movement : movements) {
        int movement_anomalies = 0;
        int movement_tests = 100;
        
        for (int test = 0; test < movement_tests; ++test) {
            std::vector<float> test_window;
            for (int t = 0; t < config.T; ++t) {
                auto frame = test_generator.generate_movement_frame(
                    test * 0.02f + t * 0.02f, movement);
                test_window.insert(test_window.end(), frame.begin(), frame.end());
            }
            
            float score = model.score_window(test_window);
            if (score > model.anomaly_threshold) {
                movement_anomalies++;
            }
        }
        
        std::cout << movement << " test results: " << movement_anomalies << "/" 
                  << movement_tests << " flagged as anomalous (";
        std::cout << std::fixed << std::setprecision(1) 
                  << (100.0f * movement_anomalies / movement_tests) << "% detection rate)\n";
    }
    
    // Step 5: Detailed analysis of scores
    std::cout << "\n--- Score Distribution Analysis ---\n";
    
    std::vector<float> idle_scores, movement_scores;
    
    // Collect idle scores
    for (int i = 0; i < 50; ++i) {
        std::vector<float> window;
        for (int t = 0; t < config.T; ++t) {
            auto frame = test_generator.generate_idle_frame(i * 0.02f + t * 0.02f);
            window.insert(window.end(), frame.begin(), frame.end());
        }
        idle_scores.push_back(model.score_window(window));
    }
    
    // Collect movement scores
    for (int i = 0; i < 50; ++i) {
        std::vector<float> window;
        for (int t = 0; t < config.T; ++t) {
            auto frame = test_generator.generate_movement_frame(i * 0.02f + t * 0.02f, "walking");
            window.insert(window.end(), frame.begin(), frame.end());
        }
        movement_scores.push_back(model.score_window(window));
    }
    
    // Compute statistics
    auto compute_stats = [](const std::vector<float>& scores) {
        float mean = std::accumulate(scores.begin(), scores.end(), 0.0f) / scores.size();
        float min_val = *std::min_element(scores.begin(), scores.end());
        float max_val = *std::max_element(scores.begin(), scores.end());
        return std::make_tuple(mean, min_val, max_val);
    };
    
    auto [idle_mean, idle_min, idle_max] = compute_stats(idle_scores);
    auto [move_mean, move_min, move_max] = compute_stats(movement_scores);
    
    std::cout << "\nIdle scores:\n";
    std::cout << "  Mean: " << std::fixed << std::setprecision(6) << idle_mean << "\n";
    std::cout << "  Range: [" << idle_min << ", " << idle_max << "]\n";
    
    std::cout << "\nMovement scores:\n";
    std::cout << "  Mean: " << std::fixed << std::setprecision(6) << move_mean << "\n";
    std::cout << "  Range: [" << move_min << ", " << move_max << "]\n";
    
    std::cout << "\nThreshold: " << model.anomaly_threshold << "\n";
    std::cout << "Separation ratio: " << (move_mean / idle_mean) << "x\n";
    
    // Success criteria
    std::cout << "\n=========================================\n";
    std::cout << "  Proof-of-Concept Results\n";
    std::cout << "=========================================\n";
    
    bool poc_success = (idle_anomalies < 10) &&  // <10% false positives
                       (move_mean > idle_mean * 2.0f);  // Clear separation
    
    if (poc_success) {
        std::cout << "✓ SUCCESS: Model successfully learned idle behavior!\n";
        std::cout << "  - Low false positive rate on idle data\n";
        std::cout << "  - Clear separation between idle and movement scores\n";
        std::cout << "  - Any movement is detected as anomalous\n";
    } else {
        std::cout << "⚠ PARTIAL SUCCESS: Model needs tuning\n";
        if (idle_anomalies >= 10) {
            std::cout << "  - High false positive rate, consider:\n";
            std::cout << "    * Increasing threshold multiplier\n";
            std::cout << "    * Adding more training data\n";
        }
        if (move_mean <= idle_mean * 2.0f) {
            std::cout << "  - Insufficient separation, consider:\n";
            std::cout << "    * Increasing invariance loss weight\n";
            std::cout << "    * Using more epochs\n";
        }
    }
    
    std::cout << "\nNext steps:\n";
    std::cout << "1. Test with real robot idle data\n";
    std::cout << "2. Verify detection of actual robot movements\n";
    std::cout << "3. Fine-tune threshold based on real data\n";
    std::cout << "4. Train separate models for different behaviors\n";
    
    return 0;
}