#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "../src/io/binary_log.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data.bin>\n";
        return 1;
    }
    
    BinaryLogReader reader;
    reader.open(argv[1]);
    
    // Read just first 100 frames for detailed analysis
    AutoencoderFrame frame;
    std::vector<std::vector<float>> frames;
    
    for (int i = 0; i < 100 && reader.read_frame(frame); ++i) {
        std::vector<float> features;
        for (int j = 0; j < 256; ++j) {
            features.push_back(frame.features[j]);
        }
        frames.push_back(features);
    }
    
    std::cout << "=== Detailed Feature Analysis (First 100 frames) ===\n\n";
    
    // Map feature indices to likely meanings based on your logger
    std::map<int, std::string> feature_names = {
        {0, "ACCEL_X_MEAN"}, {1, "ACCEL_X_STD"}, {2, "ACCEL_X_MAX"},
        {3, "ACCEL_Y_MEAN"}, {4, "ACCEL_Y_STD"}, {5, "ACCEL_Y_MAX"},
        {6, "ACCEL_Z_MEAN"}, {7, "ACCEL_Z_STD"}, {8, "ACCEL_Z_MAX"},
        {9, "GYRO_X_MEAN"}, {10, "GYRO_X_STD"}, {11, "GYRO_X_MAX"},
        {18, "POS_X"}, {19, "POS_Y"}, {20, "POS_Z"},
        {21, "VEL_X"}, {22, "VEL_Y"}, {23, "VEL_Z"},
        {24, "ROLL"}, {25, "PITCH"}, {26, "YAW"},
        {27, "YAW_SPEED"},
        {28, "FOOT_FORCE_0"}, {29, "FOOT_FORCE_1"}, 
        {30, "FOOT_FORCE_2"}, {31, "FOOT_FORCE_3"},
        {32, "BODY_HEIGHT"}, {33, "MODE"}, {34, "GAIT_TYPE"},
        {36, "POWER_V_MEAN"}, {38, "POWER_A_MEAN"},
        {40, "MOTOR_TORQUE_0"}, {52, "MOTOR_VEL_0"}, {64, "MOTOR_TEMP_0"}
    };
    
    // Analyze each feature
    for (int f = 0; f < 87; ++f) {  // Check first 87 features
        std::vector<float> values;
        bool has_nan = false;
        
        for (const auto& frame_features : frames) {
            if (f < frame_features.size()) {
                float val = frame_features[f];
                if (std::isnan(val)) {
                    has_nan = true;
                } else {
                    values.push_back(val);
                }
            }
        }
        
        if (values.empty()) continue;
        
        float min_val = *std::min_element(values.begin(), values.end());
        float max_val = *std::max_element(values.begin(), values.end());
        float mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
        
        // Only show non-NaN features with concerning values
        if (!has_nan && (std::abs(min_val) > 100 || std::abs(max_val) > 100 || 
            (max_val == min_val && max_val != 0))) {
            
            std::string name = feature_names.count(f) ? feature_names[f] : "UNKNOWN";
            std::cout << "Feature " << f << " (" << name << "):\n";
            std::cout << "  Range: [" << min_val << ", " << max_val << "]\n";
            std::cout << "  Mean: " << mean << "\n";
            
            // Show first few values
            std::cout << "  First values: ";
            for (int i = 0; i < std::min(5, (int)values.size()); ++i) {
                std::cout << values[i] << " ";
            }
            std::cout << "\n\n";
        }
    }
    
    // Check what's actually changing in "idle" data
    std::cout << "=== Variance Analysis (should be low for idle) ===\n";
    for (int f = 0; f < 40; ++f) {
        std::vector<float> values;
        for (const auto& frame_features : frames) {
            if (!std::isnan(frame_features[f])) {
                values.push_back(frame_features[f]);
            }
        }
        
        if (values.size() > 1) {
            float mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
            float variance = 0;
            for (float v : values) {
                variance += (v - mean) * (v - mean);
            }
            variance /= values.size();
            
            if (variance > 0.01 && std::abs(mean) < 100) {  // Skip corrupt features
                std::string name = feature_names.count(f) ? feature_names[f] : "UNKNOWN";
                std::cout << "Feature " << f << " (" << name << "): variance=" 
                         << variance << ", std=" << std::sqrt(variance) << "\n";
            }
        }
    }
    
    reader.close();
    return 0;
}