#pragma once
#include <fstream>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdint>

struct BinaryFileHeader {
    uint32_t magic;
    uint8_t version;
    uint8_t is_little_endian;
    uint16_t header_size;
    int64_t steady_origin_ns;
    int64_t system_origin_ns;
    uint32_t fusion_rate_hz;
    uint32_t feature_count;
    uint32_t reserved[7];
};

struct AutoencoderFrame {
    uint64_t timestamp_ns;
    uint32_t modality_mask;
    uint16_t sample_counts[5];
    float staleness_ms[5];
    float features[256];
    uint32_t crc32;
};

class BinaryLogReader {
private:
    std::ifstream file;
    BinaryFileHeader header;
    bool header_read = false;
    
public:
    bool open(const std::string& filename) {
        file.open(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return false;
        }
        
        // Read header
        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (header.magic != 0x474F3244) {
            std::cerr << "Invalid magic number\n";
            return false;
        }
        
        header_read = true;
        std::cout << "Opened binary log:\n";
        std::cout << "  Features: " << header.feature_count << "\n";
        std::cout << "  Fusion rate: " << header.fusion_rate_hz << " Hz\n";
        
        return true;
    }
    
    bool read_frame(AutoencoderFrame& frame) {
        if (!header_read) return false;
        return file.read(reinterpret_cast<char*>(&frame), sizeof(frame)).good();
    }
    
    BinaryFileHeader get_header() const { return header; }
    
    void close() {
        file.close();
    }
    
    // Read all frames and extract feature windows
    std::vector<std::vector<float>> read_all_windows(size_t window_size, size_t stride) {
        std::vector<std::vector<float>> windows;
        std::vector<AutoencoderFrame> buffer;
        AutoencoderFrame frame;
        
        size_t frame_count = 0;
        while (read_frame(frame)) {
            buffer.push_back(frame);
            frame_count++;
            
            // Create window if we have enough frames
            if (buffer.size() >= window_size) {
                // Check if this is a stride point
                if ((buffer.size() - window_size) % stride == 0) {
                    std::vector<float> window;
                    window.reserve(window_size * 256);
                    
                    // Extract features from window
                    for (size_t t = buffer.size() - window_size; t < buffer.size(); ++t) {
                        for (size_t f = 0; f < 256; ++f) {
                            window.push_back(buffer[t].features[f]);
                        }
                    }
                    
                    windows.push_back(window);
                }
                
                // Remove old frames
                if (buffer.size() > window_size + stride) {
                    buffer.erase(buffer.begin(), buffer.begin() + stride);
                }
            }
        }
        
        std::cout << "Read " << frame_count << " frames\n";
        std::cout << "Created " << windows.size() << " windows\n";
        
        return windows;
    }
    
    // Count NaN statistics
    void analyze_nan_statistics() {
        AutoencoderFrame frame;
        int total_frames = 0;
        std::vector<int> nan_counts(256, 0);
        
        // Reset to beginning
        file.clear();
        file.seekg(sizeof(BinaryFileHeader));
        
        while (read_frame(frame)) {
            total_frames++;
            for (int i = 0; i < 256; ++i) {
                if (std::isnan(frame.features[i]) || std::isinf(frame.features[i])) {
                    nan_counts[i]++;
                }
            }
        }
        
        std::cout << "\n=== NaN Statistics ===\n";
        std::cout << "Total frames: " << total_frames << "\n";
        
        int always_nan = 0, never_nan = 0, sometimes_nan = 0;
        for (int i = 0; i < 256; ++i) {
            if (nan_counts[i] == total_frames) {
                always_nan++;
            } else if (nan_counts[i] == 0) {
                never_nan++;
            } else {
                sometimes_nan++;
            }
        }
        
        std::cout << "Features always NaN: " << always_nan << "\n";
        std::cout << "Features never NaN: " << never_nan << "\n";
        std::cout << "Features sometimes NaN: " << sometimes_nan << "\n";
        
        // Reset to beginning for next read
        file.clear();
        file.seekg(sizeof(BinaryFileHeader));
    }
};