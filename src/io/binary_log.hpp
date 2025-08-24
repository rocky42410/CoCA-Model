// ============================================================================
// io/binary_log.hpp - Binary log format for telemetry data
// ============================================================================
#pragma once
#include <cstdint>
#include <iostream>
#include <vector>
#include <cstring>
#include <array>

namespace roca {

constexpr uint32_t MAGIC_GO2D = 0x474F3244;
constexpr uint8_t VERSION = 2;

struct BinaryFileHeader {
    uint32_t magic = MAGIC_GO2D;
    uint8_t  version = VERSION;
    uint8_t  is_little_endian = 1;
    uint16_t header_size = sizeof(BinaryFileHeader);
    int64_t  steady_origin_ns = 0;
    int64_t  system_origin_ns = 0;
    uint32_t fusion_rate_hz = 50;
    uint32_t feature_count = 0;
    uint32_t reserved[7] = {0};
};

struct AutoencoderFrame {
    uint64_t timestamp_ns = 0;
    uint32_t modality_mask = 0;
    uint16_t sample_counts[5] = {0};
    float    staleness_ms[5] = {0};
    float    features[256] = {0};
    uint32_t crc32 = 0;
};

inline uint32_t crc32(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320 : 0);
        }
    }
    return ~crc;
}

inline bool read_header(std::istream& is, BinaryFileHeader& header) {
    is.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!is || header.magic != MAGIC_GO2D) return false;
    if (header.version != VERSION) return false;
    return true;
}

inline bool read_frame(std::istream& is, AutoencoderFrame& frame) {
    is.read(reinterpret_cast<char*>(&frame), sizeof(frame));
    if (!is) return false;
    
    // Verify CRC
    uint32_t computed_crc = crc32(
        reinterpret_cast<const uint8_t*>(&frame), 
        offsetof(AutoencoderFrame, crc32)
    );
    return computed_crc == frame.crc32;
}

inline void write_header(std::ostream& os, const BinaryFileHeader& header) {
    os.write(reinterpret_cast<const char*>(&header), sizeof(header));
}

inline void write_frame(std::ostream& os, AutoencoderFrame& frame) {
    frame.crc32 = crc32(
        reinterpret_cast<const uint8_t*>(&frame),
        offsetof(AutoencoderFrame, crc32)
    );
    os.write(reinterpret_cast<const char*>(&frame), sizeof(frame));
}

} // namespace roca