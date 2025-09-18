
// ============================================================================
// data/window_maker.hpp - Sliding window generation
// ============================================================================
#pragma once
#include <vector>
#include <deque>

namespace roca {

struct WindowConfig {
    size_t T;        // Window length
    size_t stride;   // Stride
    size_t D;        // Feature dimension
};

class WindowMaker {
private:
    WindowConfig config;
    std::deque<std::vector<float>> buffer;
    size_t frames_since_last = 0;
    
public:
    WindowMaker(const WindowConfig& cfg) : config(cfg) {}
    
    void push(const std::vector<float>& frame) {
        buffer.push_back(frame);
        if (buffer.size() > config.T) {
            buffer.pop_front();
        }
        frames_since_last++;
    }
    
    bool ready() const {
        return buffer.size() == config.T && frames_since_last >= config.stride;
    }
    
    std::vector<float> get_window() {
        std::vector<float> window;
        window.reserve(config.T * config.D);
        
        for (const auto& frame : buffer) {
            window.insert(window.end(), frame.begin(), frame.end());
        }
        
        frames_since_last = 0;
        return window;
    }
    
    void clear() {
        buffer.clear();
        frames_since_last = 0;
    }
};

} // namespace roca
