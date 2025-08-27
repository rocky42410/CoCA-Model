#include <iostream>
#include <vector>
#include "../src/io/binary_log.hpp"

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    
    BinaryLogReader reader;
    reader.open(argv[1]);
    
    AutoencoderFrame frame;
    int frame_num = 0;
    int first_corrupt = -1;
    int last_clean = -1;
    
    std::cout << "Scanning for data corruption...\n";
    
    while (reader.read_frame(frame)) {
        bool corrupt = false;
        
        // Check for extreme values
        for (int i = 0; i < 256; ++i) {
            if (!std::isnan(frame.features[i]) && 
                (frame.features[i] < -1000 || frame.features[i] > 1000)) {
                corrupt = true;
                if (first_corrupt == -1) {
                    first_corrupt = frame_num;
                    std::cout << "\nFirst corruption at frame " << frame_num << ":\n";
                    std::cout << "  Feature " << i << " = " << frame.features[i] << "\n";
                    
                    // Show a few more corrupt values
                    for (int j = i+1; j < 256 && j < i+10; ++j) {
                        if (frame.features[j] < -1000 || frame.features[j] > 1000) {
                            std::cout << "  Feature " << j << " = " << frame.features[j] << "\n";
                        }
                    }
                }
                break;
            }
        }
        
        if (!corrupt && first_corrupt == -1) {
            last_clean = frame_num;
        }
        
        frame_num++;
        
        if (frame_num % 10000 == 0) {
            std::cout << "Processed " << frame_num << " frames...\n";
        }
    }
    
    std::cout << "\n=== Results ===\n";
    std::cout << "Total frames: " << frame_num << "\n";
    
    if (first_corrupt == -1) {
        std::cout << "No corruption detected!\n";
    } else {
        std::cout << "Clean frames: 0-" << last_clean << " (" << (last_clean+1) << " frames)\n";
        std::cout << "First corrupt frame: " << first_corrupt << "\n";
        std::cout << "Corruption starts at " << (first_corrupt * 20) << "ms (" 
                  << (first_corrupt / 50.0) << " seconds)\n";
        
        // Calculate windows from clean portion
        int clean_windows = (last_clean - 10) / 5;  // window_size=10, stride=5
        std::cout << "\nUsable clean windows: ~" << clean_windows << "\n";
    }
    
    reader.close();
    return 0;
}