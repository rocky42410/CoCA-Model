// Debug program to verify COCA state issues
#include <iostream>
#include <vector>
#include <iomanip>
#include "../src/coca_model.hpp"
#include "../src/io/csv_reader.hpp"

using namespace coca;

int main(int argc, char** argv) {
    std::cout << "\n=== COCA Debug Test ===\n\n";
    
    // Create minimal config
    COCAConfig config;
    config.T = 10;
    config.D = 50;  // Small feature count for testing
    config.epochs = 5;  // Quick training
    config.batch_size = 16;
    
    // Create synthetic data - IDENTICAL windows
    std::cout << "1. Creating 100 IDENTICAL test windows...\n";
    std::vector<std::vector<float>> identical_windows;
    
    // Create one pattern
    std::vector<float> pattern(config.T * config.D);
    for (size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = std::sin(i * 0.1f) * 0.5f + 0.5f;
    }
    
    // Replicate it 100 times
    for (int i = 0; i < 100; ++i) {
        identical_windows.push_back(pattern);
    }
    
    std::cout << "   Created " << identical_windows.size() << " identical windows\n\n";
    
    // Train model
    std::cout << "2. Training COCA model...\n";
    COCAModel model(config);
    train_coca_model(model, identical_windows, config);
    std::cout << "   Training complete\n\n";
    
    // Test scoring identical windows
    std::cout << "3. Testing scores on IDENTICAL windows:\n";
    std::cout << "   (These should all be EXACTLY the same if no state accumulation)\n\n";
    
    std::vector<float> scores;
    for (int i = 0; i < 20; ++i) {
        float score = model.score_window(identical_windows[0]);
        scores.push_back(score);
        std::cout << "   Window " << std::setw(2) << i << ": " 
                  << std::fixed << std::setprecision(8) << score;
        
        if (i > 0 && std::abs(score - scores[0]) > 1e-6) {
            std::cout << " ❌ DIFFERENT! (delta=" << (score - scores[0]) << ")";
        } else if (i > 0) {
            std::cout << " ✓ same";
        }
        std::cout << "\n";
    }
    
    // Check for monotonic increase
    std::cout << "\n4. Checking for monotonic behavior:\n";
    bool is_monotonic = true;
    int increases = 0, decreases = 0, same = 0;
    
    for (size_t i = 1; i < scores.size(); ++i) {
        if (scores[i] > scores[i-1] + 1e-8) {
            increases++;
            std::cout << "   ↑";
        } else if (scores[i] < scores[i-1] - 1e-8) {
            decreases++;
            is_monotonic = false;
            std::cout << "   ↓";
        } else {
            same++;
            std::cout << "   =";
        }
    }
    
    std::cout << "\n\n   Results: " << increases << " increases, " 
              << decreases << " decreases, " << same << " unchanged\n";
    
    if (increases > 0 && decreases == 0) {
        std::cout << "\n   🚨 CRITICAL: Scores are monotonically increasing!\n";
        std::cout << "      State accumulation bug is ACTIVE!\n";
    } else if (same == scores.size() - 1) {
        std::cout << "\n   ✅ GOOD: All scores are identical (expected for identical inputs)\n";
    } else {
        std::cout << "\n   ⚠️  WARNING: Scores are varying for identical inputs!\n";
    }
    
    // Test with different windows
    std::cout << "\n5. Testing with DIFFERENT windows:\n";
    std::vector<float> varied_scores;
    
    for (int i = 0; i < 10; ++i) {
        // Create slightly different window
        std::vector<float> varied_window = pattern;
        for (size_t j = 0; j < 10; ++j) {
            varied_window[j] += i * 0.01f;  // Small variation
        }
        
        float score = model.score_window(varied_window);
        varied_scores.push_back(score);
        std::cout << "   Variation " << i << ": " 
                  << std::fixed << std::setprecision(8) << score << "\n";
    }
    
    // Final diagnosis
    std::cout << "\n=== DIAGNOSIS ===\n";
    
    float range = *std::max_element(scores.begin(), scores.end()) - 
                  *std::min_element(scores.begin(), scores.end());
                  
    if (range < 1e-6) {
        std::cout << "✅ Identical inputs produce identical scores - GOOD\n";
    } else {
        std::cout << "❌ Identical inputs produce different scores - BAD\n";
        std::cout << "   Score range: " << range << " (should be ~0)\n";
        std::cout << "   This indicates state accumulation between forward passes\n";
    }
    
    float varied_range = *std::max_element(varied_scores.begin(), varied_scores.end()) - 
                        *std::min_element(varied_scores.begin(), varied_scores.end());
                        
    if (varied_range > 1e-4) {
        std::cout << "✅ Different inputs produce different scores - GOOD\n";
        std::cout << "   Variation range: " << varied_range << "\n";
    } else {
        std::cout << "⚠️  Different inputs produce similar scores\n";
        std::cout << "   Model might not be discriminative enough\n";
    }
    
    return 0;
}
