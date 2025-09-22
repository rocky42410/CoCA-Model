#!/bin/bash
# Comprehensive verification and fix script for COCA state accumulation bug

echo "============================================"
echo "COCA Bug Verification & Fix Script"
echo "============================================"
echo ""

# Function to check for the bug signatures
check_for_bug() {
    echo "Checking src/coca_model.hpp for bug signatures..."
    echo ""
    
    # Check if clear_caches method exists
    if grep -q "void clear_caches()" src/coca_model.hpp; then
        echo "✅ Found clear_caches() method"
        CLEAR_CACHES_EXISTS=1
    else
        echo "❌ Missing clear_caches() method - FIX NOT APPLIED"
        CLEAR_CACHES_EXISTS=0
    fi
    
    # Check if clear_all_caches exists
    if grep -q "void clear_all_caches()" src/coca_model.hpp; then
        echo "✅ Found clear_all_caches() method"
        CLEAR_ALL_CACHES_EXISTS=1
    else
        echo "❌ Missing clear_all_caches() method - FIX NOT APPLIED"
        CLEAR_ALL_CACHES_EXISTS=0
    fi
    
    # Check if is_training flag exists
    if grep -q "bool is_training = false;" src/coca_model.hpp; then
        echo "✅ Found is_training flag"
        IS_TRAINING_EXISTS=1
    else
        echo "❌ Missing is_training flag - FIX NOT APPLIED"
        IS_TRAINING_EXISTS=0
    fi
    
    # Check if score_window calls clear_all_caches
    if grep -A5 "float score_window" src/coca_model.hpp | grep -q "clear_all_caches()"; then
        echo "✅ score_window() calls clear_all_caches()"
        SCORE_CLEARS=1
    else
        echo "❌ score_window() doesn't call clear_all_caches() - FIX NOT APPLIED"
        SCORE_CLEARS=0
    fi
    
    echo ""
    
    # Overall status
    if [ $CLEAR_CACHES_EXISTS -eq 1 ] && [ $CLEAR_ALL_CACHES_EXISTS -eq 1 ] && [ $IS_TRAINING_EXISTS -eq 1 ] && [ $SCORE_CLEARS -eq 1 ]; then
        echo "✅ ALL FIXES APPEAR TO BE PRESENT"
        return 0
    else
        echo "❌ FIXES NOT PROPERLY APPLIED"
        return 1
    fi
}

# Step 1: Check current state
echo "Step 1: Checking current implementation..."
echo "=========================================="
check_for_bug

if [ $? -ne 0 ]; then
    echo ""
    echo "Step 2: Applying the fix..."
    echo "=========================================="
    
    # Backup current version
    cp src/coca_model.hpp src/coca_model_backup_$(date +%Y%m%d_%H%M%S).hpp
    echo "✓ Backed up current version"
    
    # Apply the fix
    cp /mnt/user-data/outputs/coca_model_fixed.hpp src/coca_model.hpp
    echo "✓ Applied fixed version"
    
    echo ""
    echo "Step 3: Verifying fix was applied..."
    echo "=========================================="
    check_for_bug
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Fix still not present after copying!"
        echo "Check file permissions or path issues."
        exit 1
    fi
else
    echo ""
    echo "Fixes appear to be present in the code."
    echo "Checking if rebuild is needed..."
fi

echo ""
echo "Step 4: Rebuilding with fixed code..."
echo "=========================================="

cd build
if [ -f "coca_train" ]; then
    TRAIN_OLD_TIME=$(stat -c %Y coca_train 2>/dev/null || stat -f %m coca_train 2>/dev/null)
fi

echo "Cleaning old build..."
make clean > /dev/null 2>&1
rm -f CMakeCache.txt

echo "Reconfiguring..."
cmake .. > cmake_log.txt 2>&1
if [ $? -ne 0 ]; then
    echo "❌ CMake configuration failed! Check cmake_log.txt"
    exit 1
fi

echo "Building..."
make -j$(nproc) > build_log.txt 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Build failed! Check build_log.txt"
    exit 1
fi

if [ -f "coca_train" ]; then
    TRAIN_NEW_TIME=$(stat -c %Y coca_train 2>/dev/null || stat -f %m coca_train 2>/dev/null)
    if [ "$TRAIN_OLD_TIME" != "$TRAIN_NEW_TIME" ]; then
        echo "✅ Executables successfully rebuilt"
    else
        echo "⚠️  Warning: Executables may not have been rebuilt"
    fi
fi

echo ""
echo "Step 5: Compiling debug test..."
echo "=========================================="

g++ -std=c++17 -O3 -I../src -o coca_debug ../debug_coca.cpp
if [ $? -eq 0 ]; then
    echo "✅ Debug test compiled"
else
    echo "❌ Failed to compile debug test"
fi

echo ""
echo "Step 6: Running debug test..."
echo "=========================================="

if [ -f "coca_debug" ]; then
    ./coca_debug
else
    echo "Debug test not available"
fi

echo ""
echo "Step 7: Testing on your data..."
echo "=========================================="

# Test with a small portion of your data
if [ -f "../anom_test_reversed.csv" ]; then
    echo "Training new model with fixed code..."
    ./coca_train --csv ../anom_test.csv --window 10 --stride 10 --output fixed_model.coca --epochs 10 > train_log.txt 2>&1
    
    if [ -f "fixed_model.coca" ]; then
        echo "Testing on reversed data..."
        ./coca_test --csv ../anom_test_reversed.csv --model fixed_model.coca --window 10 --stride 10 > test_log.txt 2>&1
        
        # Analyze the scores
        echo ""
        echo "Analyzing score behavior..."
        python3 << 'EOF'
import csv

try:
    scores = []
    with open('test_results.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row['score']))
    
    if len(scores) > 0:
        # Check for stuck values
        stuck_count = 0
        last_score = scores[0]
        max_stuck = 0
        current_stuck = 1
        
        for score in scores[1:]:
            if abs(score - last_score) < 1e-8:
                current_stuck += 1
                max_stuck = max(max_stuck, current_stuck)
            else:
                current_stuck = 1
            last_score = score
        
        # Check for monotonic increase
        increases = sum(1 for i in range(1, len(scores)) if scores[i] > scores[i-1])
        decreases = sum(1 for i in range(1, len(scores)) if scores[i] < scores[i-1])
        
        print(f"Score Analysis:")
        print(f"  Total windows: {len(scores)}")
        print(f"  Score range: [{min(scores):.6f}, {max(scores):.6f}]")
        print(f"  Increases: {increases}/{len(scores)-1} ({100*increases/(len(scores)-1):.1f}%)")
        print(f"  Decreases: {decreases}/{len(scores)-1} ({100*decreases/(len(scores)-1):.1f}%)")
        print(f"  Max consecutive identical scores: {max_stuck}")
        
        if max_stuck > 100:
            print("\n❌ PROBLEM: Scores stuck at same value for >100 windows!")
        elif increases > 0.9 * (len(scores) - 1):
            print("\n❌ PROBLEM: Scores still mostly increasing!")
        elif decreases < 0.1 * (len(scores) - 1):
            print("\n⚠️  WARNING: Very few decreases in scores")
        else:
            print("\n✅ Scores appear to be varying normally")
            
except Exception as e:
    print(f"Could not analyze scores: {e}")
EOF
    else
        echo "Model training failed"
    fi
else
    echo "Test data not found"
fi

echo ""
echo "============================================"
echo "Verification Complete"
echo "============================================"
echo ""
echo "IMPORTANT NEXT STEPS:"
echo "1. If fixes were just applied, you MUST retrain your model"
echo "2. Old models trained with buggy code will still have the bug"
echo "3. Only use models trained AFTER applying and rebuilding with the fix"
echo ""
echo "To train a new model:"
echo "  ./coca_train --csv your_data.csv --window 10 --stride 5"
echo ""
echo "To test the new model:"
echo "  ./coca_test --csv test_data.csv --model trained_model.coca"
