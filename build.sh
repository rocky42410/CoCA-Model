#!/bin/bash
# Build and run script for RoCA idle baseline proof-of-concept

set -e  # Exit on error

echo "=================================="
echo "RoCA Idle Baseline Build Script"
echo "=================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required directories
if [ ! -d "src" ]; then
    echo -e "${YELLOW}Creating source directory structure...${NC}"
    mkdir -p src/{io,math,model,train,data,augment,serialize}
fi

if [ ! -d "apps" ]; then
    mkdir -p apps
fi

if [ ! -d "build" ]; then
    mkdir -p build
fi

# Function to build a target
build_target() {
    local target=$1
    local source=$2
    local output=$3
    
    echo -e "${GREEN}Building $target...${NC}"
    g++ -std=c++17 -O3 -march=native -pthread \
        -I./src \
        -Wall -Wextra -Wno-unused-parameter \
        $source \
        -o $output
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $target built successfully${NC}"
    else
        echo -e "${RED}✗ Failed to build $target${NC}"
        exit 1
    fi
}

# Build based on argument
case "${1:-all}" in
    diagnostic)
        build_target "Diagnostic Tool" "apps/roca_diagnostic.cpp" "build/roca_diagnostic"
        ;;
    
    poc)
        build_target "Idle PoC" "apps/idle_poc.cpp" "build/idle_poc"
        ;;
    
    trainer)
        build_target "Fixed Trainer" "apps/roca_trainer_fixed.cpp" "build/roca_trainer"
        ;;
    
    all)
        build_target "Diagnostic Tool" "apps/roca_diagnostic.cpp" "build/roca_diagnostic"
        build_target "Idle PoC" "apps/idle_poc.cpp" "build/idle_poc"
        echo -e "${GREEN}All targets built successfully!${NC}"
        ;;
    
    run-poc)
        # Build and run the PoC
        build_target "Idle PoC" "apps/idle_poc.cpp" "build/idle_poc"
        echo
        echo "=================================="
        echo "Running Idle Baseline PoC"
        echo "=================================="
        echo
        ./build/idle_poc --synthetic
        ;;
    
    run-diagnostic)
        # Check if a file was provided
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Please provide a binary log file${NC}"
            echo "Usage: $0 run-diagnostic <binary_log_file>"
            exit 1
        fi
        
        build_target "Diagnostic Tool" "apps/roca_diagnostic.cpp" "build/roca_diagnostic"
        echo
        echo "=================================="
        echo "Running Diagnostic on $2"
        echo "=================================="
        echo
        ./build/roca_diagnostic "$2"
        ;;
    
    test-real)
        # Test with real data
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Please provide a binary log file${NC}"
            echo "Usage: $0 test-real <idle_data.bin>"
            exit 1
        fi
        
        # First run diagnostic
        build_target "Diagnostic Tool" "apps/roca_diagnostic.cpp" "build/roca_diagnostic"
        echo
        echo "Step 1: Analyzing data quality..."
        echo "=================================="
        ./build/roca_diagnostic "$2"
        
        echo
        echo -e "${YELLOW}Press Enter to continue with training...${NC}"
        read
        
        # Then run PoC with real data
        build_target "Idle PoC" "apps/idle_poc.cpp" "build/idle_poc"
        echo
        echo "Step 2: Training on real idle data..."
        echo "======================================"
        ./build/idle_poc --file "$2"
        ;;
    
    clean)
        echo -e "${YELLOW}Cleaning build directory...${NC}"
        rm -rf build/*
        echo -e "${GREEN}✓ Clean complete${NC}"
        ;;
    
    help|*)
        echo "Usage: $0 [command] [args]"
        echo
        echo "Commands:"
        echo "  diagnostic        Build diagnostic tool"
        echo "  poc              Build idle PoC application"
        echo "  trainer          Build fixed trainer"
        echo "  all              Build all targets (default)"
        echo "  run-poc          Build and run PoC with synthetic data"
        echo "  run-diagnostic   Build and run diagnostic on a file"
        echo "  test-real        Run full pipeline on real data"
        echo "  clean            Clean build directory"
        echo "  help             Show this help"
        echo
        echo "Examples:"
        echo "  $0 run-poc                    # Quick test with synthetic data"
        echo "  $0 run-diagnostic data.bin    # Analyze your data file"
        echo "  $0 test-real idle_30min.bin   # Full test on real data"
        ;;
esac

echo
echo "=================================="
echo "Build script complete"
echo "==================================">