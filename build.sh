#!/bin/bash
# ============================================================================
# Build script for COCA Implementation
# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║      COCA Build Script                 ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}\n"

# Check for required tools
echo "Checking requirements..."

if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: cmake is not installed${NC}"
    exit 1
fi

if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo -e "${RED}Error: No C++ compiler found${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} All requirements met\n"

# Parse arguments
BUILD_TYPE="Release"
RUN_TESTS=false
CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --debug    Build in debug mode"
            echo "  --test     Run tests after building"
            echo "  --clean    Clean build directory first"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Clean if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning build directory..."
    rm -rf build/
    echo -e "${GREEN}✓${NC} Clean complete\n"
fi

# Create build directory
echo "Setting up build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE .. > cmake_output.log 2>&1

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ CMake configuration failed${NC}"
    echo "See build/cmake_output.log for details"
    exit 1
fi

echo -e "${GREEN}✓${NC} Configuration complete (Build type: $BUILD_TYPE)\n"

# Build
echo "Building COCA..."
make -j$(nproc) 2>&1 | tee build_output.log

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Build failed${NC}"
    echo "See build/build_output.log for details"
    exit 1
fi

echo -e "${GREEN}✓${NC} Build complete\n"

# List built executables
echo "Built executables:"
echo "  - coca_train:       Main training application"
echo "  - coca_test:        Testing application"
echo "  - coca_synthetic:   Synthetic data generator"
echo "  - coca_unit_tests:  Unit tests"
echo ""

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    echo -e "\n${YELLOW}Running tests...${NC}"
    
    # Run unit tests
    echo "Running unit tests..."
    ./coca_unit_tests
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Unit tests passed\n"
    else
        echo -e "${RED}✗ Unit tests failed${NC}\n"
        exit 1
    fi
    
    # Run synthetic test
    echo "Running synthetic data test..."
    ./coca_synthetic --test
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} Synthetic test passed\n"
    else
        echo -e "${RED}✗ Synthetic test failed${NC}\n"
        exit 1
    fi
fi

# Create necessary directories
echo "Creating output directories..."
mkdir -p models logs data
echo -e "${GREEN}✓${NC} Directories created\n"

# Copy config file to build directory
cp ../coca_config.yaml .
echo -e "${GREEN}✓${NC} Configuration file copied\n"

# Success message
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         BUILD SUCCESSFUL               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}\n"

echo "Quick start:"
echo "  1. Generate synthetic data:"
echo "     ./coca_synthetic --normal 10000 --anomaly 1000"
echo ""
echo "  2. Train model:"
echo "     ./coca_train --data synthetic_train.bin"
echo ""
echo "  3. Test model:"
echo "     ./coca_test --test synthetic_test.bin --labels synthetic_labels.txt"
echo ""

cd ..
