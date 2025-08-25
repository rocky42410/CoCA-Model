CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -pthread -Wall -Wextra
INCLUDES = -I./src
BUILDDIR = build
SRCDIR = src
APPDIR = apps

# Targets
all: diagnostic enhanced train_one_class test_one_class batch_eval live_analyzer

# Diagnostic tools
diagnostic: $(BUILDDIR)/roca_diagnostic

$(BUILDDIR)/roca_diagnostic: $(APPDIR)/roca_diagnostic.cpp
	@echo "Building diagnostic tool..."
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@
	@echo "✓ Diagnostic tool built"

enhanced: $(BUILDDIR)/roca_diagnostic_enhanced

$(BUILDDIR)/roca_diagnostic_enhanced: $(APPDIR)/roca_diagnostic_enhanced.cpp
	@echo "Building enhanced diagnostic..."
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@
	@echo "✓ Enhanced diagnostic built"

# One-class training
train_one_class: $(BUILDDIR)/train_one_class

$(BUILDDIR)/train_one_class: $(APPDIR)/train_one_class.cpp
	@echo "Building one-class trainer..."
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ -lstdc++fs
	@echo "✓ One-class trainer built"

# One-class testing
test_one_class: $(BUILDDIR)/test_one_class

$(BUILDDIR)/test_one_class: $(APPDIR)/test_one_class.cpp
	@echo "Building one-class tester..."
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@
	@echo "✓ One-class tester built"

# Batch evaluator
batch_eval: $(BUILDDIR)/roca_batch_eval

$(BUILDDIR)/roca_batch_eval: $(APPDIR)/roca_batch_eval.cpp
	@echo "Building batch evaluator..."
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@
	@echo "✓ Batch evaluator built"

# Live analyzer
live_analyzer: $(BUILDDIR)/roca_live

$(BUILDDIR)/roca_live: $(APPDIR)/roca_live_filtered.cpp
	@echo "Building live analyzer..."
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@
	@echo "✓ Live analyzer built"

clean:
	rm -rf $(BUILDDIR)/*

.PHONY: all clean diagnostic enhanced train_one_class test_one_class batch_eval live_analyzer
