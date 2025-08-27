#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../src/io/binary_log.hpp"

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    
    // Load valid features
    std::vector<size_t> valid_features;
    std::ifstream feat_file("valid_features.txt");
    std::string line;
    while (std::getline(feat_file, line)) {
        if (line[0] != '#') {
            valid_features.push_back(std::stoul(line));
        }
    }
    
    size_t num_features = valid_features.size();
    
    BinaryLogReader reader;
    reader.open(argv[1]);
    
    // Load 1000 samples
    std::vector<std::vector<float>> samples;
    AutoencoderFrame frame;
    
    for (int i = 0; i < 1000 && reader.read_frame(frame); ++i) {
        std::vector<float> sample;
        for (size_t idx : valid_features) {
            sample.push_back(frame.features[idx]);
        }
        samples.push_back(sample);
    }
    
    // Convert to Eigen matrix
    Eigen::MatrixXf data(samples.size(), num_features);
    for (size_t i = 0; i < samples.size(); ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            data(i, j) = samples[i][j];
        }
    }
    
    // Center the data
    Eigen::VectorXf mean = data.colwise().mean();
    Eigen::MatrixXf centered = data.rowwise() - mean.transpose();
    
    // Compute covariance
    Eigen::MatrixXf cov = (centered.transpose() * centered) / float(samples.size() - 1);
    
    // Eigendecomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(cov);
    Eigen::VectorXf eigenvalues = solver.eigenvalues();
    
    // Sort eigenvalues
    std::vector<float> sorted_eigenvalues;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        sorted_eigenvalues.push_back(eigenvalues(i));
    }
    std::sort(sorted_eigenvalues.rbegin(), sorted_eigenvalues.rend());
    
    // Calculate explained variance
    float total_var = 0;
    for (float ev : sorted_eigenvalues) total_var += ev;
    
    std::cout << "=== PCA Analysis ===\n";
    std::cout << "Total variance: " << total_var << "\n\n";
    
    float cumulative = 0;
    for (size_t i = 0; i < std::min(size_t(20), sorted_eigenvalues.size()); ++i) {
        cumulative += sorted_eigenvalues[i];
        float pct = (sorted_eigenvalues[i] / total_var) * 100;
        float cum_pct = (cumulative / total_var) * 100;
        std::cout << "PC" << (i+1) << ": " << pct << "% (cumulative: " << cum_pct << "%)\n";
    }
    
    // How many components for 95% variance?
    cumulative = 0;
    int components_needed = 0;
    for (size_t i = 0; i < sorted_eigenvalues.size(); ++i) {
        cumulative += sorted_eigenvalues[i];
        if (cumulative / total_var > 0.95) {
            components_needed = i + 1;
            break;
        }
    }
    
    std::cout << "\nComponents needed for 95% variance: " << components_needed << "/" << num_features << "\n";
    
    if (components_needed > num_features * 0.5) {
        std::cout << "WARNING: Data is not easily compressible - might be mostly noise\n";
    }
    
    return 0;
}