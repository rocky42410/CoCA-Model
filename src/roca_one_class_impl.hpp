// ============================================================================
// roca_one_class_impl.hpp - Implementation of model I/O methods
// ============================================================================
#pragma once
#include "roca_one_class.hpp"
#include "model_io_complete.hpp"
#include <filesystem>

namespace roca {

// Implementation of save_model for OneClassRoCA
bool OneClassRoCA::save_model(const std::string& filepath) {
    ModelMetadata metadata;
    metadata.T = config.T;
    metadata.D = config.D;
    metadata.C = config.C;
    metadata.K = config.K;
    metadata.anomaly_threshold = anomaly_threshold;
    metadata.num_valid_features = processor.get_valid_indices().size();
    
    // Collect encoder weights and biases
    std::vector<std::vector<float>> encoder_weights = {
        encoder1.W, encoder2.W, encoder3.W
    };
    std::vector<std::vector<float>> encoder_biases = {
        encoder1.b, encoder2.b, encoder3.b
    };
    
    // Collect decoder weights and biases
    std::vector<std::vector<float>> decoder_weights = { decoder.W };
    std::vector<std::vector<float>> decoder_biases = { decoder.b };
    
    return ModelSerializer::save_model(
        filepath,
        metadata,
        processor.get_mean(),
        processor.get_std(),
        processor.get_constant_mask(),
        processor.get_valid_indices(),
        encoder_weights,
        encoder_biases,
        decoder_weights,
        decoder_biases,
        projector.W,
        projector.b,
        center  // Use 'center' instead of 'Ce' for one-class model
    );
}

bool OneClassRoCA::load_model(const std::string& filepath) {
    ModelMetadata metadata;
    std::vector<float> feature_mean, feature_std;
    std::vector<bool> is_constant;
    std::vector<size_t> valid_indices;
    std::vector<std::vector<float>> encoder_weights, encoder_biases;
    std::vector<std::vector<float>> decoder_weights, decoder_biases;
    std::vector<float> projector_weights, projector_bias;
    std::vector<float> loaded_center;
    
    if (!ModelSerializer::load_model(
            filepath,
            metadata,
            feature_mean,
            feature_std,
            is_constant,
            valid_indices,
            encoder_weights,
            encoder_biases,
            decoder_weights,
            decoder_biases,
            projector_weights,
            projector_bias,
            loaded_center)) {
        return false;
    }
    
    // Update config
    config.T = metadata.T;
    config.D = metadata.D;
    config.C = metadata.C;
    config.K = metadata.K;
    anomaly_threshold = metadata.anomaly_threshold;
    
    // Recreate layers with correct dimensions
    encoder1 = DenseLayer(config.T * config.D, 128, true);
    encoder2 = DenseLayer(128, 64, true);
    encoder3 = DenseLayer(64, config.C, false);
    decoder = DenseLayer(config.C, config.T * config.D, false);
    projector = DenseLayer(config.C, config.K, false);
    
    // Load weights
    encoder1.W = encoder_weights[0];
    encoder1.b = encoder_biases[0];
    encoder2.W = encoder_weights[1];
    encoder2.b = encoder_biases[1];
    encoder3.W = encoder_weights[2];
    encoder3.b = encoder_biases[2];
    decoder.W = decoder_weights[0];
    decoder.b = decoder_biases[0];
    projector.W = projector_weights;
    projector.b = projector_bias;
    
    // Load center
    center = loaded_center;
    
    // Update processor
    processor.set_stats(feature_mean, feature_std, is_constant, valid_indices);
    
    return true;
}

} // namespace roca