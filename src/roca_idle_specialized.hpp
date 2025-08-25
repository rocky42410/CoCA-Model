// ============================================================================
// Specialized RoCA implementation for idle baseline training
// Handles near-constant features while learning idle state representation
// ============================================================================

#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <iomanip>
#include <../src/model_io_complete.hpp>

namespace roca_idle {

// ============================================================================
// Configuration for idle-specialized model
// ============================================================================
struct IdleRoCAConfig {
    // Model architecture - smaller for idle data
    size_t T = 10;          // Window size
    size_t D = 256;         // Feature dimension  
    size_t C = 32;          // Latent dimension (smaller for simple idle patterns)
    size_t K = 16;          // Projection dimension
    
    // Training parameters - tuned for idle
    float lr = 5e-4f;                // Lower learning rate for stability
    float lambda_rec = 1.0f;         // Reconstruction weight
    float lambda_inv = 0.5f;         // Lower invariance weight initially
    float lambda_var = 0.05f;        // Lower variance regularization
    float zeta = 0.5f;               // Lower variance target for idle
    
    // Idle-specific parameters
    float noise_scale = 0.001f;      // Expected noise level in idle sensors
    float min_std = 0.01f;           // Minimum std for normalization
    float anomaly_multiplier = 3.0f; // Multiplier for threshold (3-sigma)
    
    size_t batch_size = 32;
    size_t epochs = 100;
    float val_split = 0.2f;
};

// ============================================================================
// Feature preprocessing for idle data
// ============================================================================
class IdleFeatureProcessor {
private:
    std::vector<float> feature_mean;
    std::vector<float> feature_std;
    std::vector<float> feature_noise_floor;
    std::vector<bool> is_constant;
    size_t D;
    std::vector<size_t> valid_indices; 
    
public:

// Add these methods to IdleFeatureProcessor class:
public:
    void set_stats(const std::vector<float>& mean, 
                   const std::vector<float>& std,
                   const std::vector<bool>& constant,
                   const std::vector<size_t>& valid_idx) {
        feature_mean = mean;
        feature_std = std;
        is_constant = constant;
        valid_indices = valid_idx;
        D = mean.size();
    }
    
    const std::vector<size_t>& get_valid_indices() const { return valid_indices; }


    void compute_stats(const std::vector<std::vector<float>>& windows, 
                      size_t T, size_t feature_dim) {
        D = feature_dim;
        feature_mean.resize(D, 0.0f);
        feature_std.resize(D, 0.0f);
        feature_noise_floor.resize(D, 0.0f);
        is_constant.resize(D, false);
        
        // First pass: compute mean
        size_t total_samples = 0;
        for (const auto& window : windows) {
            for (size_t t = 0; t < T; ++t) {
                for (size_t d = 0; d < D; ++d) {
                    float val = window[t * D + d];
                    if (!std::isnan(val)) {
                        feature_mean[d] += val;
                        total_samples++;
                    }
                }
            }
        }
        
        for (size_t d = 0; d < D; ++d) {
            feature_mean[d] /= (windows.size() * T);
        }
        
        // Second pass: compute std and detect constant features
        for (const auto& window : windows) {
            for (size_t t = 0; t < T; ++t) {
                for (size_t d = 0; d < D; ++d) {
                    float val = window[t * D + d];
                    if (!std::isnan(val)) {
                        float diff = val - feature_mean[d];
                        feature_std[d] += diff * diff;
                    }
                }
            }
        }
        
        size_t constant_count = 0;
        for (size_t d = 0; d < D; ++d) {
            feature_std[d] = std::sqrt(feature_std[d] / (windows.size() * T));
            
            // Detect constant features (very low variance)
            if (feature_std[d] < 1e-6f) {
                is_constant[d] = true;
                constant_count++;
                // For constant features, use a noise floor based on typical sensor precision
                feature_noise_floor[d] = std::abs(feature_mean[d]) * 0.001f + 0.001f;
                feature_std[d] = feature_noise_floor[d];
            } else {
                // For varying features, estimate noise floor as fraction of std
                feature_noise_floor[d] = feature_std[d] * 0.1f;
            }
        }
        
        std::cout << "\nIdle Feature Analysis:\n";
        std::cout << "  Total features: " << D << "\n";
        std::cout << "  Constant features: " << constant_count << " (" 
                  << (100.0f * constant_count / D) << "%)\n";
        std::cout << "  Variable features: " << (D - constant_count) << "\n";
        
        // Print sample of each type
        std::cout << "\nSample constant features (first 5):\n";
        int shown = 0;
        for (size_t d = 0; d < D && shown < 5; ++d) {
            if (is_constant[d]) {
                std::cout << "  Feature " << d << ": value=" << std::fixed 
                         << std::setprecision(4) << feature_mean[d] << "\n";
                shown++;
            }
        }
        
        std::cout << "\nSample variable features (first 5):\n";
        shown = 0;
        for (size_t d = 0; d < D && shown < 5; ++d) {
            if (!is_constant[d]) {
                std::cout << "  Feature " << d << ": mean=" << std::fixed 
                         << std::setprecision(4) << feature_mean[d]
                         << ", std=" << feature_std[d] << "\n";
                shown++;
            }
        }
    }
    
    std::vector<float> normalize(const std::vector<float>& window, size_t T) const {
        std::vector<float> normalized = window;
        
        for (size_t t = 0; t < T; ++t) {
            for (size_t d = 0; d < D; ++d) {
                size_t idx = t * D + d;
                
                if (std::isnan(normalized[idx])) {
                    normalized[idx] = 0.0f;
                    continue;
                }
                
                if (is_constant[d]) {
                    // For constant features, center around zero but preserve small variations
                    normalized[idx] = (normalized[idx] - feature_mean[d]) / feature_noise_floor[d];
                } else {
                    // Normal standardization for variable features
                    normalized[idx] = (normalized[idx] - feature_mean[d]) / feature_std[d];
                }
                
                // Clip to prevent extreme values
                normalized[idx] = std::max(-5.0f, std::min(5.0f, normalized[idx]));
            }
        }
        
        return normalized;
    }
    
    const std::vector<float>& get_mean() const { return feature_mean; }
    const std::vector<float>& get_std() const { return feature_std; }
    const std::vector<bool>& get_constant_mask() const { return is_constant; }
};

// ============================================================================
// Simplified dense layer with numerical stability
// ============================================================================
class StableDenseLayer {
public:
    std::vector<float> W;
    std::vector<float> b;
    std::vector<float> dW;
    std::vector<float> db;
    
    // Adam optimizer state
    std::vector<float> mW, vW;
    std::vector<float> mb, vb;
    int adam_t = 0;
    
    size_t in_dim;
    size_t out_dim;
    bool use_relu;
    float dropout_rate;
    
    // Cache for forward/backward
    std::vector<float> x_cache;
    std::vector<float> z_cache;
    std::vector<float> a_cache;
    std::vector<bool> dropout_mask;
    
    StableDenseLayer(size_t in, size_t out, bool relu = true, float dropout = 0.0f) 
        : in_dim(in), out_dim(out), use_relu(relu), dropout_rate(dropout) {
        
        size_t w_size = in * out;
        W.resize(w_size);
        b.resize(out);
        dW.resize(w_size);
        db.resize(out);
        
        // Adam state
        mW.resize(w_size, 0.0f);
        vW.resize(w_size, 0.0f);
        mb.resize(out, 0.0f);
        vb.resize(out, 0.0f);
    }
    
    void init_weights(std::mt19937& rng) {
        // He initialization with scale adjustment for idle data
        float scale = std::sqrt(2.0f / (in_dim + out_dim));  // Slightly more conservative
        std::normal_distribution<float> dist(0.0f, scale);
        
        for (auto& w : W) {
            w = dist(rng) * 0.5f;  // Scale down for idle data
        }
        
        // Small positive bias
        std::fill(b.begin(), b.end(), 0.01f);
    }
    
    std::vector<float> forward(const std::vector<float>& x, bool training = true) {
        size_t batch_size = x.size() / in_dim;
        x_cache = x;
        
        // Compute z = x @ W + b
        z_cache.resize(batch_size * out_dim);
        std::fill(z_cache.begin(), z_cache.end(), 0.0f);
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < in_dim; ++i) {
                for (size_t j = 0; j < out_dim; ++j) {
                    z_cache[b * out_dim + j] += x[b * in_dim + i] * W[i * out_dim + j];
                }
            }
            for (size_t j = 0; j < out_dim; ++j) {
                z_cache[b * out_dim + j] += this->b[j];
            }
        }
        
        // Apply activation
        a_cache = z_cache;
        if (use_relu) {
            for (auto& a : a_cache) {
                a = std::max(0.0f, a);
            }
        }
        
        // Apply dropout in training
        if (training && dropout_rate > 0.0f) {
            dropout_mask.resize(a_cache.size());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            std::mt19937 rng(42);  // Fixed seed for reproducibility
            
            for (size_t i = 0; i < a_cache.size(); ++i) {
                dropout_mask[i] = (dist(rng) > dropout_rate);
                if (!dropout_mask[i]) {
                    a_cache[i] = 0.0f;
                } else {
                    a_cache[i] /= (1.0f - dropout_rate);  // Scale up
                }
            }
        }
        
        return a_cache;
    }
    
    std::vector<float> backward(const std::vector<float>& grad_output) {
        size_t batch_size = grad_output.size() / out_dim;
        
        // Apply dropout gradient
        std::vector<float> grad_a = grad_output;
        if (dropout_rate > 0.0f && !dropout_mask.empty()) {
            for (size_t i = 0; i < grad_a.size(); ++i) {
                if (!dropout_mask[i]) {
                    grad_a[i] = 0.0f;
                } else {
                    grad_a[i] /= (1.0f - dropout_rate);
                }
            }
        }
        
        // Gradient through ReLU
        std::vector<float> grad_z = grad_a;
        if (use_relu) {
            for (size_t i = 0; i < grad_z.size(); ++i) {
                if (z_cache[i] <= 0) {
                    grad_z[i] = 0.0f;
                }
            }
        }
        
        // Compute weight gradients
        std::fill(dW.begin(), dW.end(), 0.0f);
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < in_dim; ++i) {
                for (size_t j = 0; j < out_dim; ++j) {
                    dW[i * out_dim + j] += x_cache[b * in_dim + i] * 
                                           grad_z[b * out_dim + j] / batch_size;
                }
            }
        }
        
        // Compute bias gradients
        std::fill(db.begin(), db.end(), 0.0f);
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t j = 0; j < out_dim; ++j) {
                db[j] += grad_z[b * out_dim + j] / batch_size;
            }
        }
        
        // Compute input gradients
        std::vector<float> grad_x(batch_size * in_dim, 0.0f);
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < in_dim; ++i) {
                for (size_t j = 0; j < out_dim; ++j) {
                    grad_x[b * in_dim + i] += grad_z[b * out_dim + j] * 
                                              W[i * out_dim + j];
                }
            }
        }
        
        return grad_x;
    }
    
    void update_adam(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
        adam_t++;
        
        // Bias correction
        float lr_t = lr * std::sqrt(1.0f - std::pow(beta2, adam_t)) / 
                     (1.0f - std::pow(beta1, adam_t));
        
        // Update weights
        for (size_t i = 0; i < W.size(); ++i) {
            mW[i] = beta1 * mW[i] + (1 - beta1) * dW[i];
            vW[i] = beta2 * vW[i] + (1 - beta2) * dW[i] * dW[i];
            W[i] -= lr_t * mW[i] / (std::sqrt(vW[i]) + eps);
        }
        
        // Update biases
        for (size_t i = 0; i < b.size(); ++i) {
            mb[i] = beta1 * mb[i] + (1 - beta1) * db[i];
            vb[i] = beta2 * vb[i] + (1 - beta2) * db[i] * db[i];
            b[i] -= lr_t * mb[i] / (std::sqrt(vb[i]) + eps);
        }
    }
};

// ============================================================================
// Idle-specialized RoCA model
// ============================================================================
class IdleRoCAModel {
private:
    IdleRoCAConfig config;
    IdleFeatureProcessor processor;
    
    // Network layers - smaller architecture for idle
    StableDenseLayer encoder1;
    StableDenseLayer encoder2;
    StableDenseLayer encoder3;
    StableDenseLayer decoder;
    StableDenseLayer projector;
    
    // One-class center
    std::vector<float> Ce;
    
public:

// Add these public methods to IdleRoCAModel class:
    // Model persistence methods
    bool save_model(const std::string& filepath) {
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
            Ce
        );
    }
    
    bool load_model(const std::string& filepath) {
        ModelMetadata metadata;
        std::vector<float> feature_mean, feature_std;
        std::vector<bool> is_constant;
        std::vector<size_t> valid_indices;
        std::vector<std::vector<float>> encoder_weights, encoder_biases;
        std::vector<std::vector<float>> decoder_weights, decoder_biases;
        std::vector<float> projector_weights, projector_bias;
        std::vector<float> loaded_Ce;
        
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
                loaded_Ce)) {
            return false;
        }
        
        // Update config
        config.T = metadata.T;
        config.D = metadata.D;
        config.C = metadata.C;
        config.K = metadata.K;
        anomaly_threshold = metadata.anomaly_threshold;
        
        // Recreate layers with correct dimensions
        encoder1 = StableDenseLayer(config.T * config.D, 128, true, 0.1f);
        encoder2 = StableDenseLayer(128, 64, true, 0.1f);
        encoder3 = StableDenseLayer(64, config.C, false, 0.0f);
        decoder = StableDenseLayer(config.C, config.T * config.D, false, 0.0f);
        projector = StableDenseLayer(config.C, config.K, false, 0.0f);
        
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
        Ce = loaded_Ce;
        
        // Update processor
        processor.set_stats(feature_mean, feature_std, is_constant, valid_indices);
        
        return true;
    }
    
    const IdleRoCAConfig& get_config() const { return config; }
    const std::vector<size_t>& get_valid_indices() const { 
        return processor.get_valid_indices(); 
    }
    float anomaly_threshold = 0.0f;
    
    IdleRoCAModel(const IdleRoCAConfig& cfg) 
        : config(cfg),
          encoder1(cfg.T * cfg.D, 128, true, 0.1f),    // Slight dropout
          encoder2(128, 64, true, 0.1f),
          encoder3(64, cfg.C, false, 0.0f),            // No ReLU on latent
          decoder(cfg.C, cfg.T * cfg.D, false, 0.0f),
          projector(cfg.C, cfg.K, false, 0.0f) {
        
        // Initialize center
        Ce.resize(cfg.K);
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        float norm = 0.0f;
        for (auto& c : Ce) {
            c = dist(rng);
            norm += c * c;
        }
        norm = std::sqrt(norm);
        for (auto& c : Ce) c /= norm;
    }
    
    void init_weights(std::mt19937& rng) {
        encoder1.init_weights(rng);
        encoder2.init_weights(rng);
        encoder3.init_weights(rng);
        decoder.init_weights(rng);
        projector.init_weights(rng);
    }
    
    struct ForwardResult {
        std::vector<float> z;
        std::vector<float> x_hat;
        std::vector<float> z_prime;
        std::vector<float> q;
        std::vector<float> q_prime;
        float rec_loss;
        float inv_loss;
        float var_loss;
    };
    
    ForwardResult forward(const std::vector<float>& x, bool training = true) {
        ForwardResult result;
        
        // Encode
        auto h1 = encoder1.forward(x, training);
        auto h2 = encoder2.forward(h1, training);
        result.z = encoder3.forward(h2, training);
        
        // Decode
        result.x_hat = decoder.forward(result.z, training);
        
        // Re-encode (no dropout for consistency)
        auto h1_prime = encoder1.forward(result.x_hat, false);
        auto h2_prime = encoder2.forward(h1_prime, false);
        result.z_prime = encoder3.forward(h2_prime, false);
        
        // Project and normalize
        result.q = projector.forward(result.z, false);
        result.q_prime = projector.forward(result.z_prime, false);
        
        // L2 normalize projections
        l2_normalize_vector(result.q);
        l2_normalize_vector(result.q_prime);
        
        return result;
    }
    
    void compute_losses(const std::vector<float>& x, ForwardResult& fwd) {
        size_t batch_size = x.size() / (config.T * config.D);
        
        // Reconstruction loss with emphasis on variable features
        fwd.rec_loss = 0.0f;
        const auto& constant_mask = processor.get_constant_mask();
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < config.T * config.D; ++i) {
                size_t idx = b * config.T * config.D + i;
                float diff = x[idx] - fwd.x_hat[idx];
                
                // Weight loss based on whether feature is constant
                size_t feature_idx = i % config.D;
                float weight = constant_mask[feature_idx] ? 0.1f : 1.0f;
                
                fwd.rec_loss += weight * diff * diff;
            }
        }
        fwd.rec_loss /= (batch_size * config.T * config.D);
        
        // Invariance loss
        fwd.inv_loss = 0.0f;
        for (size_t b = 0; b < batch_size; ++b) {
            float cos_q = 0.0f, cos_q_prime = 0.0f;
            for (size_t k = 0; k < config.K; ++k) {
                cos_q += fwd.q[b * config.K + k] * Ce[k];
                cos_q_prime += fwd.q_prime[b * config.K + k] * Ce[k];
            }
            fwd.inv_loss += (2.0f - cos_q - cos_q_prime);
        }
        fwd.inv_loss /= batch_size;
        
        // Variance loss
        fwd.var_loss = 0.0f;
        for (size_t k = 0; k < config.K; ++k) {
            float mean = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                mean += fwd.q[b * config.K + k];
            }
            mean /= batch_size;
            
            float var = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                float diff = fwd.q[b * config.K + k] - mean;
                var += diff * diff;
            }
            var /= batch_size;
            
            fwd.var_loss += std::max(0.0f, config.zeta - std::sqrt(var));
        }
        fwd.var_loss /= config.K;
    }
    
    float train_step(const std::vector<float>& x_batch, float lr, size_t epoch) {
        // Forward pass
        auto fwd = forward(x_batch, true);
        compute_losses(x_batch, fwd);
        
        // Total loss with schedule
        float inv_weight = config.lambda_inv * std::min(1.0f, epoch / 20.0f);  // Ramp up
        float total_loss = config.lambda_rec * fwd.rec_loss + 
                          inv_weight * fwd.inv_loss + 
                          config.lambda_var * fwd.var_loss;
        
        // Backward pass for reconstruction
        size_t batch_size = x_batch.size() / (config.T * config.D);
        std::vector<float> grad_x_hat(x_batch.size());
        
        for (size_t i = 0; i < grad_x_hat.size(); ++i) {
            grad_x_hat[i] = -2.0f * config.lambda_rec * (x_batch[i] - fwd.x_hat[i]) / 
                           (batch_size * config.T * config.D);
        }
        
        // Backprop through decoder
        auto grad_z = decoder.backward(grad_x_hat);
        
        // Backprop through encoder
        auto grad_h2 = encoder3.backward(grad_z);
        auto grad_h1 = encoder2.backward(grad_h2);
        encoder1.backward(grad_h1);
        
        // Update all layers
        encoder1.update_adam(lr);
        encoder2.update_adam(lr);
        encoder3.update_adam(lr);
        decoder.update_adam(lr);
        
        // Update Ce with EMA during warmup
        if (epoch < 10) {
            std::vector<float> mean_q(config.K, 0.0f);
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t k = 0; k < config.K; ++k) {
                    mean_q[k] += (fwd.q[b * config.K + k] + fwd.q_prime[b * config.K + k]) / 2.0f;
                }
            }
            for (auto& m : mean_q) m /= batch_size;
            
            float alpha = 0.95f;
            for (size_t k = 0; k < config.K; ++k) {
                Ce[k] = alpha * Ce[k] + (1 - alpha) * mean_q[k];
            }
            l2_normalize_vector(Ce);
        }
        
        return total_loss;
    }
    
    float score_window(const std::vector<float>& window) {
        auto normalized = processor.normalize(window, config.T);
        auto fwd = forward(normalized, false);
        compute_losses(normalized, fwd);
        
        // For idle model, use combination of reconstruction and invariance
        return fwd.rec_loss + 0.5f * fwd.inv_loss;
    }
    
    void set_processor(const IdleFeatureProcessor& proc) {
        processor = proc;
    }
    
private:
    void l2_normalize_vector(std::vector<float>& vec) {
        size_t batch_size = vec.size() / config.K;
        
        for (size_t b = 0; b < batch_size; ++b) {
            float norm = 0.0f;
            for (size_t k = 0; k < config.K; ++k) {
                float val = vec[b * config.K + k];
                norm += val * val;
            }
            norm = std::sqrt(norm + 1e-8f);
            
            for (size_t k = 0; k < config.K; ++k) {
                vec[b * config.K + k] /= norm;
            }
        }
    }
};

// ============================================================================
// Training function for idle baseline
// ============================================================================
void train_idle_model(IdleRoCAModel& model,
                     std::vector<std::vector<float>>& windows,
                     const IdleRoCAConfig& config) {
    
    std::cout << "\n=== Training Idle Baseline Model ===\n";
    std::cout << "Training on " << windows.size() << " windows of idle behavior\n";
    
    // Compute feature statistics
    IdleFeatureProcessor processor;
    processor.compute_stats(windows, config.T, config.D);
    model.set_processor(processor);
    
    // Split train/val
    std::mt19937 rng(42);
    std::vector<size_t> indices(windows.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    size_t val_size = static_cast<size_t>(windows.size() * config.val_split);
    size_t train_size = windows.size() - val_size;
    
    std::vector<size_t> train_indices(indices.begin(), indices.begin() + train_size);
    std::vector<size_t> val_indices(indices.begin() + train_size, indices.end());
    
    std::cout << "  Train: " << train_indices.size() << " windows\n";
    std::cout << "  Val: " << val_indices.size() << " windows\n";
    
    // Initialize model
    model.init_weights(rng);
    
    // Training loop
    size_t batches_per_epoch = (train_indices.size() + config.batch_size - 1) / config.batch_size;
    float best_val_loss = std::numeric_limits<float>::max();
    
    for (size_t epoch = 0; epoch < config.epochs; ++epoch) {
        std::shuffle(train_indices.begin(), train_indices.end(), rng);
        
        float epoch_loss = 0.0f;
        
        // Learning rate schedule
        float lr = config.lr;
        if (epoch > 50) lr *= 0.5f;
        if (epoch > 75) lr *= 0.5f;
        
        for (size_t batch_idx = 0; batch_idx < batches_per_epoch; ++batch_idx) {
            // Create batch
            std::vector<float> batch;
            size_t batch_start = batch_idx * config.batch_size;
            size_t batch_end = std::min(batch_start + config.batch_size, train_indices.size());
            
            for (size_t i = batch_start; i < batch_end; ++i) {
                auto normalized = processor.normalize(windows[train_indices[i]], config.T);
                batch.insert(batch.end(), normalized.begin(), normalized.end());
            }
            
            // Pad batch if necessary
            while (batch.size() < config.batch_size * config.T * config.D) {
                auto normalized = processor.normalize(windows[train_indices[0]], config.T);
                batch.insert(batch.end(), normalized.begin(), normalized.end());
            }
            
            float loss = model.train_step(batch, lr, epoch);
            epoch_loss += loss;
        }
        
        epoch_loss /= batches_per_epoch;
        
        // Validation
        if ((epoch + 1) % 10 == 0) {
            float val_loss = 0.0f;
            for (size_t i = 0; i < val_indices.size(); ++i) {
                float score = model.score_window(windows[val_indices[i]]);
                val_loss += score;
            }
            val_loss /= val_indices.size();
            
            std::cout << "Epoch " << std::setw(3) << (epoch + 1) << "/" << config.epochs;
            std::cout << " | Train Loss: " << std::fixed << std::setprecision(6) << epoch_loss;
            std::cout << " | Val Loss: " << val_loss;
            
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                std::cout << " *";
            }
            std::cout << "\n";
        }
    }
    
    // Compute threshold on validation set
    std::vector<float> val_scores;
    for (size_t idx : val_indices) {
        val_scores.push_back(model.score_window(windows[idx]));
    }
    
    std::sort(val_scores.begin(), val_scores.end());
    
    // Use mean + k*std for threshold
    float mean_score = std::accumulate(val_scores.begin(), val_scores.end(), 0.0f) / val_scores.size();
    float std_score = 0.0f;
    for (float s : val_scores) {
        float diff = s - mean_score;
        std_score += diff * diff;
    }
    std_score = std::sqrt(std_score / val_scores.size());
    
    model.anomaly_threshold = mean_score + config.anomaly_multiplier * std_score;
    
    std::cout << "\n=== Training Complete ===\n";
    std::cout << "Validation score statistics:\n";
    std::cout << "  Mean: " << mean_score << "\n";
    std::cout << "  Std: " << std_score << "\n";
    std::cout << "  Min: " << val_scores.front() << "\n";
    std::cout << "  Max: " << val_scores.back() << "\n";
    std::cout << "  Anomaly threshold: " << model.anomaly_threshold << "\n";
    std::cout << "\nModel is ready to detect any movement as anomalous!\n";
}

} // namespace roca_idle