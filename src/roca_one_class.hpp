// ============================================================================
// roca_one_class.hpp - Self-Supervised One-Class Anomaly Detection
// 
// RoCA: Reconstruction-based One-Class Anomaly detection
// Trains ONLY on normal data using self-supervised learning objectives
// ============================================================================

#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <iomanip>

namespace roca {

// ============================================================================
// Configuration for One-Class RoCA Model
// ============================================================================
struct OneClassConfig {
    // Model architecture
    size_t T = 10;          // Window size (temporal dimension)
    size_t D = 256;         // Feature dimension  
    size_t C = 32;          // Latent representation dimension
    size_t K = 16;          // Projection dimension for one-class learning
    
    // Self-supervised learning hyperparameters
    float lr = 5e-4f;                   // Learning rate
    float lambda_rec = 1.0f;            // Reconstruction loss weight (self-supervised)
    float lambda_oc = 0.5f;             // One-class loss weight 
    float lambda_aug = 0.3f;            // Augmentation consistency loss weight
    float lambda_var = 0.05f;           // Variance regularization (prevent collapse)
    float zeta = 0.5f;                  // Target variance for regularization
    
    // One-class specific parameters
    float center_momentum = 0.95f;      // EMA momentum for center update
    size_t center_warmup_epochs = 10;   // Epochs to warm up center
    float outlier_percentile = 95.0f;   // Percentile for threshold calibration
    
    // Data augmentation for self-supervised learning
    float noise_level = 0.01f;          // Gaussian noise for augmentation
    float dropout_rate = 0.1f;          // Dropout for augmentation
    
    // Training parameters
    size_t batch_size = 32;
    size_t epochs = 100;
    float val_split = 0.2f;
    
    // Anomaly detection
    float anomaly_multiplier = 3.0f;    // Sigma multiplier for threshold
    bool use_adaptive_threshold = true; // Adapt threshold based on validation
};

// ============================================================================
// Self-Supervised Data Augmentation for One-Class Learning
// ============================================================================
class SelfSupervisedAugmentor {
private:
    std::mt19937 rng;
    float noise_level;
    float dropout_rate;
    
public:
    SelfSupervisedAugmentor(float noise = 0.01f, float dropout = 0.1f, unsigned seed = 42)
        : rng(seed), noise_level(noise), dropout_rate(dropout) {}
    
    // Generate augmented view for self-supervised learning
    std::vector<float> augment(const std::vector<float>& x, size_t T, size_t D) {
        std::vector<float> augmented = x;
        
        // Add Gaussian noise (preserves normal patterns while creating variation)
        std::normal_distribution<float> noise_dist(0.0f, noise_level);
        for (auto& val : augmented) {
            val += noise_dist(rng);
        }
        
        // Random feature dropout (forces model to learn redundant representations)
        std::uniform_real_distribution<float> dropout_dist(0.0f, 1.0f);
        for (size_t t = 0; t < T; ++t) {
            for (size_t d = 0; d < D; ++d) {
                if (dropout_dist(rng) < dropout_rate) {
                    augmented[t * D + d] = 0.0f;
                }
            }
        }
        
        return augmented;
    }
    
    // Temporal jittering for time-series augmentation
    std::vector<float> temporal_jitter(const std::vector<float>& x, size_t T, size_t D) {
        std::vector<float> jittered = x;
        std::uniform_int_distribution<int> shift_dist(-1, 1);
        
        for (size_t d = 0; d < D; ++d) {
            int shift = shift_dist(rng);
            if (shift == 0) continue;
            
            for (size_t t = 0; t < T; ++t) {
                int src_t = std::max(0, std::min((int)T - 1, (int)t + shift));
                jittered[t * D + d] = x[src_t * D + d];
            }
        }
        
        return jittered;
    }
};

// ============================================================================
// One-Class Feature Processor (for normal data preprocessing)
// ============================================================================
class OneClassFeatureProcessor {
private:
    std::vector<float> normal_mean;      // Mean of NORMAL training data
    std::vector<float> normal_std;       // Std of NORMAL training data
    std::vector<float> robust_scale;     // Robust scaling factors
    std::vector<bool> is_constant;       // Constant features in normal data
    std::vector<size_t> valid_indices;   // Valid feature indices
    size_t D;
    
public:
    void fit_normal_data(const std::vector<std::vector<float>>& normal_windows, 
                        size_t T, size_t feature_dim) {
        D = feature_dim;
        normal_mean.resize(D, 0.0f);
        normal_std.resize(D, 0.0f);
        robust_scale.resize(D, 1.0f);
        is_constant.resize(D, false);
        valid_indices.clear();
        
        std::cout << "\n=== Learning Normal Data Distribution ===\n";
        std::cout << "Fitting on " << normal_windows.size() << " normal samples\n";
        
        // Compute statistics from NORMAL data only
        std::vector<std::vector<float>> feature_values(D);
        
        for (const auto& window : normal_windows) {
            for (size_t t = 0; t < T; ++t) {
                for (size_t d = 0; d < D; ++d) {
                    float val = window[t * D + d];
                    if (!std::isnan(val) && !std::isinf(val)) {
                        feature_values[d].push_back(val);
                    }
                }
            }
        }
        
        // Compute robust statistics for each feature
        size_t constant_count = 0;
        for (size_t d = 0; d < D; ++d) {
            if (feature_values[d].empty()) continue;
            
            // Sort for percentile-based statistics
            std::sort(feature_values[d].begin(), feature_values[d].end());
            
            // Use median instead of mean for robustness
            size_t mid = feature_values[d].size() / 2;
            float median = feature_values[d][mid];
            
            // Use IQR for robust scale
            size_t q1_idx = feature_values[d].size() / 4;
            size_t q3_idx = 3 * feature_values[d].size() / 4;
            float iqr = feature_values[d][q3_idx] - feature_values[d][q1_idx];
            
            normal_mean[d] = median;
            
            // Check if constant
            if (iqr < 1e-6f) {
                is_constant[d] = true;
                constant_count++;
                robust_scale[d] = 1.0f;  // Avoid division by zero
                normal_std[d] = 0.001f;   // Small noise floor
            } else {
                normal_std[d] = iqr / 1.349f;  // Convert IQR to std equivalent
                robust_scale[d] = normal_std[d];
                valid_indices.push_back(d);
            }
        }
        
        std::cout << "Normal data characteristics:\n";
        std::cout << "  Constant features: " << constant_count << "/" << D << "\n";
        std::cout << "  Variable features: " << valid_indices.size() << "\n";
        std::cout << "  These statistics define the NORMAL behavior\n";
    }
    
    // Transform data relative to learned normal distribution
    std::vector<float> transform(const std::vector<float>& window, size_t T) const {
        std::vector<float> transformed = window;
        
        for (size_t t = 0; t < T; ++t) {
            for (size_t d = 0; d < D; ++d) {
                size_t idx = t * D + d;
                
                if (std::isnan(transformed[idx]) || std::isinf(transformed[idx])) {
                    transformed[idx] = 0.0f;
                    continue;
                }
                
                // Center relative to normal data and scale
                transformed[idx] = (transformed[idx] - normal_mean[d]) / robust_scale[d];
                
                // Clip extreme values
                transformed[idx] = std::max(-5.0f, std::min(5.0f, transformed[idx]));
            }
        }
        
        return transformed;
    }
    
    // Getters for model persistence
    const std::vector<float>& get_mean() const { return normal_mean; }
    const std::vector<float>& get_std() const { return normal_std; }
    const std::vector<bool>& get_constant_mask() const { return is_constant; }
    const std::vector<size_t>& get_valid_indices() const { return valid_indices; }
    
    void set_stats(const std::vector<float>& mean, 
                   const std::vector<float>& std,
                   const std::vector<bool>& constant,
                   const std::vector<size_t>& valid_idx) {
        normal_mean = mean;
        normal_std = std;
        robust_scale = std;  // Use std as scale
        is_constant = constant;
        valid_indices = valid_idx;
        D = mean.size();
    }
};

// ============================================================================
// Dense Layer (unchanged but included for completeness)
// ============================================================================
class DenseLayer {
public:
    std::vector<float> W, b, dW, db;
    std::vector<float> mW, vW, mb, vb;  // Adam optimizer state
    int adam_t = 0;
    size_t in_dim, out_dim;
    bool use_relu;
    
    // Cache for forward/backward
    std::vector<float> x_cache, z_cache, a_cache;
    
    DenseLayer(size_t in, size_t out, bool relu = true) 
        : in_dim(in), out_dim(out), use_relu(relu) {
        size_t w_size = in * out;
        W.resize(w_size);
        b.resize(out);
        dW.resize(w_size);
        db.resize(out);
        mW.resize(w_size, 0.0f);
        vW.resize(w_size, 0.0f);
        mb.resize(out, 0.0f);
        vb.resize(out, 0.0f);
    }
    
    void init_weights(std::mt19937& rng) {
        float scale = std::sqrt(2.0f / in_dim);
        std::normal_distribution<float> dist(0.0f, scale);
        for (auto& w : W) w = dist(rng);
        std::fill(b.begin(), b.end(), 0.01f);
    }
    
    std::vector<float> forward(const std::vector<float>& x) {
        size_t batch_size = x.size() / in_dim;
        x_cache = x;
        z_cache.resize(batch_size * out_dim);
        std::fill(z_cache.begin(), z_cache.end(), 0.0f);
        
        // z = x @ W + b
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
            for (auto& a : a_cache) a = std::max(0.0f, a);
        }
        
        return a_cache;
    }
    
    std::vector<float> backward(const std::vector<float>& grad_output) {
        size_t batch_size = grad_output.size() / out_dim;
        std::vector<float> grad_z = grad_output;
        
        // Gradient through ReLU
        if (use_relu) {
            for (size_t i = 0; i < grad_z.size(); ++i) {
                if (z_cache[i] <= 0) grad_z[i] = 0.0f;
            }
        }
        
        // Compute gradients
        std::fill(dW.begin(), dW.end(), 0.0f);
        std::fill(db.begin(), db.end(), 0.0f);
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < in_dim; ++i) {
                for (size_t j = 0; j < out_dim; ++j) {
                    dW[i * out_dim + j] += x_cache[b * in_dim + i] * 
                                           grad_z[b * out_dim + j] / batch_size;
                }
            }
            for (size_t j = 0; j < out_dim; ++j) {
                db[j] += grad_z[b * out_dim + j] / batch_size;
            }
        }
        
        // Input gradients
        std::vector<float> grad_x(batch_size * in_dim, 0.0f);
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < in_dim; ++i) {
                for (size_t j = 0; j < out_dim; ++j) {
                    grad_x[b * in_dim + i] += grad_z[b * out_dim + j] * W[i * out_dim + j];
                }
            }
        }
        
        return grad_x;
    }
    
    void update_adam(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
        adam_t++;
        float lr_t = lr * std::sqrt(1.0f - std::pow(beta2, adam_t)) / 
                     (1.0f - std::pow(beta1, adam_t));
        
        for (size_t i = 0; i < W.size(); ++i) {
            mW[i] = beta1 * mW[i] + (1 - beta1) * dW[i];
            vW[i] = beta2 * vW[i] + (1 - beta2) * dW[i] * dW[i];
            W[i] -= lr_t * mW[i] / (std::sqrt(vW[i]) + eps);
        }
        
        for (size_t i = 0; i < b.size(); ++i) {
            mb[i] = beta1 * mb[i] + (1 - beta1) * db[i];
            vb[i] = beta2 * vb[i] + (1 - beta2) * db[i] * db[i];
            b[i] -= lr_t * mb[i] / (std::sqrt(vb[i]) + eps);
        }
    }
};

// ============================================================================
// Self-Supervised One-Class RoCA Model
// ============================================================================
class OneClassRoCA {
private:
    OneClassConfig config;
    OneClassFeatureProcessor processor;
    SelfSupervisedAugmentor augmentor;
    
    // Encoder-Decoder Architecture
    DenseLayer encoder1;
    DenseLayer encoder2;
    DenseLayer encoder3;
    DenseLayer decoder;
    DenseLayer projector;
    
    // One-class center (learned from normal data only)
    std::vector<float> center;
    
    // Training statistics
    std::vector<float> training_scores;
    
public:
    float anomaly_threshold = 0.0f;
    
    OneClassRoCA(const OneClassConfig& cfg) 
        : config(cfg),
          augmentor(cfg.noise_level, cfg.dropout_rate),
          encoder1(cfg.T * cfg.D, 128, true),
          encoder2(128, 64, true),
          encoder3(64, cfg.C, false),
          decoder(cfg.C, cfg.T * cfg.D, false),
          projector(cfg.C, cfg.K, false) {
        
        // Initialize one-class center randomly (will be updated during training)
        center.resize(cfg.K);
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& c : center) c = dist(rng);
        l2_normalize(center);
        
        std::cout << "Initialized One-Class RoCA Model\n";
        std::cout << "  Architecture: " << cfg.T << "×" << cfg.D << " → " 
                  << cfg.C << " → " << cfg.K << " (one-class space)\n";
    }
    
    void init_weights(std::mt19937& rng) {
        encoder1.init_weights(rng);
        encoder2.init_weights(rng);
        encoder3.init_weights(rng);
        decoder.init_weights(rng);
        projector.init_weights(rng);
    }
    
    struct ForwardResult {
        std::vector<float> z;           // Latent representation
        std::vector<float> x_hat;       // Reconstruction
        std::vector<float> z_aug;       // Augmented latent
        std::vector<float> q;           // One-class projection
        std::vector<float> q_aug;       // Augmented projection
        float rec_loss = 0.0f;          // Reconstruction loss (self-supervised)
        float oc_loss = 0.0f;           // One-class loss
        float aug_loss = 0.0f;          // Augmentation consistency loss
        float var_loss = 0.0f;          // Variance regularization
    };
    
    ForwardResult forward(const std::vector<float>& x_normal, bool training = true) {
        ForwardResult result;
        size_t batch_size = x_normal.size() / (config.T * config.D);
        
        // === Self-Supervised Learning Path ===
        
        // 1. Encode normal data
        auto h1 = encoder1.forward(x_normal);
        auto h2 = encoder2.forward(h1);
        result.z = encoder3.forward(h2);
        
        // 2. Decode (reconstruction task - self-supervised)
        result.x_hat = decoder.forward(result.z);
        
        // 3. Project to one-class space
        result.q = projector.forward(result.z);
        l2_normalize_batch(result.q, config.K);
        
        // 4. If training, create augmented view for consistency
        if (training) {
            std::vector<float> x_aug;
            for (size_t b = 0; b < batch_size; ++b) {
                std::vector<float> sample(x_normal.begin() + b * config.T * config.D,
                                         x_normal.begin() + (b + 1) * config.T * config.D);
                auto aug_sample = augmentor.augment(sample, config.T, config.D);
                x_aug.insert(x_aug.end(), aug_sample.begin(), aug_sample.end());
            }
            
            // Encode augmented view
            auto h1_aug = encoder1.forward(x_aug);
            auto h2_aug = encoder2.forward(h1_aug);
            result.z_aug = encoder3.forward(h2_aug);
            result.q_aug = projector.forward(result.z_aug);
            l2_normalize_batch(result.q_aug, config.K);
        }
        
        return result;
    }
    
    void compute_losses(const std::vector<float>& x_normal, ForwardResult& fwd) {
        size_t batch_size = x_normal.size() / (config.T * config.D);
        
        // 1. Reconstruction Loss (Self-Supervised)
        fwd.rec_loss = 0.0f;
        for (size_t i = 0; i < x_normal.size(); ++i) {
            float diff = x_normal[i] - fwd.x_hat[i];
            fwd.rec_loss += diff * diff;
        }
        fwd.rec_loss /= x_normal.size();
        
        // 2. One-Class Loss (pulls normal data to center)
        fwd.oc_loss = 0.0f;
        for (size_t b = 0; b < batch_size; ++b) {
            float dist_sq = 0.0f;
            for (size_t k = 0; k < config.K; ++k) {
                float diff = fwd.q[b * config.K + k] - center[k];
                dist_sq += diff * diff;
            }
            fwd.oc_loss += dist_sq;
        }
        fwd.oc_loss /= batch_size;
        
        // 3. Augmentation Consistency Loss (Self-Supervised)
        if (!fwd.q_aug.empty()) {
            fwd.aug_loss = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t k = 0; k < config.K; ++k) {
                    float diff = fwd.q[b * config.K + k] - fwd.q_aug[b * config.K + k];
                    fwd.aug_loss += diff * diff;
                }
            }
            fwd.aug_loss /= (batch_size * config.K);
        }
        
        // 4. Variance Regularization (prevent collapse)
        fwd.var_loss = 0.0f;
        for (size_t k = 0; k < config.K; ++k) {
            float mean = 0.0f, var = 0.0f;
            
            for (size_t b = 0; b < batch_size; ++b) {
                mean += fwd.q[b * config.K + k];
            }
            mean /= batch_size;
            
            for (size_t b = 0; b < batch_size; ++b) {
                float diff = fwd.q[b * config.K + k] - mean;
                var += diff * diff;
            }
            var /= batch_size;
            
            fwd.var_loss += std::max(0.0f, config.zeta - std::sqrt(var + 1e-8f));
        }
        fwd.var_loss /= config.K;
    }
    
    float train_step(const std::vector<float>& x_batch_normal, float lr, size_t epoch) {
        // Forward pass with augmentation
        auto fwd = forward(x_batch_normal, true);
        compute_losses(x_batch_normal, fwd);
        
        // Total loss (all self-supervised except one-class term)
        float total_loss = config.lambda_rec * fwd.rec_loss +     // Self-supervised
                          config.lambda_oc * fwd.oc_loss +         // One-class
                          config.lambda_aug * fwd.aug_loss +       // Self-supervised
                          config.lambda_var * fwd.var_loss;        // Regularization
        
        // Backpropagation
        size_t batch_size = x_batch_normal.size() / (config.T * config.D);
        
        // Gradient for reconstruction
        std::vector<float> grad_x_hat(x_batch_normal.size());
        for (size_t i = 0; i < grad_x_hat.size(); ++i) {
            grad_x_hat[i] = -2.0f * config.lambda_rec * 
                           (x_batch_normal[i] - fwd.x_hat[i]) / x_batch_normal.size();
        }
        
        // Backprop through decoder
        auto grad_z = decoder.backward(grad_x_hat);
        
        // Add gradient from one-class loss
        auto grad_q = projector.backward(grad_z);
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t k = 0; k < config.K; ++k) {
                size_t idx = b * config.K + k;
                grad_q[idx] += 2.0f * config.lambda_oc * 
                               (fwd.q[idx] - center[k]) / batch_size;
            }
        }
        
        // Backprop through encoder
        auto grad_h2 = encoder3.backward(grad_z);
        auto grad_h1 = encoder2.backward(grad_h2);
        encoder1.backward(grad_h1);
        
        // Update all layers
        encoder1.update_adam(lr);
        encoder2.update_adam(lr);
        encoder3.update_adam(lr);
        decoder.update_adam(lr);
        projector.update_adam(lr);
        
        // Update one-class center (EMA during warmup)
        if (epoch < config.center_warmup_epochs) {
            std::vector<float> batch_center(config.K, 0.0f);
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t k = 0; k < config.K; ++k) {
                    batch_center[k] += fwd.q[b * config.K + k];
                }
            }
            for (auto& c : batch_center) c /= batch_size;
            
            // Exponential moving average
            for (size_t k = 0; k < config.K; ++k) {
                center[k] = config.center_momentum * center[k] + 
                           (1 - config.center_momentum) * batch_center[k];
            }
            l2_normalize(center);
        }
        
        return total_loss;
    }
    
    // Score for anomaly detection (distance from normal)
    float anomaly_score(const std::vector<float>& window) {
        auto normalized = processor.transform(window, config.T);
        auto fwd = forward(normalized, false);
        compute_losses(normalized, fwd);
        
        // Combined anomaly score
        float rec_error = fwd.rec_loss;
        float oc_distance = fwd.oc_loss;
        
        // Weight reconstruction more for detecting movement
        return rec_error + 0.5f * oc_distance;
    }
    
    void set_processor(const OneClassFeatureProcessor& proc) {
        processor = proc;
    }
    
    void calibrate_threshold(const std::vector<float>& validation_scores) {
        if (validation_scores.empty()) return;
        
        std::vector<float> sorted_scores = validation_scores;
        std::sort(sorted_scores.begin(), sorted_scores.end());
        
        if (config.use_adaptive_threshold) {
            // Use percentile-based threshold
            size_t idx = static_cast<size_t>(sorted_scores.size() * 
                                            config.outlier_percentile / 100.0f);
            anomaly_threshold = sorted_scores[std::min(idx, sorted_scores.size() - 1)];
        } else {
            // Use mean + k*std
            float mean = std::accumulate(sorted_scores.begin(), sorted_scores.end(), 0.0f) 
                        / sorted_scores.size();
            float std = 0.0f;
            for (float s : sorted_scores) {
                float diff = s - mean;
                std += diff * diff;
            }
            std = std::sqrt(std / sorted_scores.size());
            anomaly_threshold = mean + config.anomaly_multiplier * std;
        }
        
        std::cout << "\nCalibrated anomaly threshold: " << anomaly_threshold << "\n";
        std::cout << "  Based on " << sorted_scores.size() << " normal validation samples\n";
        std::cout << "  Min normal score: " << sorted_scores.front() << "\n";
        std::cout << "  Max normal score: " << sorted_scores.back() << "\n";
    }
    
    // Model persistence
    bool save_model(const std::string& filepath);
    bool load_model(const std::string& filepath);
    
    const OneClassConfig& get_config() const { return config; }
    const std::vector<size_t>& get_valid_indices() const { 
        return processor.get_valid_indices(); 
    }
    
private:
    void l2_normalize(std::vector<float>& vec) {
        float norm = 0.0f;
        for (float v : vec) norm += v * v;
        norm = std::sqrt(norm + 1e-8f);
        for (auto& v : vec) v /= norm;
    }
    
    void l2_normalize_batch(std::vector<float>& batch, size_t dim) {
        size_t batch_size = batch.size() / dim;
        for (size_t b = 0; b < batch_size; ++b) {
            float norm = 0.0f;
            for (size_t d = 0; d < dim; ++d) {
                float val = batch[b * dim + d];
                norm += val * val;
            }
            norm = std::sqrt(norm + 1e-8f);
            for (size_t d = 0; d < dim; ++d) {
                batch[b * dim + d] /= norm;
            }
        }
    }
};

// ============================================================================
// Training function for One-Class Learning (ONLY NORMAL DATA)
// ============================================================================
void train_one_class_model(OneClassRoCA& model,
                          std::vector<std::vector<float>>& normal_windows,
                          const OneClassConfig& config) {
    
    std::cout << "\n╔════════════════════════════════════════╗\n";
    std::cout << "║   Self-Supervised One-Class Training   ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    std::cout << "Training on " << normal_windows.size() << " NORMAL samples only\n";
    std::cout << "This is ONE-CLASS learning - no anomalies in training data!\n\n";
    
    // Learn normal data distribution
    OneClassFeatureProcessor processor;
    processor.fit_normal_data(normal_windows, config.T, config.D);
    model.set_processor(processor);
    
    // Split normal data into train/val
    std::mt19937 rng(42);
    std::vector<size_t> indices(normal_windows.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    size_t val_size = static_cast<size_t>(normal_windows.size() * config.val_split);
    size_t train_size = normal_windows.size() - val_size;
    
    std::vector<size_t> train_indices(indices.begin(), indices.begin() + train_size);
    std::vector<size_t> val_indices(indices.begin() + train_size, indices.end());
    
    std::cout << "Normal data split:\n";
    std::cout << "  Training: " << train_indices.size() << " normal samples\n";
    std::cout << "  Validation: " << val_indices.size() << " normal samples\n\n";
    
    // Initialize model
    model.init_weights(rng);
    
    // Training loop - SELF-SUPERVISED on normal data
    size_t batches_per_epoch = (train_indices.size() + config.batch_size - 1) / 
                              config.batch_size;
    
    std::cout << "Starting self-supervised training...\n";
    std::cout << "Losses: Rec (reconstruction), OC (one-class), Aug (augmentation), Var (variance)\n\n";
    
    for (size_t epoch = 0; epoch < config.epochs; ++epoch) {
        std::shuffle(train_indices.begin(), train_indices.end(), rng);
        
        float epoch_loss = 0.0f;
        float lr = config.lr;
        
        // Learning rate schedule
        if (epoch > 50) lr *= 0.5f;
        if (epoch > 75) lr *= 0.5f;
        
        for (size_t batch_idx = 0; batch_idx < batches_per_epoch; ++batch_idx) {
            // Create batch of NORMAL data
            std::vector<float> batch;
            size_t batch_start = batch_idx * config.batch_size;
            size_t batch_end = std::min(batch_start + config.batch_size, train_indices.size());
            
            for (size_t i = batch_start; i < batch_end; ++i) {
                auto normalized = processor.transform(normal_windows[train_indices[i]], config.T);
                batch.insert(batch.end(), normalized.begin(), normalized.end());
            }
            
            // Pad batch if necessary
            while (batch.size() < config.batch_size * config.T * config.D) {
                auto normalized = processor.transform(normal_windows[train_indices[0]], config.T);
                batch.insert(batch.end(), normalized.begin(), normalized.end());
            }
            
            float loss = model.train_step(batch, lr, epoch);
            epoch_loss += loss;
        }
        
        epoch_loss /= batches_per_epoch;
        
        // Validation on held-out NORMAL data
        if ((epoch + 1) % 10 == 0) {
            std::vector<float> val_scores;
            for (size_t idx : val_indices) {
                float score = model.anomaly_score(normal_windows[idx]);
                val_scores.push_back(score);
            }
            
            float val_mean = std::accumulate(val_scores.begin(), val_scores.end(), 0.0f) 
                           / val_scores.size();
            
            std::cout << "Epoch " << std::setw(3) << (epoch + 1) << "/" << config.epochs;
            std::cout << " | Train Loss: " << std::fixed << std::setprecision(4) << epoch_loss;
            std::cout << " | Val Score (normal): " << val_mean << "\n";
        }
    }
    
    // Calibrate threshold on validation NORMAL data
    std::cout << "\n=== Calibrating Anomaly Threshold ===\n";
    
    std::vector<float> val_scores;
    for (size_t idx : val_indices) {
        val_scores.push_back(model.anomaly_score(normal_windows[idx]));
    }
    
    model.calibrate_threshold(val_scores);
    
    std::cout << "\n✅ One-Class Training Complete!\n";
    std::cout << "Model has learned the NORMAL behavior distribution.\n";
    std::cout << "Any deviation from this learned distribution will be flagged as anomalous.\n";
}

} // namespace roca

// Include implementation
#include "roca_one_class_impl.hpp"