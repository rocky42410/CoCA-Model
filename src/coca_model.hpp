// ============================================================================
// COCA (Competitive One-Class Anomaly) Model Implementation - FIXED VERSION
// Fixes state accumulation bug that causes monotonically increasing scores
// ============================================================================

#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

namespace coca {

// ============================================================================
// COCA Configuration (unchanged)
// ============================================================================
struct COCAConfig {
    // Model architecture
    size_t T = 10;          // Window size
    size_t D = 256;         // Feature dimension  
    size_t C = 32;          // Latent dimension
    size_t K = 16;          // Projection dimension
    
    // Loss weights
    float lambda_inv = 1.0f;     // Invariance weight
    float lambda_var = 1.0f;     // Variance weight
    float lambda_rec = 0.1f;     // Reconstruction weight (0 for pure COCA)
    
    // Variance parameters
    float zeta = 1.0f;              // Target std for variance loss
    float variance_epsilon = 1e-4f; // Epsilon for numerical stability
    
    // Center parameters
    size_t center_warmup_epochs = 8;  // Epochs to warm up center
    size_t inv_ramp_epochs = 20;      // Epochs to ramp up invariance
    
    // Training parameters
    float learning_rate = 5e-4f;
    size_t batch_size = 32;
    size_t epochs = 50;
    float val_split = 0.2f;
    
    // Score and threshold
    std::string score_mix = "inv_only";  // "inv_only" or "inv_plus_rec"
    std::string threshold_mode = "quantile";  // "zscore" or "quantile"
    float threshold_quantile = 0.995f;
    float threshold_zscore_k = 3.0f;
    
    // Misc
    float min_std = 1e-4f;
    float dropout_rate = 0.1f;
    unsigned seed = 42;
    
    // Early stopping
    size_t early_stop_patience = 10;
    float early_stop_min_delta = 1e-6f;
};

// ============================================================================
// Feature Processor - FIXED to be truly stateless
// ============================================================================
class FeatureProcessor {
private:
    std::vector<float> feature_mean;
    std::vector<float> feature_std;
    std::vector<bool> is_constant;
    std::vector<size_t> constant_indices;
    size_t D;
    float min_std;
    
public:
    void compute_stats(const std::vector<std::vector<float>>& windows, 
                      size_t T, size_t feature_dim, float min_std_val = 1e-4f) {
        D = feature_dim;
        min_std = min_std_val;
        feature_mean.resize(D, 0.0f);
        feature_std.resize(D, 0.0f);
        is_constant.resize(D, false);
        constant_indices.clear();
        
        // First pass: compute mean
        size_t total_samples = windows.size() * T;
        for (const auto& window : windows) {
            for (size_t t = 0; t < T; ++t) {
                for (size_t d = 0; d < D; ++d) {
                    size_t idx = t * D + d;
                    float val = window[idx];
                    if (!std::isnan(val) && !std::isinf(val)) {
                        feature_mean[d] += val;
                    }
                }
            }
        }
        
        for (size_t d = 0; d < D; ++d) {
            feature_mean[d] /= total_samples;
        }
        
        // Second pass: compute std
        for (const auto& window : windows) {
            for (size_t t = 0; t < T; ++t) {
                for (size_t d = 0; d < D; ++d) {
                    size_t idx = t * D + d;
                    float val = window[idx];
                    if (!std::isnan(val) && !std::isinf(val)) {
                        float diff = val - feature_mean[d];
                        feature_std[d] += diff * diff;
                    }
                }
            }
        }
        
        // Compute std with minimum threshold
        for (size_t d = 0; d < D; ++d) {
            feature_std[d] = std::sqrt(feature_std[d] / total_samples);
            
            // Apply minimum std to avoid division by zero
            if (feature_std[d] < min_std) {
                is_constant[d] = true;
                constant_indices.push_back(d);
                feature_std[d] = std::max(feature_std[d], min_std);
            }
        }
        
        std::cout << "\nFeature Statistics:\n";
        std::cout << "  Total features: " << D << "\n";
        std::cout << "  Constant features: " << constant_indices.size() << "\n";
        std::cout << "  Variable features: " << (D - constant_indices.size()) << "\n";
    }
    
    // CRITICAL FIX: Make normalize truly const and stateless
    std::vector<float> normalize(const std::vector<float>& window, size_t T) const {
        std::vector<float> normalized = window;
        
        for (size_t t = 0; t < T; ++t) {
            for (size_t d = 0; d < D; ++d) {
                size_t idx = t * D + d;
                
                // Handle NaN/Inf
                if (std::isnan(normalized[idx]) || std::isinf(normalized[idx])) {
                    normalized[idx] = 0.0f;
                    continue;
                }
                
                // Standardize
                normalized[idx] = (normalized[idx] - feature_mean[d]) / feature_std[d];
                
                // Clip extreme values
                normalized[idx] = std::max(-5.0f, std::min(5.0f, normalized[idx]));
            }
        }
        
        return normalized;
    }
    
    const std::vector<float>& get_mean() const { return feature_mean; }
    const std::vector<float>& get_std() const { return feature_std; }
    const std::vector<bool>& get_constant_mask() const { return is_constant; }
    const std::vector<size_t>& get_constant_indices() const { return constant_indices; }
    
    // Method to restore statistics from saved model
    void set_stats(const std::vector<float>& mean, 
                   const std::vector<float>& std,
                   const std::vector<bool>& const_mask) {
        feature_mean = mean;
        feature_std = std;
        is_constant = const_mask;
        D = mean.size();
        
        // Rebuild constant indices
        constant_indices.clear();
        for (size_t i = 0; i < is_constant.size(); ++i) {
            if (is_constant[i]) {
                constant_indices.push_back(i);
            }
        }
    }
};

// ============================================================================
// Dense Layer - FIXED to properly handle inference vs training
// ============================================================================
class DenseLayer {
public:
    std::vector<float> W, b, dW, db;
    std::vector<float> mW, vW, mb, vb;  // Adam state
    int adam_t = 0;
    
    size_t in_dim, out_dim;
    bool use_relu;
    float dropout_rate;
    
    // Cache for backward pass - ONLY used during training
    std::vector<float> x_cache, z_cache, a_cache;
    std::vector<bool> dropout_mask;
    
    DenseLayer(size_t in, size_t out, bool relu = true, float dropout = 0.0f) 
        : in_dim(in), out_dim(out), use_relu(relu), dropout_rate(dropout) {
        
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
        float scale = std::sqrt(2.0f / in_dim);  // He initialization
        std::normal_distribution<float> dist(0.0f, scale);
        
        for (auto& w : W) {
            w = dist(rng);
        }
        std::fill(b.begin(), b.end(), 0.01f);
    }
    
    // CRITICAL FIX: Clear caches before each forward pass to prevent state accumulation
    std::vector<float> forward(const std::vector<float>& x, bool training, std::mt19937& rng) {
        size_t batch_size = x.size() / in_dim;
        
        // CRITICAL: Only cache during training
        if (training) {
            x_cache = x;
        }
        
        // Linear: z = x @ W + b
        std::vector<float> z(batch_size * out_dim, 0.0f);
        
        for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
            for (size_t i = 0; i < in_dim; ++i) {
                for (size_t j = 0; j < out_dim; ++j) {
                    z[b_idx * out_dim + j] += 
                        x[b_idx * in_dim + i] * W[i * out_dim + j];
                }
            }
            for (size_t j = 0; j < out_dim; ++j) {
                z[b_idx * out_dim + j] += b[j];
            }
        }
        
        // Store z only during training
        if (training) {
            z_cache = z;
        }
        
        // Activation
        std::vector<float> a = z;
        if (use_relu) {
            for (auto& val : a) {
                val = std::max(0.0f, val);
            }
        }
        
        // Store activated values only during training
        if (training) {
            a_cache = a;
        }
        
        // Dropout (only during training)
        if (training && dropout_rate > 0.0f) {
            dropout_mask.resize(a.size());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            
            for (size_t i = 0; i < a.size(); ++i) {
                dropout_mask[i] = (dist(rng) > dropout_rate);
                if (!dropout_mask[i]) {
                    a[i] = 0.0f;
                } else {
                    a[i] /= (1.0f - dropout_rate);
                }
            }
        } else {
            // CRITICAL: Clear dropout mask during inference
            dropout_mask.clear();
        }
        
        return a;
    }
    
    std::vector<float> backward(const std::vector<float>& grad_output) {
        size_t batch_size = grad_output.size() / out_dim;
        
        // Apply dropout gradient
        std::vector<float> grad_a = grad_output;
        if (!dropout_mask.empty()) {
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
        for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
            for (size_t i = 0; i < in_dim; ++i) {
                for (size_t j = 0; j < out_dim; ++j) {
                    dW[i * out_dim + j] += 
                        x_cache[b_idx * in_dim + i] * grad_z[b_idx * out_dim + j];
                }
            }
        }
        for (auto& dw : dW) dw /= batch_size;
        
        // Compute bias gradients
        std::fill(db.begin(), db.end(), 0.0f);
        for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
            for (size_t j = 0; j < out_dim; ++j) {
                db[j] += grad_z[b_idx * out_dim + j];
            }
        }
        for (auto& d : db) d /= batch_size;
        
        // Compute input gradients
        std::vector<float> grad_x(batch_size * in_dim, 0.0f);
        for (size_t b_idx = 0; b_idx < batch_size; ++b_idx) {
            for (size_t i = 0; i < in_dim; ++i) {
                for (size_t j = 0; j < out_dim; ++j) {
                    grad_x[b_idx * in_dim + i] += 
                        grad_z[b_idx * out_dim + j] * W[i * out_dim + j];
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
    
    // CRITICAL NEW METHOD: Clear all caches
    void clear_caches() {
        x_cache.clear();
        z_cache.clear();
        a_cache.clear();
        dropout_mask.clear();
    }
};

// ============================================================================
// COCA Model - FIXED to prevent state accumulation
// ============================================================================
class COCAModel {
public:
    COCAConfig config;
    FeatureProcessor processor;
    
    // Network layers
    std::vector<DenseLayer> encoder_layers;
    std::vector<DenseLayer> decoder_layers;
    DenseLayer projector;
    
    // One-class center - FROZEN after training
    std::vector<float> Ce;
    
    // Anomaly detection
    float anomaly_threshold = 0.0f;
    
    // RNG for reproducibility
    std::mt19937 rng;
    
    // Training metrics
    std::vector<float> train_losses;
    std::vector<float> val_losses;
    
    // Training state flag
    bool is_training = false;
    
    COCAModel(const COCAConfig& cfg) 
        : config(cfg), 
          projector(cfg.C, cfg.K, false, 0.0f),
          rng(cfg.seed) {
        
        // Build encoder
        encoder_layers.emplace_back(cfg.T * cfg.D, 128, true, cfg.dropout_rate);
        encoder_layers.emplace_back(128, 64, true, cfg.dropout_rate);
        encoder_layers.emplace_back(64, cfg.C, false, 0.0f);  // No ReLU on latent
        
        // Build decoder
        decoder_layers.emplace_back(cfg.C, cfg.T * cfg.D, false, 0.0f);
        
        // Initialize center on unit sphere
        Ce.resize(cfg.K);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        float norm = 0.0f;
        for (auto& c : Ce) {
            c = dist(rng);
            norm += c * c;
        }
        norm = std::sqrt(norm);
        for (auto& c : Ce) c /= norm;
    }
    
    void init_weights() {
        for (auto& layer : encoder_layers) {
            layer.init_weights(rng);
        }
        for (auto& layer : decoder_layers) {
            layer.init_weights(rng);
        }
        projector.init_weights(rng);
    }
    
    struct ForwardResult {
        std::vector<float> z;
        std::vector<float> x_hat;
        std::vector<float> z_prime;
        std::vector<float> q;
        std::vector<float> q_prime;
        float rec_loss = 0.0f;
        float inv_loss = 0.0f;
        float var_loss = 0.0f;
    };
    
    // CRITICAL FIX: Clear all layer caches before forward pass
    void clear_all_caches() {
        for (auto& layer : encoder_layers) {
            layer.clear_caches();
        }
        for (auto& layer : decoder_layers) {
            layer.clear_caches();
        }
        projector.clear_caches();
    }
    
    ForwardResult forward(const std::vector<float>& x, bool training = true) {
        ForwardResult result;
        
        // CRITICAL: Clear caches for inference to prevent state accumulation
        if (!training) {
            clear_all_caches();
        }
        
        // Encode: x -> z
        std::vector<float> h = x;
        for (auto& layer : encoder_layers) {
            h = layer.forward(h, training, rng);
        }
        result.z = h;
        
        // Decode: z -> x̂
        h = result.z;
        for (auto& layer : decoder_layers) {
            h = layer.forward(h, training, rng);
        }
        result.x_hat = h;
        
        // Re-encode: x̂ -> z′ (NEVER use dropout for consistency)
        h = result.x_hat;
        for (auto& layer : encoder_layers) {
            h = layer.forward(h, false, rng);  // ALWAYS false for re-encoding
        }
        result.z_prime = h;
        
        // Project and normalize: z -> q, z′ -> q′
        result.q = projector.forward(result.z, false, rng);
        result.q_prime = projector.forward(result.z_prime, false, rng);
        
        // L2 normalize projections
        l2_normalize_batch(result.q, config.K);
        l2_normalize_batch(result.q_prime, config.K);
        
        return result;
    }
    
    void compute_losses(const std::vector<float>& x, ForwardResult& fwd) {
        size_t batch_size = x.size() / (config.T * config.D);
        
        // Reconstruction loss with optional weighting
        fwd.rec_loss = 0.0f;
        const auto& constant_mask = processor.get_constant_mask();
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < config.T * config.D; ++i) {
                size_t idx = b * config.T * config.D + i;
                float diff = x[idx] - fwd.x_hat[idx];
                
                // Weight by feature importance (only if we have mask info)
                float weight = 1.0f;
                if (!constant_mask.empty() && (i % config.D) < constant_mask.size()) {
                    size_t feature_idx = i % config.D;
                    weight = constant_mask[feature_idx] ? 0.1f : 1.0f;
                }
                
                fwd.rec_loss += weight * diff * diff;
            }
        }
        fwd.rec_loss /= (batch_size * config.T * config.D);
        
        // Invariance loss: 2 - cos(q, Ce) - cos(q′, Ce)
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
        
        // Variance loss with stability fix
        fwd.var_loss = 0.0f;
        
        // CRITICAL FIX: Skip variance loss for batch_size < 2 (degenerate case)
        if (batch_size < 2) {
            fwd.var_loss = 0.0f;
            return;
        }
        
        for (size_t k = 0; k < config.K; ++k) {
            // Compute mean
            float mean = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                mean += fwd.q[b * config.K + k];
            }
            mean /= batch_size;
            
            // Compute variance with epsilon
            float var = 0.0f;
            for (size_t b = 0; b < batch_size; ++b) {
                float diff = fwd.q[b * config.K + k] - mean;
                var += diff * diff;
            }
            var /= batch_size;
            
            // Stable std computation
            float std = std::sqrt(var + config.variance_epsilon);
            fwd.var_loss += std::max(0.0f, config.zeta - std);
        }
        fwd.var_loss /= config.K;
    }
    
    float train_step(const std::vector<float>& x_batch, float lr, size_t epoch) {
        is_training = true;
        
        // Forward pass
        auto fwd = forward(x_batch, true);
        compute_losses(x_batch, fwd);
        
        // Compute total loss with ramp
        float inv_ramp = std::min(1.0f, static_cast<float>(epoch) / config.inv_ramp_epochs);
        float inv_weight = config.lambda_inv * inv_ramp;
        
        float total_loss = config.lambda_rec * fwd.rec_loss + 
                          inv_weight * fwd.inv_loss + 
                          config.lambda_var * fwd.var_loss;
        
        // Backward pass
        size_t batch_size = x_batch.size() / (config.T * config.D);
        
        // Gradient for reconstruction
        std::vector<float> grad_x_hat(x_batch.size());
        for (size_t i = 0; i < grad_x_hat.size(); ++i) {
            grad_x_hat[i] = -2.0f * config.lambda_rec * (x_batch[i] - fwd.x_hat[i]) / 
                           (batch_size * config.T * config.D);
        }
        
        // Backprop through decoder
        std::vector<float> grad_z = grad_x_hat;
        for (int i = decoder_layers.size() - 1; i >= 0; --i) {
            grad_z = decoder_layers[i].backward(grad_z);
        }
        
        // Backprop through encoder
        std::vector<float> grad_h = grad_z;
        for (int i = encoder_layers.size() - 1; i >= 0; --i) {
            grad_h = encoder_layers[i].backward(grad_h);
        }
        
        // Update weights
        for (auto& layer : encoder_layers) {
            layer.update_adam(lr);
        }
        for (auto& layer : decoder_layers) {
            layer.update_adam(lr);
        }
        
        // Update center during warm-up ONLY
        if (epoch < config.center_warmup_epochs) {
            update_center(fwd.q, fwd.q_prime, batch_size);
        }
        
        is_training = false;
        return total_loss;
    }
    
    // CRITICAL FIX: Make scoring completely stateless
    float score_window(const std::vector<float>& window) {
        // Clear all caches before scoring
        clear_all_caches();
        
        // Normalize window
        auto normalized = processor.normalize(window, config.T);
        
        // Forward pass with training=false to prevent any state updates
        auto fwd = forward(normalized, false);
        compute_losses(normalized, fwd);
        
        // Score based on config
        if (config.score_mix == "inv_only") {
            return fwd.inv_loss;
        } else {  // inv_plus_rec
            return fwd.rec_loss + 0.5f * fwd.inv_loss;
        }
    }
    
    float compute_threshold(const std::vector<float>& val_scores) {
        if (config.threshold_mode == "quantile") {
            // Compute percentile
            std::vector<float> sorted_scores = val_scores;
            std::sort(sorted_scores.begin(), sorted_scores.end());
            size_t idx = static_cast<size_t>(config.threshold_quantile * sorted_scores.size());
            idx = std::min(idx, sorted_scores.size() - 1);
            return sorted_scores[idx];
        } else {  // zscore
            // Mean + k*std
            float mean = std::accumulate(val_scores.begin(), val_scores.end(), 0.0f) / val_scores.size();
            float var = 0.0f;
            for (float s : val_scores) {
                float diff = s - mean;
                var += diff * diff;
            }
            float std = std::sqrt(var / val_scores.size());
            return mean + config.threshold_zscore_k * std;
        }
    }
    
private:
    void l2_normalize_batch(std::vector<float>& vec, size_t dim) {
        size_t batch_size = vec.size() / dim;
        
        for (size_t b = 0; b < batch_size; ++b) {
            float norm = 0.0f;
            for (size_t k = 0; k < dim; ++k) {
                float val = vec[b * dim + k];
                norm += val * val;
            }
            norm = std::sqrt(norm + 1e-8f);
            
            for (size_t k = 0; k < dim; ++k) {
                vec[b * dim + k] /= norm;
            }
        }
    }
    
    // CRITICAL: Only update center during training warmup
    void update_center(const std::vector<float>& q, const std::vector<float>& q_prime, size_t batch_size) {
        // CRITICAL: Only update during training and warmup
        if (!is_training) return;
        
        // Compute batch mean of q and q′
        std::vector<float> mean_q(config.K, 0.0f);
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t k = 0; k < config.K; ++k) {
                mean_q[k] += (q[b * config.K + k] + q_prime[b * config.K + k]) / 2.0f;
            }
        }
        for (auto& m : mean_q) m /= batch_size;
        
        // EMA update
        float alpha = 0.95f;
        for (size_t k = 0; k < config.K; ++k) {
            Ce[k] = alpha * Ce[k] + (1 - alpha) * mean_q[k];
        }
        
        // Renormalize
        float norm = 0.0f;
        for (auto& c : Ce) norm += c * c;
        norm = std::sqrt(norm);
        for (auto& c : Ce) c /= norm;
    }
};

// Training function remains the same but with explicit training mode management
void train_coca_model(COCAModel& model,
                     std::vector<std::vector<float>>& windows,
                     const COCAConfig& config) {
    
    std::cout << "\n=== Training COCA Model ===\n";
    std::cout << "Windows: " << windows.size() << "\n";
    std::cout << "Config: λ_rec=" << config.lambda_rec 
              << ", λ_inv=" << config.lambda_inv 
              << ", λ_var=" << config.lambda_var 
              << ", ζ=" << config.zeta << "\n";
    
    // Set training mode
    model.is_training = true;
    
    // Compute feature statistics
    model.processor.compute_stats(windows, config.T, config.D, config.min_std);
    
    // Split train/val
    std::vector<size_t> indices(windows.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), model.rng);
    
    size_t val_size = static_cast<size_t>(windows.size() * config.val_split);
    size_t train_size = windows.size() - val_size;
    
    std::vector<size_t> train_indices(indices.begin(), indices.begin() + train_size);
    std::vector<size_t> val_indices(indices.begin() + train_size, indices.end());
    
    std::cout << "Train: " << train_indices.size() << " | Val: " << val_indices.size() << "\n\n";
    
    // Initialize weights
    model.init_weights();
    
    // Training loop
    size_t batches_per_epoch = (train_indices.size() + config.batch_size - 1) / config.batch_size;
    float best_val_loss = std::numeric_limits<float>::max();
    size_t patience_counter = 0;
    
    // Log file
    std::ofstream log_file("training_log.csv");
    log_file << "epoch,train_loss,val_loss,rec_loss,inv_loss,var_loss\n";
    
    for (size_t epoch = 0; epoch < config.epochs; ++epoch) {
        std::shuffle(train_indices.begin(), train_indices.end(), model.rng);
        
        float epoch_loss = 0.0f;
        float epoch_rec = 0.0f, epoch_inv = 0.0f, epoch_var = 0.0f;
        
        // Learning rate schedule
        float lr = config.learning_rate;
        if (epoch > config.epochs * 0.5) lr *= 0.5f;
        if (epoch > config.epochs * 0.75) lr *= 0.5f;
        
        // Training batches
        for (size_t batch_idx = 0; batch_idx < batches_per_epoch; ++batch_idx) {
            // Create batch
            std::vector<float> batch;
            size_t batch_start = batch_idx * config.batch_size;
            size_t batch_end = std::min(batch_start + config.batch_size, train_indices.size());
            size_t actual_batch_size = batch_end - batch_start;
            
            // CRITICAL: Only process batches with at least 2 samples for variance
            if (actual_batch_size < 2 && batch_idx > 0) {
                continue; // Skip single-sample batches except the first
            }
            
            for (size_t i = batch_start; i < batch_end; ++i) {
                auto normalized = model.processor.normalize(windows[train_indices[i]], config.T);
                batch.insert(batch.end(), normalized.begin(), normalized.end());
            }
            
            // Pad batch if necessary (but ensure at least 2 samples)
            while (batch.size() < config.batch_size * config.T * config.D) {
                auto normalized = model.processor.normalize(windows[train_indices[0]], config.T);
                batch.insert(batch.end(), normalized.begin(), normalized.end());
            }
            
            // Train step
            float loss = model.train_step(batch, lr, epoch);
            epoch_loss += loss;
            
            // Get component losses for logging
            auto fwd = model.forward(batch, false);
            model.compute_losses(batch, fwd);
            epoch_rec += fwd.rec_loss;
            epoch_inv += fwd.inv_loss;
            epoch_var += fwd.var_loss;
        }
        
        epoch_loss /= batches_per_epoch;
        epoch_rec /= batches_per_epoch;
        epoch_inv /= batches_per_epoch;
        epoch_var /= batches_per_epoch;
        
        // Validation
        float val_loss = 0.0f;
        for (size_t i = 0; i < val_indices.size(); ++i) {
            float score = model.score_window(windows[val_indices[i]]);
            val_loss += score;
        }
        val_loss /= val_indices.size();
        
        // Log metrics
        log_file << epoch << "," << epoch_loss << "," << val_loss << ","
                 << epoch_rec << "," << epoch_inv << "," << epoch_var << "\n";
        
        // Print progress
        if ((epoch + 1) % 5 == 0) {
            std::cout << "Epoch " << std::setw(3) << (epoch + 1) << "/" << config.epochs;
            std::cout << " | Train: " << std::fixed << std::setprecision(6) << epoch_loss;
            std::cout << " | Val: " << val_loss;
            std::cout << " | L_rec=" << epoch_rec;
            std::cout << " | L_inv=" << epoch_inv;
            std::cout << " | L_var=" << epoch_var;
            
            if (val_loss < best_val_loss - config.early_stop_min_delta) {
                best_val_loss = val_loss;
                patience_counter = 0;
                std::cout << " *";
            } else {
                patience_counter++;
            }
            std::cout << "\n";
        }
        
        // Early stopping
        if (patience_counter >= config.early_stop_patience) {
            std::cout << "Early stopping at epoch " << (epoch + 1) << "\n";
            break;
        }
    }
    
    log_file.close();
    
    // CRITICAL: Set training mode to false after training
    model.is_training = false;
    
    // Compute threshold on validation set
    std::vector<float> val_scores;
    for (size_t idx : val_indices) {
        val_scores.push_back(model.score_window(windows[idx]));
    }
    
    model.anomaly_threshold = model.compute_threshold(val_scores);
    
    // Final statistics
    std::sort(val_scores.begin(), val_scores.end());
    float mean_score = std::accumulate(val_scores.begin(), val_scores.end(), 0.0f) / val_scores.size();
    
    std::cout << "\n=== Training Complete ===\n";
    std::cout << "Validation scores:\n";
    std::cout << "  Mean: " << mean_score << "\n";
    std::cout << "  Min: " << val_scores.front() << "\n";
    std::cout << "  Max: " << val_scores.back() << "\n";
    std::cout << "  Threshold: " << model.anomaly_threshold << "\n";
}

} // namespace coca
