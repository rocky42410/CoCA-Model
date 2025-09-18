# COCA CSV Usage Guide

The COCA implementation now directly accepts CSV files without requiring conversion to binary format.

## Quick Start

### 1. Training with CSV Data

```bash
# Basic training
./build/bin/coca_train --csv your_data.csv

# With custom parameters
./build/bin/coca_train \
    --csv robot_idle_data.csv \
    --window 10 \              # Window size (default: 10)
    --stride 5 \               # Window stride (default: 5)
    --output my_model.coca \   # Output model file
    --config my_config.yaml    # Custom config file
```

### 2. Testing with CSV Data

```bash
# Basic testing
./build/bin/coca_test --csv test_data.csv --model trained_model.coca

# With options
./build/bin/coca_test \
    --csv robot_test_data.csv \
    --model my_model.coca \
    --window 10 \
    --stride 5 \
    --verbose                  # Show detailed results
```

## CSV Format Requirements

### Expected Structure

Your CSV should have:
- **Header row** (optional, but recommended)
- **Timestamp column** (optional, first column if present)
- **Feature columns** (numeric values)

### Example Format

Based on your provided data:

```csv
timestamp_ms,accel_x_mean,accel_x_std,accel_y_mean,accel_y_std,accel_z_mean,...
0,-0.179864,0.033396,-0.305261,0.008001,9.602844,...
19,-0.189860,0.036552,-0.302867,0.008756,9.610086,...
```

### Handling Missing Values

- Empty cells are converted to 0.0
- NaN and Inf values are converted to 0.0
- The system automatically detects and handles constant features

## Command Line Options

### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--csv <file>` | Input CSV file | Required |
| `--config <file>` | Configuration YAML file | coca_config.yaml |
| `--output <file>` | Output model filename | trained_model.coca |
| `--window <N>` | Window size for time series | 10 |
| `--stride <N>` | Stride between windows | 5 |
| `--no-header` | CSV has no header row | Has header |
| `--no-timestamp` | Don't skip first column | Skip first column |

### Testing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--csv <file>` | Test CSV file | Required |
| `--model <file>` | Trained model file | trained_model.coca |
| `--window <N>` | Window size (should match training) | 10 |
| `--stride <N>` | Stride for creating test windows | 5 |
| `--labels <file>` | Ground truth labels (optional) | None |
| `--verbose` | Show detailed results | Off |
| `--no-header` | CSV has no header row | Has header |
| `--no-timestamp` | Don't skip first column | Skip first column |

## Configuration File

Edit `coca_config.yaml` to tune hyperparameters:

```yaml
# Model architecture
T: 10           # Window size (can override with --window)
C: 32           # Latent dimension
K: 16           # Projection dimension

# Loss weights
lambda_inv: 1.0     # Invariance loss weight
lambda_var: 1.0     # Variance loss weight
lambda_rec: 0.1     # Reconstruction weight

# Training
epochs: 50
batch_size: 32
learning_rate: 0.0005

# Scoring and threshold
score_mix: inv_only         # or 'inv_plus_rec'
threshold_mode: quantile    # or 'zscore'
threshold_quantile: 0.995
```

## Workflow Examples

### Example 1: Basic Idle Detection

```bash
# Train on idle robot data
./build/bin/coca_train --csv robot_idle_30min.csv --window 20 --stride 10

# Test on new data
./build/bin/coca_test --csv robot_test.csv --window 20 --stride 10
```

### Example 2: Custom Configuration

```bash
# Create custom config
cat > custom_config.yaml << EOF
lambda_rec: 0.0    # Pure COCA without reconstruction
lambda_inv: 2.0    # Higher invariance weight
epochs: 100        # More training
score_mix: inv_only
EOF

# Train with custom config
./build/bin/coca_train --csv data.csv --config custom_config.yaml
```

### Example 3: Multiple CSV Files

```bash
# Concatenate multiple CSV files (keep one header)
head -1 idle_session1.csv > combined_idle.csv
tail -n +2 idle_session1.csv >> combined_idle.csv
tail -n +2 idle_session2.csv >> combined_idle.csv
tail -n +2 idle_session3.csv >> combined_idle.csv

# Train on combined data
./build/bin/coca_train --csv combined_idle.csv
```

### Example 4: With Ground Truth Labels

```bash
# Create labels file (0=normal, 1=anomaly)
cat > labels.txt << EOF
0
0
0
1
1
0
EOF

# Test with labels to get metrics
./build/bin/coca_test --csv test.csv --labels labels.txt --verbose
```

## Output Files

After training:
- `trained_model.coca` - Trained model (binary format)
- `training_log.csv` - Training metrics per epoch
- `training_summary.txt` - Training configuration and results
- `training_config_used.yaml` - Exact configuration used

After testing:
- `test_results.csv` - Per-window scores and predictions

## Data Requirements

### Minimum Data
- **Training**: At least 100 windows recommended (e.g., 110 samples for window=10, stride=1)
- **Testing**: At least 1 window (e.g., 10 samples for window=10)

### Feature Count
- Maximum 256 features (additional columns ignored)
- System automatically handles constant and variable features

### Sample Rate
- Data should be uniformly sampled
- Window size should match your anomaly duration
- Example: For 50Hz data and 0.2s anomalies, use window=10

## Troubleshooting

### "Warning: Very few training windows"
- Use smaller stride: `--stride 1`
- Collect more data
- Reduce window size: `--window 5`

### "Feature count mismatch"
- Ensure training and test CSVs have same number of columns
- Check if timestamp column handling matches

### "All predictions are normal/anomaly"
- Check data variation (not all constant)
- Adjust threshold in config
- Try different score_mix setting

### Empty cells in CSV
- These are automatically converted to 0.0
- Consider preprocessing if this isn't appropriate

## Performance Tips

1. **Window Size**: Match expected anomaly duration
2. **Stride**: Use stride=1 for maximum windows, higher for faster training
3. **Features**: Remove obviously irrelevant columns before training
4. **Epochs**: Start with 50, increase if loss still decreasing

## Python Integration

```python
import subprocess
import pandas as pd

# Prepare data
df = pd.read_csv("sensor_data.csv")
df.to_csv("training_data.csv", index=False)

# Train model
subprocess.run([
    "./build/bin/coca_train",
    "--csv", "training_data.csv",
    "--window", "10",
    "--stride", "5"
])

# Test model
subprocess.run([
    "./build/bin/coca_test",
    "--csv", "test_data.csv",
    "--model", "trained_model.coca"
])

# Parse results
results = pd.read_csv("test_results.csv")
anomalies = results[results['detected_anomaly'] == 1]
print(f"Found {len(anomalies)} anomalies")
```

## Next Steps

1. Prepare your CSV data with appropriate features
2. Train initial model with default settings
3. Evaluate on test data
4. Tune hyperparameters based on results
5. Deploy for real-time detection
