# COCA Implementation - CSV Support Update

## Summary

The COCA implementation has been successfully updated to **directly accept CSV files** without requiring conversion to binary format.

## What Changed

### Previous Workflow (Binary-based)
```
CSV → csv_to_bin → Binary → coca_train → Model
```

### New Workflow (Direct CSV)
```
CSV → coca_train → Model
```

## Key Updates

1. **Updated Applications**
   - `coca_train`: Now accepts CSV files directly via `--csv` option
   - `coca_test`: Now accepts CSV files directly via `--csv` option

2. **New CSV Reader**
   - `src/io/csv_reader.hpp`: Full-featured CSV parser
   - Handles missing values (empty cells → 0.0)
   - Automatic feature statistics
   - Windowing support built-in

3. **Preserved Features**
   - All COCA algorithm features intact
   - Configuration file support
   - Model serialization
   - Training logs and metrics

## Usage Examples

### Training
```bash
# Simple training with your data format
./build/bin/coca_train --csv your_robot_data.csv

# With custom parameters
./build/bin/coca_train \
    --csv robot_idle.csv \
    --window 10 \
    --stride 5 \
    --output my_model.coca
```

### Testing
```bash
# Test with trained model
./build/bin/coca_test \
    --csv test_data.csv \
    --model my_model.coca \
    --verbose
```

## CSV Format

The system accepts CSV files matching your provided format:

```csv
timestamp_ms,accel_x_mean,accel_x_std,accel_y_mean,accel_y_std,accel_z_mean,accel_z_std,gyro_x_mean,gyro_y_mean,gyro_z_mean,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,roll,pitch,yaw,yaw_speed,foot_force_0,foot_force_1,foot_force_2,foot_force_3,body_height,mode,gait_type,error_code,power_v_mean,power_a_mean,motor_torque_0,motor_torque_1,motor_torque_2,motor_velocity_0,motor_velocity_1,motor_velocity_2,motor_temp_0,motor_temp_1,motor_temp_2,bit_flag,uwb_distance,uwb_roll,uwb_pitch,uwb_yaw,range_front,range_left,range_right,height_mean,height_std,height_gradient,height_min,height_max,height_range,height_coverage,height_roughness,height_p25,height_median,height_p75,height_obstacles,height_pits,height_resolution,height_width,height_height
0,,,,,,,,,,,,,,,,,,,,,,,,,,,,30.402752,-1.867000,0.519504,-0.865840,3.793203,-0.038755,0.093013,0.000000,33.000000,32.000000,36.000000,36.000000,,,,,,,,,,,,,,,,,,,,,,,
19,-0.179864,0.033396,-0.305261,0.008001,9.602844,0.024927,-0.012783,-0.004794,0.000000,0.157357,0.149516,0.309013,-0.004799,0.004595,-0.017185,-0.019496,0.008548,1.490809,0.005326,0.000000,0.000000,0.000000,0.000000,0.321287,0.000000,0.000000,100.000000,30.402752,-1.867000,0.519504,-0.791625,3.888033,-0.093013,0.093013,0.000000,33.000000,32.000000,36.000000,36.000000,,,,,,,,,,,,,,,,,,,,,,,
```

Key features:
- **Variable number of features**: System adapts to your CSV column count
- **Empty cells**: Automatically handled (converted to 0.0)
- **Header row**: Automatically detected and used for feature names
- **Timestamp column**: First column automatically skipped if timestamp

## Advantages of Direct CSV Support

1. **Simpler Workflow**: No conversion step needed
2. **Immediate Feedback**: See data statistics during loading
3. **Flexible Format**: Handles variable columns, missing values
4. **Direct Integration**: Easy to integrate with data pipelines
5. **Human Readable**: Can inspect and edit training data easily

## Files Included

- `coca_train` - Training application with CSV support
- `coca_test` - Testing application with CSV support  
- `CSV_USAGE_GUIDE.md` - Complete usage documentation
- `robot_data_sample.csv` - Sample data in your format
- `example_data.csv` - Simple example for testing

## Quick Test

Test the implementation with the provided sample:

```bash
# Train on sample robot data
./build/bin/coca_train --csv robot_data_sample.csv --window 3 --stride 1

# View results
cat training_summary.txt
```

## Next Steps

1. Prepare your CSV data file
2. Run training with appropriate window size
3. Test on new data
4. Adjust hyperparameters as needed

The implementation is ready for your robot anomaly detection tasks!
