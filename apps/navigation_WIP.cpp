#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <deque>
#include <chrono>
#include <thread>
#include <iomanip>
#include <algorithm>
#include <cstring>

// Configuration Constants
constexpr double RATE_HZ = 50.0;
constexpr double DT = 1.0 / RATE_HZ;

// Path geometry (world frame)
constexpr double PATH_LENGTH = 2.0;  // meters

// Controller parameters
constexpr double V_NOM = 0.25;       // m/s forward along segment
constexpr double KY = 0.6;           // lateral PD gain
constexpr double KPSI = 0.6;         // heading PD gain
constexpr double VY_LIM = 0.10;      // m/s
constexpr double WZ_LIM = 0.50;      // rad/s

// Command low-pass filter
constexpr double TAU_LP = 0.25;      // seconds

// Estimator complementary gains
constexpr double K_POS = 0.05;       // position gain
constexpr double K_YAW = 0.05;       // yaw gain

// Innovation rate limits (per second, will scale by DT)
constexpr double INNOV_RATE_XY_MAX = 1.5 * V_NOM;  // m/s
constexpr double INNOV_RATE_YAW_MAX = 0.2;         // rad/s

// UWB gating thresholds
constexpr double UWB_MAX_JUMP_XY = 0.5;     // meters
constexpr double UWB_MAX_JUMP_YAW = 0.35;   // rad
constexpr double UWB_MIN_QUALITY = 0.5;     // 0..1
constexpr double UWB_MAX_LATENCY = 0.20;    // seconds

// Position tolerance for waypoint reaching
constexpr double POS_TOL = 0.05;  // meters

// Pi constant
constexpr double PI = 3.14159265358979323846;

// Enums
enum class Mode {
    OPEN_LOOP,
    SMOOTH_LOOP,
    SHADOW
};

enum class Phase {
    FWD,
    SIT1,
    TURN,
    BACK,
    SIT2
};

// Basic structures
struct Pose {
    double x;
    double y;
    double yaw;
    
    Pose(double x_ = 0, double y_ = 0, double yaw_ = 0) 
        : x(x_), y(y_), yaw(yaw_) {}
    
    Pose operator-(const Pose& other) const {
        return Pose(x - other.x, y - other.y, yaw - other.yaw);
    }
    
    Pose operator+(const Pose& other) const {
        return Pose(x + other.x, y + other.y, yaw + other.yaw);
    }
};

struct Twist {
    double vx;
    double vy;
    double wz;
    
    Twist(double vx_ = 0, double vy_ = 0, double wz_ = 0)
        : vx(vx_), vy(vy_), wz(wz_) {}
};

struct UWBMeas {
    Pose pose_world;
    double timestamp;
    double quality;
    bool valid;
    
    UWBMeas() : timestamp(0), quality(0), valid(false) {}
};

struct PathSegment {
    Pose A;
    Pose B;
    double heading;
    bool valid;
    
    PathSegment() : valid(false) {}
    PathSegment(const Pose& a, const Pose& b, double h) 
        : A(a), B(b), heading(h), valid(true) {}
};

struct LogEntry {
    double ts;
    Mode mode;
    Phase phase;
    
    // Raw UWB
    Pose uwb_raw_pose;
    double uwb_quality;
    double uwb_latency;
    bool uwb_valid;
    
    // Estimator
    Pose pred_pose;
    Pose est_pose;
    Pose innov_xyz;
    bool innov_applied;
    Pose innov_clipped;
    
    // Commands
    Twist cmd_sent;
    double lp_alpha;
    
    // Path tracking errors
    double cross_track;
    double heading_err;
};

// Helper functions
inline double wrap_angle(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

inline double angdiff(double a, double b) {
    return wrap_angle(a - b);
}

inline double clamp(double val, double min_val, double max_val) {
    return std::max(min_val, std::min(val, max_val));
}

inline double norm2D(double dx, double dy) {
    return std::hypot(dx, dy);
}

inline double get_time() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

inline double alpha_from_tau(double dt, double tau) {
    return std::exp(-dt / tau);
}

// Main controller class
class Go2FusionController {
private:
    // State
    Pose est_pose;
    Pose pred_pose;
    Twist last_cmd;
    double lp_vy_state;
    double lp_wz_state;
    
    // Phase machine
    Phase phase;
    double hold_timer;
    
    // Sensor buffers
    std::deque<UWBMeas> uwb_buffer;
    UWBMeas prev_uwb;
    bool has_prev_uwb;
    
    // Configuration
    Mode mode;
    double ky_adaptive;
    double kpsi_adaptive;
    
    // Logging
    std::ofstream log_file;
    std::vector<LogEntry> log_buffer;
    
    // Timing
    double start_time;
    double last_uwb_time;
    
public:
    Go2FusionController(Mode initial_mode = Mode::SMOOTH_LOOP) 
        : est_pose(0, 0, 0), pred_pose(0, 0, 0), last_cmd(0, 0, 0),
          lp_vy_state(0), lp_wz_state(0), phase(Phase::FWD), hold_timer(0),
          has_prev_uwb(false), mode(initial_mode),
          ky_adaptive(KY), kpsi_adaptive(KPSI) {
        
        start_time = get_time();
        last_uwb_time = start_time;
        
        // Open log file
        log_file.open("fusion_controller_log.csv");
        write_log_header();
    }
    
    ~Go2FusionController() {
        if (log_file.is_open()) {
            flush_logs();
            log_file.close();
        }
    }
    
    void write_log_header() {
        log_file << "ts,mode,phase,uwb_x,uwb_y,uwb_yaw,uwb_quality,uwb_latency,uwb_valid,"
                 << "pred_x,pred_y,pred_yaw,est_x,est_y,est_yaw,"
                 << "innov_x,innov_y,innov_yaw,innov_applied,"
                 << "innov_clip_x,innov_clip_y,innov_clip_yaw,"
                 << "cmd_vx,cmd_vy,cmd_wz,lp_alpha,cross_track,heading_err\n";
    }
    
    void flush_logs() {
        for (const auto& entry : log_buffer) {
            log_file << std::fixed << std::setprecision(6)
                    << entry.ts << ","
                    << static_cast<int>(entry.mode) << ","
                    << static_cast<int>(entry.phase) << ","
                    << entry.uwb_raw_pose.x << ","
                    << entry.uwb_raw_pose.y << ","
                    << entry.uwb_raw_pose.yaw << ","
                    << entry.uwb_quality << ","
                    << entry.uwb_latency << ","
                    << entry.uwb_valid << ","
                    << entry.pred_pose.x << ","
                    << entry.pred_pose.y << ","
                    << entry.pred_pose.yaw << ","
                    << entry.est_pose.x << ","
                    << entry.est_pose.y << ","
                    << entry.est_pose.yaw << ","
                    << entry.innov_xyz.x << ","
                    << entry.innov_xyz.y << ","
                    << entry.innov_xyz.yaw << ","
                    << entry.innov_applied << ","
                    << entry.innov_clipped.x << ","
                    << entry.innov_clipped.y << ","
                    << entry.innov_clipped.yaw << ","
                    << entry.cmd_sent.vx << ","
                    << entry.cmd_sent.vy << ","
                    << entry.cmd_sent.wz << ","
                    << entry.lp_alpha << ","
                    << entry.cross_track << ","
                    << entry.heading_err << "\n";
        }
        log_buffer.clear();
    }
    
    Pose propagate(const Pose& pose, const Twist& cmd, double dt) {
        double c = std::cos(pose.yaw);
        double s = std::sin(pose.yaw);
        double dx = (cmd.vx * c - cmd.vy * s) * dt;
        double dy = (cmd.vx * s + cmd.vy * c) * dt;
        double dyaw = cmd.wz * dt;
        return Pose(pose.x + dx, pose.y + dy, wrap_angle(pose.yaw + dyaw));
    }
    
    Pose predict_uwb_to_now(const UWBMeas& meas, const Twist& cmd_since, double dt_since) {
        return propagate(meas.pose_world, cmd_since, dt_since);
    }
    
    bool valid_uwb(const UWBMeas& prev, const UWBMeas& curr) {
        if (curr.quality < UWB_MIN_QUALITY) return false;
        
        if (has_prev_uwb) {
            double dist = norm2D(curr.pose_world.x - prev.pose_world.x,
                               curr.pose_world.y - prev.pose_world.y);
            if (dist > UWB_MAX_JUMP_XY) return false;
            
            double yaw_diff = std::abs(angdiff(curr.pose_world.yaw, prev.pose_world.yaw));
            if (yaw_diff > UWB_MAX_JUMP_YAW) return false;
        }
        return true;
    }
    
    bool gate_innovation(const Pose& innov) {
        if (std::abs(innov.x) > 2.0 || std::abs(innov.y) > 2.0) return false;
        if (std::abs(innov.yaw) > 0.8) return false;
        return true;
    }
    
    Pose rate_limit_innovation(const Pose& innov, double dt) {
        double max_dx = INNOV_RATE_XY_MAX * dt;
        double max_dy = INNOV_RATE_XY_MAX * dt;
        double max_dyaw = INNOV_RATE_YAW_MAX * dt;
        
        return Pose(
            clamp(innov.x, -max_dx, max_dx),
            clamp(innov.y, -max_dy, max_dy),
            clamp(innov.yaw, -max_dyaw, max_dyaw)
        );
    }
    
    std::tuple<Pose, bool, Pose> fuse_complementary(const Pose& pred, const Pose& uwb, double dt) {
        Pose innov(
            uwb.x - pred.x,
            uwb.y - pred.y,
            angdiff(uwb.yaw, pred.yaw)
        );
        
        if (!gate_innovation(innov)) {
            return {pred, false, Pose(0, 0, 0)};
        }
        
        Pose bounded = rate_limit_innovation(innov, dt);
        
        Pose fused(
            pred.x + K_POS * bounded.x,
            pred.y + K_POS * bounded.y,
            wrap_angle(pred.yaw + K_YAW * bounded.yaw)
        );
        
        Pose clipped(
            bounded.x != innov.x ? 1 : 0,
            bounded.y != innov.y ? 1 : 0,
            bounded.yaw != innov.yaw ? 1 : 0
        );
        
        return {fused, true, clipped};
    }
    
    PathSegment segment_for_phase(Phase p) {
        if (p == Phase::FWD) {
            return PathSegment(Pose(0, 0, 0), Pose(PATH_LENGTH, 0, 0), 0);
        } else if (p == Phase::BACK) {
            return PathSegment(Pose(PATH_LENGTH, 0, PI), Pose(0, 0, PI), PI);
        }
        return PathSegment();  // invalid for SIT/TURN
    }
    
    std::pair<double, double> cross_track_and_heading_error(const Pose& pose, const PathSegment& seg) {
        double dx = seg.B.x - seg.A.x;
        double dy = seg.B.y - seg.A.y;
        double L = norm2D(dx, dy);
        
        if (L < 1e-6) return {0, 0};  // degenerate segment
        
        double nx = -dy / L;
        double ny = dx / L;
        double rx = pose.x - seg.A.x;
        double ry = pose.y - seg.A.y;
        double e_y = rx * nx + ry * ny;
        double e_psi = angdiff(seg.heading, pose.yaw);
        
        return {e_y, e_psi};
    }
    
    std::tuple<Twist, double, double> path_track(const Pose& pose, const PathSegment& seg) {
        auto [e_y, e_psi] = cross_track_and_heading_error(pose, seg);
        
        Twist cmd;
        cmd.vx = V_NOM;
        cmd.vy = clamp(-ky_adaptive * e_y, -VY_LIM, VY_LIM);
        cmd.wz = clamp(-kpsi_adaptive * e_psi, -WZ_LIM, WZ_LIM);
        
        return {cmd, e_y, e_psi};
    }
    
    double lp_step(double prev_y, double x, double alpha) {
        return alpha * prev_y + (1 - alpha) * x;
    }
    
    bool reached(const Pose& target, const Pose& pose, double tol) {
        return norm2D(pose.x - target.x, pose.y - target.y) < tol;
    }
    
    void send_twist(const Twist& cmd) {
        // In real implementation, this would send to Go2 motion service
        // For now, just record it
        last_cmd = cmd;
    }
    
    void send_sit() {
        // Send sit command to robot
        std::cout << "[CMD] SIT\n";
    }
    
    void send_stand() {
        // Send stand command to robot
        std::cout << "[CMD] STAND\n";
    }
    
    void add_uwb_measurement(const Pose& pose, double quality) {
        UWBMeas meas;
        meas.pose_world = pose;
        meas.timestamp = get_time();
        meas.quality = quality;
        meas.valid = true;
        
        uwb_buffer.push_back(meas);
        if (uwb_buffer.size() > 200) {  // Keep last ~4s at 50Hz
            uwb_buffer.pop_front();
        }
    }
    
    UWBMeas get_latest_uwb() {
        if (uwb_buffer.empty()) {
            return UWBMeas();
        }
        return uwb_buffer.back();
    }
    
    void control_tick() {
        double NOW = get_time();
        double alpha_lp = alpha_from_tau(DT, TAU_LP);
        
        // 1) READ SENSORS
        UWBMeas meas = get_latest_uwb();
        bool have_uwb = false;
        Pose uwb_pose_now;
        double uwb_latency = 0;
        
        if (meas.valid) {
            uwb_latency = NOW - meas.timestamp;
            if (uwb_latency <= UWB_MAX_LATENCY && valid_uwb(prev_uwb, meas)) {
                uwb_pose_now = predict_uwb_to_now(meas, last_cmd, uwb_latency);
                have_uwb = true;
                prev_uwb = meas;
                has_prev_uwb = true;
                last_uwb_time = NOW;
            }
        }
        
        // 2) ESTIMATOR PREDICTION
        pred_pose = propagate(est_pose, last_cmd, DT);
        
        // 3) ESTIMATOR UPDATE (FUSION)
        bool innov_applied = false;
        Pose innov_clipped(0, 0, 0);
        Pose innov_xyz(0, 0, 0);
        
        if (mode == Mode::SMOOTH_LOOP && have_uwb) {
            auto [fused, applied, clipped] = fuse_complementary(pred_pose, uwb_pose_now, DT);
            est_pose = fused;
            innov_applied = applied;
            innov_clipped = clipped;
            innov_xyz = uwb_pose_now - pred_pose;
        } else {
            est_pose = pred_pose;
        }
        
        // Shadow mode calculation
        if (mode == Mode::SHADOW && have_uwb) {
            auto [shadow_fused, shadow_applied, shadow_clipped] = 
                fuse_complementary(pred_pose, uwb_pose_now, DT);
            // Log shadow results but don't apply
            innov_xyz = uwb_pose_now - pred_pose;
        }
        
        // 4) PHASE STATE MACHINE
        Twist cmd(0, 0, 0);
        double e_y = 0, e_psi = 0;
        
        switch (phase) {
            case Phase::FWD: {
                PathSegment seg = segment_for_phase(Phase::FWD);
                auto [track_cmd, track_ey, track_epsi] = path_track(est_pose, seg);
                cmd = track_cmd;
                e_y = track_ey;
                e_psi = track_epsi;
                
                if (reached(seg.B, est_pose, POS_TOL)) {
                    send_twist(Twist(0, 0, 0));
                    send_sit();
                    if (have_uwb) est_pose = uwb_pose_now;  // optional reanchor
                    phase = Phase::SIT1;
                    hold_timer = NOW + 2.0;
                } else {
                    cmd.vy = lp_step(lp_vy_state, cmd.vy, alpha_lp);
                    lp_vy_state = cmd.vy;
                    cmd.wz = lp_step(lp_wz_state, cmd.wz, alpha_lp);
                    lp_wz_state = cmd.wz;
                    send_twist(cmd);
                }
                break;
            }
            
            case Phase::SIT1: {
                if (NOW >= hold_timer) {
                    send_stand();
                    phase = Phase::TURN;
                } else {
                    last_cmd = Twist(0, 0, 0);
                }
                break;
            }
            
            case Phase::TURN: {
                double target_yaw = PI;
                double epsi = angdiff(target_yaw, est_pose.yaw);
                cmd = Twist(0, 0, clamp(0.6 * epsi, -0.5, 0.5));
                
                if (std::abs(epsi) < 0.05) {
                    send_twist(Twist(0, 0, 0));
                    phase = Phase::BACK;
                    last_cmd = Twist(0, 0, 0);
                } else {
                    cmd.wz = lp_step(lp_wz_state, cmd.wz, alpha_lp);
                    lp_wz_state = cmd.wz;
                    send_twist(cmd);
                }
                break;
            }
            
            case Phase::BACK: {
                PathSegment seg = segment_for_phase(Phase::BACK);
                auto [track_cmd, track_ey, track_epsi] = path_track(est_pose, seg);
                cmd = track_cmd;
                e_y = track_ey;
                e_psi = track_epsi;
                
                if (reached(seg.B, est_pose, POS_TOL)) {
                    send_twist(Twist(0, 0, 0));
                    send_sit();
                    if (have_uwb) est_pose = uwb_pose_now;
                    phase = Phase::SIT2;
                    hold_timer = NOW + 2.0;
                } else {
                    cmd.vy = lp_step(lp_vy_state, cmd.vy, alpha_lp);
                    lp_vy_state = cmd.vy;
                    cmd.wz = lp_step(lp_wz_state, cmd.wz, alpha_lp);
                    lp_wz_state = cmd.wz;
                    send_twist(cmd);
                }
                break;
            }
            
            case Phase::SIT2: {
                if (NOW >= hold_timer) {
                    send_stand();
                    phase = Phase::FWD;
                } else {
                    last_cmd = Twist(0, 0, 0);
                }
                break;
            }
        }
        
        // 6) LOGGING
        LogEntry entry;
        entry.ts = NOW - start_time;
        entry.mode = mode;
        entry.phase = phase;
        entry.uwb_raw_pose = meas.valid ? meas.pose_world : Pose(0, 0, 0);
        entry.uwb_quality = meas.quality;
        entry.uwb_latency = uwb_latency;
        entry.uwb_valid = have_uwb;
        entry.pred_pose = pred_pose;
        entry.est_pose = est_pose;
        entry.innov_xyz = innov_xyz;
        entry.innov_applied = (mode == Mode::SMOOTH_LOOP) ? innov_applied : false;
        entry.innov_clipped = innov_clipped;
        entry.cmd_sent = last_cmd;
        entry.lp_alpha = alpha_lp;
        entry.cross_track = (phase == Phase::FWD || phase == Phase::BACK) ? e_y : 0;
        entry.heading_err = (phase == Phase::FWD || phase == Phase::BACK) ? e_psi : 0;
        
        log_buffer.push_back(entry);
        
        // Periodic log flush
        if (log_buffer.size() >= 100) {
            flush_logs();
        }
        
        // 7) DROPOUT HANDLING
        if ((NOW - last_uwb_time) > 2.0) {
            ky_adaptive = std::min(ky_adaptive, 0.5);
            kpsi_adaptive = std::min(kpsi_adaptive, 0.6);
        } else {
            ky_adaptive = KY;
            kpsi_adaptive = KPSI;
        }
    }
    
    void run() {
        std::cout << "Starting Go2 Fusion Controller\n";
        std::cout << "Mode: " << (mode == Mode::SMOOTH_LOOP ? "SMOOTH_LOOP" : 
                                 mode == Mode::OPEN_LOOP ? "OPEN_LOOP" : "SHADOW") << "\n";
        std::cout << "Path length: " << PATH_LENGTH << " meters\n";
        std::cout << "Control rate: " << RATE_HZ << " Hz\n\n";
        
        auto next_tick = std::chrono::steady_clock::now();
        const auto tick_duration = std::chrono::microseconds(static_cast<long>(1000000.0 / RATE_HZ));
        
        // Main control loop
        for (int iter = 0; iter < 1000; ++iter) {  // Run for 20 seconds (1000 ticks at 50Hz)
            // Simulate UWB measurements (in real system, these come from UWB hardware)
            if (iter % 10 == 0) {  // UWB at 5Hz
                // Add some noise to simulate real UWB
                double noise_x = (std::rand() % 100 - 50) * 0.001;
                double noise_y = (std::rand() % 100 - 50) * 0.001;
                double noise_yaw = (std::rand() % 100 - 50) * 0.001;
                
                add_uwb_measurement(
                    Pose(est_pose.x + noise_x, 
                         est_pose.y + noise_y, 
                         est_pose.yaw + noise_yaw),
                    0.9  // quality
                );
            }
            
            // Execute control tick
            control_tick();
            
            // Sleep until next tick
            next_tick += tick_duration;
            std::this_thread::sleep_until(next_tick);
            
            // Print status every second
            if (iter % 50 == 0) {
                std::cout << "t=" << std::fixed << std::setprecision(1) << (iter * DT)
                         << "s | Phase: " << static_cast<int>(phase)
                         << " | Pose: (" << std::setprecision(3) << est_pose.x 
                         << ", " << est_pose.y << ", " << est_pose.yaw << ")"
                         << " | Cmd: (" << last_cmd.vx << ", " << last_cmd.vy 
                         << ", " << last_cmd.wz << ")\n";
            }
        }
        
        // Final flush
        flush_logs();
        std::cout << "\nController finished. Log saved to fusion_controller_log.csv\n";
    }
};

// Main function
int main(int argc, char* argv[]) {
    // Parse command line for mode selection
    Mode mode = Mode::SMOOTH_LOOP;
    if (argc > 1) {
        std::string mode_str(argv[1]);
        if (mode_str == "open_loop") {
            mode = Mode::OPEN_LOOP;
        } else if (mode_str == "shadow") {
            mode = Mode::SHADOW;
        }
    }
    
    // Create and run controller
    Go2FusionController controller(mode);
    controller.run();
    
    return 0;
}