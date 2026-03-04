#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Transform.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/UInt8.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float64.h>
#include <mutex>
#include <vector>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>

// replace with actual package message header if different
#include <spinal/FourAxisCommand.h>
#include <spinal/Imu.h>

// ONNX Runtime C++ API
#include <cmath> // for std::isfinite
#include <math.h> // for M_PI


using spinal::FourAxisCommand;
using spinal::Imu;


class GimbalResponse {
public:
    GimbalResponse(const ros::NodeHandle& nh, int control_hz = 50, double duration = 10.0, int gimbal_size = 4,
        double gimbal_range = M_PI, double period_scale = 1.0, double period = 2.0, const std::string& save_path = "/tmp/")
        : nh_(nh),
        control_hz_(control_hz),
        duration_(duration),
        gimbal_size_(gimbal_size),
        gimbal_range_(gimbal_range),
        period_scale_(period_scale),
        period_(period),
        save_path_(save_path)
    {
        // Subscribers
        gimbal_sub_ = nh_.subscribe("joint_states", 1, &GimbalResponse::_gimbal_callback, this);

        // Publishers
        gimbal_pub_ = nh_.advertise<sensor_msgs::JointState>("/beetle_omni/gimbals_ctrl", 1);
        gimbal_pub_debug_ = nh_.advertise<sensor_msgs::JointState>("/beetle_omni/gimbals_ctrl_debug", 1);
        gimbal_effort_pub1_ = nh_.advertise<std_msgs::Float64>("/beetle_omni/servo_controller/gimbals/controller1/simulation/command", 1);
        gimbal_effort_pub2_ = nh_.advertise<std_msgs::Float64>("/beetle_omni/servo_controller/gimbals/controller2/simulation/command", 1);
        gimbal_effort_pub3_ = nh_.advertise<std_msgs::Float64>("/beetle_omni/servo_controller/gimbals/controller3/simulation/command", 1);
        gimbal_effort_pub4_ = nh_.advertise<std_msgs::Float64>("/beetle_omni/servo_controller/gimbals/controller4/simulation/command", 1);

        // defaults
        gimbal_default_pos_ = std::vector<float>(gimbal_size_, 0.0f);

        step_count_ = 0;
        save_count_ = static_cast<int>(control_hz_ * duration_);
    }

    ~GimbalResponse() {
    }

    void spin() {
        // wait for initial messages (timeout)
        ros::Time start = ros::Time::now();
        ros::Duration timeout(500.0);
        ros::Rate r(50);
        while (ros::ok()) {
            {
                std::lock_guard<std::mutex> lk(data_mutex_);
                if (gimbal_catch_) {
                    ROS_INFO("Received gimbal state.");
                    break;
                }
            }
            ros::spinOnce();
            r.sleep();
        }

        // start control loop (Rate-based)
        ROS_INFO("Starting control loop at %.1f Hz", control_hz_);
        ros::Rate control_rate(control_hz_);
        start_time_ = std::chrono::high_resolution_clock::now();
        sensor_msgs::JointState msg;
        step_count_ = 0;
        while (ros::ok()) {
            step_count_++;
            uint64_t infer_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::high_resolution_clock::now() - start_time_
            ).count();
            double timestep = static_cast<double>(infer_ns) / 1e9;
            if (sweep_mode_) {
                target_pos_ = sweepSinTriangleWave(timestep, period_, period_scale_ * period_,
                                                    duration_, static_cast<float>(gimbal_range_));
            } else {
                double cmdphase = std::fmod(timestep, 32 * period_);
                if (cmdphase > 16.0 * period_) {
                    target_pos_ = sinTriangleWave(timestep, period_, gimbal_range_);
                }
                else if (cmdphase > 12.0 * period_) {
                    target_pos_ = trapezoidWave(timestep, period_, gimbal_range_);
                }
                else if (cmdphase > 8.0 * period_) {
                    target_pos_ = squareWave(timestep, period_, gimbal_range_);
                }
                else if (cmdphase > 4.0 * period_) {
                    target_pos_ = triangleWave(timestep, period_, gimbal_range_);
                }
                else {
                    target_pos_ = sinWave(timestep, period_, gimbal_range_);
                }
            }

            msg.header.stamp = ros::Time::now();
            size_t num = target_pos_.size();
            for (size_t i = 0; i < num; ++i) {
                if (!gimbal_enable_[i]) {
                    target_pos_[i] = 0.0f;
                }
                else {
                    target_pos_[i] = std::max(std::min(target_pos_[i], static_cast<float>(gimbal_range_)), static_cast<float>(-gimbal_range_));
                }
            }
            // prepare names safely
            std::vector<std::string> names;
            {
                std::lock_guard<std::mutex> lk(data_mutex_);
                if (gimbal_msg_.name.size() >= num) {
                    names.assign(gimbal_msg_.name.begin(), gimbal_msg_.name.begin() + num);
                } else if (!gimbal_msg_.name.empty()) {
                    names = gimbal_msg_.name;
                    for (size_t j = names.size(); j < num; ++j) names.push_back("gimbal_" + std::to_string(j));
                } else {
                    for (size_t j = 0; j < num; ++j) names.push_back("gimbal_" + std::to_string(j));
                }
            }

            msg.name = names;
            msg.position.resize(num);
            for (size_t i = 0; i < num; ++i) {
                float default_pos = (i < gimbal_default_pos_.size()) ? gimbal_default_pos_[i] : 0.0f;
                msg.position[i] = target_pos_[i] + default_pos;
            }
            if (!gimbal_effort_ctrl_) gimbal_pub_.publish(msg);
            gimbal_pub_debug_.publish(msg);

            if (gimbal_effort_ctrl_) {
                for (size_t i = 0; i < num; ++i) {
                    float effort = 0.0f;
                    {
                        std::lock_guard<std::mutex> lk(data_mutex_);
                        float pos_err = target_pos_[i] + gimbal_default_pos_[i] - gimbal_pos_[i];
                        float vel_err = - gimbal_vel_[i];
                        effort = static_cast<float>(kp_ * pos_err + kd_ * vel_err);
                    }
                    std_msgs::Float64 effort_msg;
                    effort_msg.data = effort;
                    if (i == 0) gimbal_effort_pub1_.publish(effort_msg);
                    else if (i == 1) gimbal_effort_pub2_.publish(effort_msg);
                    else if (i == 2) gimbal_effort_pub3_.publish(effort_msg);
                    else if (i == 3) gimbal_effort_pub4_.publish(effort_msg);
                }
            }

            // warn once in a while if names missing
            {
                std::lock_guard<std::mutex> lk(data_mutex_);
                if (gimbal_msg_.name.size() < num) {
                    static ros::Time last_warn = ros::Time(0);
                    if ((ros::Time::now() - last_warn).toSec() > 5.0) {
                        ROS_WARN("gimbal_msg.name length (%zu) < target_pos length (%zu), using fallback names", gimbal_msg_.name.size(), num);
                        last_warn = ros::Time::now();
                    }
                }
            }

            gimbal_data_.push_back(std::vector<float>());
            gimbal_data_.back().push_back(static_cast<float>(timestep));
            {
                std::lock_guard<std::mutex> lk(data_mutex_);
                for (size_t i = 0; i < num; ++i) {
                    gimbal_data_.back().push_back(gimbal_pos_[i]);
                    last_gimbal_pos_[i] = gimbal_pos_[i];
                }   
            }
            for (size_t i = 0; i < num; ++i) {
                gimbal_data_.back().push_back(static_cast<float>(target_pos_[i]));
                last_target_pos_[i] = target_pos_[i];
            }
            // {
            //     std::lock_guard<std::mutex> lk(data_mutex_);
            //     for (size_t i = 0; i < num; ++i) {
            //         gimbal_data_.back().push_back(gimbal_pos_[i]);
            //     }   
            // }
            // for (size_t i = 0; i < num; ++i) {
            //     gimbal_data_.back().push_back(static_cast<float>(target_pos_[i]));
            // }
            if ( step_count_ % int(control_hz_) == 0) {
                ROS_INFO("Time: %.2f s, data length : %zu.",
                    timestep,
                    gimbal_data_.size()
                );
            }
            if (save_enable_ && step_count_ > save_count_) {
                // save to file with human-readable timestamp YYYY-MM-DD_HH-MM-SS
                std::time_t now_c = std::time(nullptr);
                std::tm tm = *std::localtime(&now_c);
                std::ostringstream ts;
                ts << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
                std::string filename = save_path_ + "/joint_" + ts.str() + ".csv";
                std::ofstream ofs(filename);
                ofs << std::fixed << std::setprecision(6);
                for (const auto& row : gimbal_data_) {
                    for (size_t i = 0; i < row.size(); ++i) {
                        ofs << row[i];
                        if (i + 1 < row.size()) ofs << ",";
                    }
                    ofs << "\n";
                }
                ofs.close();
                ROS_INFO("Gimbal response data saved to %s", filename.c_str());
                ros::shutdown();
                return;
            }
            ros::spinOnce();  // ensure callbacks processed
            control_rate.sleep();
            // optional: detect large loop overrun
        }
    }

    std::vector<float> trapezoidWave(double t, double period, float amplitude) {
        std::vector<float> wave(gimbal_size_, 0.0f);
        wave = triangleWave(t, period, amplitude * 2.0f);
        for (int i = 0; i < gimbal_size_; ++i) {
            if (wave[i] > amplitude) wave[i] = amplitude;
            if (wave[i] < -amplitude) wave[i] = -amplitude;
        }
        return wave;
    }

    std::vector<float> triangleWave(double t, double period, float amplitude) {
        std::vector<float> wave(gimbal_size_, 0.0f);
        double phase = std::fmod(t + period / 4.0, period);
        double half = period / 2.0;
        // slope to go from -A to +A over half period: 4*A/period
        double slope = (2.0 * amplitude) / half; // = 4*A/period
        float val;
        if (phase < half) {
            val = static_cast<float>(-amplitude + slope * phase);
        } else {
            val = static_cast<float>(amplitude - slope * (phase - half));
        }
        for (int i = 0; i < gimbal_size_; ++i) wave[i] = val;
        return wave;
    }

    std::vector<float> squareWave(double t, double period, float amplitude) {
        std::vector<float> wave(gimbal_size_, 0.0f);
        double phase = std::fmod(t, period);
        float val = (phase < period / 2.0) ? amplitude : -amplitude;
        for (int i = 0; i < gimbal_size_; ++i) wave[i] = val;
        return wave;
    }

    std::vector<float> sinWave(double t, double period, float amplitude) {
        std::vector<float> wave(gimbal_size_, 0.0f);
        double phase = std::fmod(t, period);
        float val = static_cast<float>(amplitude * std::sin((2.0 * M_PI / period) * phase));
        for (int i = 0; i < gimbal_size_; ++i) wave[i] = val;
        return wave;
    }

    std::vector<float> sinTriangleWave(double t, double period, float amplitude) {
        std::vector<float> wave(gimbal_size_, 0.0f);
        double phase = std::fmod(t, period);
        std::vector<float> triangleVal = triangleWave(t, period * 32.0, amplitude);
        amplitude = std::abs(triangleVal[0]);
        float val = static_cast<float>(amplitude * std::sin((2.0 * M_PI / period) * phase));
        for (int i = 0; i < gimbal_size_; ++i) wave[i] = val;
        return wave;
    }

    // Chirp sine wave whose amplitude is modulated by a triangle envelope.
    // - period sweeps linearly from period_min to period_max over the active window
    // - envelope triangle period = 16 * current_period  (abs-folded → two amplitude bumps)
    // - 1 s silence (return 0 = default) is prepended and appended for easy trimming
    std::vector<float> sweepSinTriangleWave(double t, double period_min, double period_max,
                                             double duration, float amplitude) {
        std::vector<float> wave(gimbal_size_, 0.0f);
        const double pad = 1.0;
        const double active_dur = duration - 2.0 * pad;
        // padding windows → return default (0)
        if (t < pad || t > duration - pad || active_dur <= 0.0) return wave;

        double t_a = t - pad;                               // time within active segment [0, active_dur]
        double alpha = t_a / active_dur;                    // [0, 1]
        double cur_period = period_min + (period_max - period_min) * alpha;

        // Chirp phase: φ = 2π ∫₀^{t_a} 1/period(s) ds
        // period(s) = p0 + (p1-p0)*s/T  →  φ = 2π·T/(p1-p0)·ln(cur_period/p0)
        double phi;
        if (std::abs(period_max - period_min) < 1e-9) {
            phi = 2.0 * M_PI * t_a / period_min;
        } else {
            phi = 2.0 * M_PI * active_dur / (period_max - period_min)
                  * std::log(cur_period / period_min);
        }

        // Amplitude envelope: |triangleWave| with period = 16 * cur_period
        float env = std::abs(triangleWave(t_a, 16.0 * cur_period, amplitude)[0]);

        float val = static_cast<float>(env * std::sin(phi));
        for (int i = 0; i < gimbal_size_; ++i) wave[i] = val;
        return wave;
    }

    void setControlEnable(bool gimbal_enable_0, bool gimbal_enable_1,
                          bool gimbal_enable_2, bool gimbal_enable_3) {
        gimbal_enable_[0] = gimbal_enable_0;
        gimbal_enable_[1] = gimbal_enable_1;
        gimbal_enable_[2] = gimbal_enable_2;
        gimbal_enable_[3] = gimbal_enable_3;
    }
    void setSaveEnable(bool save_enable) {
        save_enable_ = save_enable;
    }
    void setSweepMode(bool sweep_mode) {
        sweep_mode_ = sweep_mode;
    }
    void setGains(double kp, double kd, bool gimbal_effort_ctrl, double default_gimbal) {
        kp_ = kp;
        kd_ = kd;
        gimbal_effort_ctrl_ = gimbal_effort_ctrl;
        gimbal_default_pos_ = std::vector<float>(4, static_cast<float>(default_gimbal));
    }

private:
    // -------- ROS
    ros::NodeHandle nh_;
    ros::Subscriber gimbal_sub_;
    ros::Publisher gimbal_pub_;
    ros::Publisher gimbal_pub_debug_;
    ros::Publisher gimbal_effort_pub1_;
    ros::Publisher gimbal_effort_pub2_;
    ros::Publisher gimbal_effort_pub3_;
    ros::Publisher gimbal_effort_pub4_;

    // -------- data
    std::mutex data_mutex_;
    sensor_msgs::JointState gimbal_msg_;
    bool gimbal_catch_ = false;
    double kp_ = 0.285;
    double kd_ = 0.019;
    bool gimbal_effort_ctrl_ = false;

    // -------- control
    double control_hz_;
    int step_count_ = 0;
    double duration_ = 2.0; // seconds
    double period_ = 2.0; // seconds
    double gimbal_range_ = M_PI; // radians
    double period_scale_ = 1.0; // scale factor for period
    int gimbal_size_ = 4; // number of gimbals to control
    std::string save_path_;
    bool save_enable_ = false;
    bool sweep_mode_ = false;
    int save_count_ = 0;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::vector<std::vector<float>> gimbal_data_; // [step][data]
    std::vector<float> gimbal_pos_ = std::vector<float>(gimbal_size_, 0.0f);
    std::vector<float> gimbal_vel_ = std::vector<float>(gimbal_size_, 0.0f);
    std::vector<float> last_gimbal_pos_ = std::vector<float>(gimbal_size_, 0.0f);
    std::vector<float> gimbal_default_pos_;
    // publishers data
    std::vector<float> target_pos_ = std::vector<float>(gimbal_size_, 0.0f);
    std::vector<float> last_target_pos_ = std::vector<float>(gimbal_size_, 0.0f);
    std::vector<bool> gimbal_enable_ = std::vector<bool>(gimbal_size_, true);

    void _gimbal_callback(const sensor_msgs::JointState::ConstPtr& msg) {
        std::lock_guard<std::mutex> lk(data_mutex_);
        gimbal_msg_ = *msg;
        // copy positions safely
        for (size_t i = 0; i < std::min<size_t>(msg->position.size(), gimbal_pos_.size()); ++i) {
            gimbal_pos_[i] = msg->position[i];
            if (gimbal_effort_ctrl_)
                gimbal_vel_[i] = msg->velocity[i];
        }
        gimbal_catch_ = true;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "gimbal_response");
    ros::NodeHandle nh("~");

    int freq;
    bool enable_gimbal_0, enable_gimbal_1, enable_gimbal_2, enable_gimbal_3;
    bool enable_save;
    int gimbal_size = 4;
    double duration;
    double gimbal_range;
    double cmd_period;
    double period_scale;
    double kp_, kd_;
    bool gimbal_effort_ctrl;
    double default_gimbal;
    nh.param<double>("kp", kp_, 0.285);
    nh.param<double>("kd", kd_, 0.019);
    nh.param<bool>("effort_ctrl", gimbal_effort_ctrl, false);
    std::string save_path;
    nh.param<double>("duration", duration, 10.0);
    nh.param<double>("default_gimbal", default_gimbal, 0.0);
    nh.param<int>("control_freq", freq, 200);
    nh.param<double>("gimbal_range", gimbal_range, M_PI);
    nh.param<double>("cmd_period", cmd_period, 2.0);
    nh.param<double>("period_scale", period_scale, 1.0);
    nh.param<std::string>("save_path", save_path, std::string("/tmp/"));
    nh.param<bool>("enable_save", enable_save, false);
    bool sweep_mode;
    nh.param<bool>("sweep_mode", sweep_mode, false);
    nh.param<int>("gimbal_size", gimbal_size, 4);
    nh.param<bool>("enable_gimbal_0", enable_gimbal_0, false);
    nh.param<bool>("enable_gimbal_1", enable_gimbal_1, false);
    nh.param<bool>("enable_gimbal_2", enable_gimbal_2, false);
    nh.param<bool>("enable_gimbal_3", enable_gimbal_3, false);
    ROS_INFO("============================================");
    ROS_INFO("Gimbal Response:\n"
        "save_path=%s,\n"
        " control_freq=%d Hz, duration=%.1f s,\n"
        " enable_gimbal=[%d,%d,%d,%d], enable_save=%d",
        save_path.c_str(),
        freq, duration,
        static_cast<int>(enable_gimbal_0),
        static_cast<int>(enable_gimbal_1),
        static_cast<int>(enable_gimbal_2),
        static_cast<int>(enable_gimbal_3),
        static_cast<int>(enable_save)
    );

    try {
        GimbalResponse response(nh, freq, duration, gimbal_size, gimbal_range, period_scale, cmd_period, save_path);
        response.setControlEnable(enable_gimbal_0, enable_gimbal_1, enable_gimbal_2, enable_gimbal_3);
        response.setSaveEnable(enable_save);
        response.setSweepMode(sweep_mode);
        response.setGains(kp_, kd_, gimbal_effort_ctrl, default_gimbal);
        response.spin();
    } catch (const std::exception& e) {
        ROS_ERROR("Exception: %s", e.what());
        return 1;
    }

    return 0;
}