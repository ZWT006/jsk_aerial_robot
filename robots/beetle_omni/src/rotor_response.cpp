#include <ros/ros.h>

#include <geometry_msgs/WrenchStamped.h>
#include <spinal/ESCTelemetry.h>
#include <spinal/ESCTelemetryArray.h>
#include <spinal/FourAxisCommand.h>
#include <std_msgs/Empty.h>
#include <std_srvs/Empty.h>
#include <takasako_sps/PowerInfo.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

class RotorResponse {
public:
    RotorResponse(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
        : nh_(nh), nhp_(nh_private)
    {
        nhp_.param("control_freq", control_hz_, 100);
        nhp_.param("duration", duration_, 100.0);
        nhp_.param("rotor_size", rotor_size_, 4);
        nhp_.param("default_thrust", default_thrust_, 7.0);
        nhp_.param("thrust_range", thrust_range_, 3.0);
        nhp_.param("thrust_min", thrust_min_, 0.0);
        nhp_.param("thrust_max", thrust_max_, 20.0);
        nhp_.param("disabled_thrust", disabled_thrust_, 0.0);
        nhp_.param("shutdown_thrust", shutdown_thrust_, 0.0);
        nhp_.param("cmd_period", period_, 1.0);
        nhp_.param("period_scale", period_scale_, 4.0);
        nhp_.param("save_path", save_path_, std::string("/tmp"));
        nhp_.param("enable_save", save_enable_, true);

        nhp_.param("enable_force_sensor", enable_force_sensor_, true);
        nhp_.param("enable_force_sensor_calib", enable_force_sensor_calib_, true);
        nhp_.param("force_sensor_calib_service", force_sensor_calib_service_, std::string("/cfs_sensor_calib"));
        nhp_.param("enable_power_info", enable_power_info_, true);
        nhp_.param("enable_power_on_cmd", enable_power_on_cmd_, true);
        nhp_.param("enable_esc_telem", enable_esc_telem_, true);

        nhp_.param("thrust_pub_name", thrust_pub_name_, std::string("/beetle_omni/four_axes/command"));
        nhp_.param("thrust_debug_pub_name", thrust_debug_pub_name_, std::string("/beetle_omni/four_axes/command_debug"));
        nhp_.param("force_sensor_sub_name", force_sensor_sub_name_, std::string("/cfs/data"));
        nhp_.param("power_info_sub_name", power_info_sub_name_, std::string("/power_info"));
        nhp_.param("esc_telem_sub_name", esc_telem_sub_name_, std::string("/beetle_omni/esc_telem"));

        control_hz_ = std::max(1, control_hz_);
        duration_ = std::max(0.0, duration_);
        period_ = std::max(1e-6, period_);
        period_scale_ = std::max(1e-6, period_scale_);
        if (thrust_max_ < thrust_min_) {
            std::swap(thrust_max_, thrust_min_);
        }

        rotor_size_ = std::max(1, rotor_size_);
        rotor_enable_.assign(rotor_size_, true);
        for (int i = 0; i < rotor_size_; ++i) {
            bool enable = true;
            nhp_.param("enable_rotor_" + std::to_string(i), enable, true);
            rotor_enable_[i] = enable;
        }

        target_thrust_.assign(rotor_size_, static_cast<float>(disabled_thrust_));
        esc_telemetry_.fill(spinal::ESCTelemetry());

        thrust_pub_ = nh_.advertise<spinal::FourAxisCommand>(thrust_pub_name_, 1);
        thrust_debug_pub_ = nh_.advertise<spinal::FourAxisCommand>(thrust_debug_pub_name_, 1);
        power_on_pub_ = nh_.advertise<std_msgs::Empty>("/power_on_cmd", 1, true);

        if (enable_force_sensor_) {
            force_sensor_sub_ = nh_.subscribe(force_sensor_sub_name_, 1,
                                              &RotorResponse::forceSensorCallback, this,
                                              ros::TransportHints().tcpNoDelay());
        }
        if (enable_power_info_) {
            power_info_sub_ = nh_.subscribe(power_info_sub_name_, 1,
                                            &RotorResponse::powerInfoCallback, this,
                                            ros::TransportHints().tcpNoDelay());
        }
        if (enable_esc_telem_) {
            esc_telem_sub_ = nh_.subscribe(esc_telem_sub_name_, 1,
                                           &RotorResponse::escTelemCallback, this,
                                           ros::TransportHints().tcpNoDelay());
        }

        logSweepConfig();
    }

    void spin()
    {
        if (enable_force_sensor_ && enable_force_sensor_calib_) {
            calibrateForceSensor();
        }
        if (enable_power_on_cmd_) {
            std_msgs::Empty msg;
            power_on_pub_.publish(msg);
        }

        if (save_enable_) {
            openCsvFile();
        }

        ROS_INFO("Starting rotor response loop at %d Hz.", control_hz_);
        ros::Rate control_rate(control_hz_);
        start_time_ = std::chrono::high_resolution_clock::now();

        while (ros::ok()) {
            const double timestep = elapsedSeconds();
            const double dt = (step_count_ == 0) ? (1.0 / static_cast<double>(control_hz_))
                                                 : std::max(1e-6, timestep - last_timestep_);
            last_timestep_ = timestep;
            ++step_count_;

            target_thrust_ = sweepThrustCommand(timestep, dt);
            publishThrust(target_thrust_);
            writeCsvRow(timestep);

            if (step_count_ % std::max(1, control_hz_) == 0) {
                ROS_INFO("Time: %.2f s, target thrust[0]: %.3f N.", timestep, target_thrust_.front());
            }

            if (timestep >= duration_) {
                publishShutdownCommand();
                if (csv_file_.is_open()) {
                    csv_file_.close();
                    ROS_INFO("Rotor response data saved to %s.", csv_filename_.c_str());
                }
                ros::shutdown();
                return;
            }

            ros::spinOnce();
            control_rate.sleep();
        }
    }

private:
    ros::NodeHandle nh_;
    ros::NodeHandle nhp_;
    ros::Publisher thrust_pub_;
    ros::Publisher thrust_debug_pub_;
    ros::Publisher power_on_pub_;
    ros::Subscriber force_sensor_sub_;
    ros::Subscriber power_info_sub_;
    ros::Subscriber esc_telem_sub_;

    std::mutex data_mutex_;
    geometry_msgs::WrenchStamped force_msg_;
    ros::Time force_receive_time_;
    bool force_catch_ = false;
    float power_current_ = 0.0f;
    ros::Time power_receive_time_;
    std::array<spinal::ESCTelemetry, 4> esc_telemetry_;
    ros::Time esc_msg_time_;
    ros::Time esc_receive_time_;
    bool esc_catch_ = false;

    int control_hz_ = 100;
    int rotor_size_ = 4;
    int step_count_ = 0;
    double duration_ = 100.0;
    double default_thrust_ = 7.0;
    double thrust_range_ = 3.0;
    double thrust_min_ = 0.0;
    double thrust_max_ = 20.0;
    double disabled_thrust_ = 0.0;
    double shutdown_thrust_ = 0.0;
    double period_ = 1.0;
    double period_scale_ = 4.0;
    double last_timestep_ = 0.0;
    std::vector<bool> rotor_enable_;
    std::vector<float> target_thrust_;

    bool enable_force_sensor_ = true;
    bool enable_force_sensor_calib_ = true;
    bool enable_power_info_ = true;
    bool enable_power_on_cmd_ = true;
    bool enable_esc_telem_ = true;
    bool save_enable_ = true;

    std::string save_path_;
    std::string csv_filename_;
    std::string force_sensor_calib_service_;
    std::string thrust_pub_name_;
    std::string thrust_debug_pub_name_;
    std::string force_sensor_sub_name_;
    std::string power_info_sub_name_;
    std::string esc_telem_sub_name_;
    std::ofstream csv_file_;

    std::chrono::high_resolution_clock::time_point start_time_;

    double elapsedSeconds() const
    {
        const uint64_t elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now() - start_time_).count();
        return static_cast<double>(elapsed_ns) / 1e9;
    }

    std::vector<float> sweepThrustCommand(double t, double dt)
    {
        (void)dt;
        std::vector<float> thrust(rotor_size_, static_cast<float>(disabled_thrust_));

        const double pad = 1.0;
        const double active_dur = duration_ - 2.0 * pad;
        double delta = 0.0;
        if (t >= pad && t <= duration_ - pad && active_dur > 0.0) {
            const double t_a = t - pad;
            const double alpha = std::min(1.0, std::max(0.0, t_a / active_dur));
            const double sweep_progress = 1.0 - std::abs(2.0 * alpha - 1.0);
            const double phi = sweepPhase(t_a, active_dur, sweep_progress);
            delta = thrust_range_ * sweep_progress * std::sin(phi);
        }

        for (int i = 0; i < rotor_size_; ++i) {
            if (!rotor_enable_[i]) {
                thrust[i] = static_cast<float>(disabled_thrust_);
                continue;
            }

            const double raw_target = default_thrust_ + delta;
            const double positive_target = std::abs(raw_target);
            thrust[i] = static_cast<float>(clamp(positive_target, thrust_min_, thrust_max_));
        }
        return thrust;
    }

    double sweepPhase(double t_active, double active_dur, double sweep_progress) const
    {
        constexpr double kTwoPi = 6.28318530717958647692;
        const double fast_period = std::max(1e-6, std::min(period_, period_ * period_scale_));
        const double slow_period = std::max(fast_period, std::max(period_, period_ * period_scale_));
        const double delta_period = slow_period - fast_period;
        const double cur_period = fast_period + delta_period * sweep_progress;
        const double half_dur = 0.5 * active_dur;

        if (delta_period < 1e-9 || half_dur <= 1e-9) {
            return kTwoPi * t_active / fast_period;
        }

        const double slope = delta_period / half_dur;
        const double phi_half = kTwoPi / slope * std::log(slow_period / fast_period);
        if (t_active <= half_dur) {
            return kTwoPi / slope * std::log(cur_period / fast_period);
        }
        return phi_half + kTwoPi / slope * std::log(slow_period / cur_period);
    }

    void publishThrust(const std::vector<float>& thrust)
    {
        spinal::FourAxisCommand msg;
        msg.angles[0] = 0.0f;
        msg.angles[1] = 0.0f;
        msg.angles[2] = 0.0f;
        msg.body_rates[0] = 0.0f;
        msg.body_rates[1] = 0.0f;
        msg.body_rates[2] = 0.0f;
        msg.base_thrust = thrust;
        thrust_pub_.publish(msg);
        thrust_debug_pub_.publish(msg);
    }

    void publishShutdownCommand()
    {
        std::vector<float> thrust(rotor_size_, static_cast<float>(shutdown_thrust_));
        for (int i = 0; i < 10 && ros::ok(); ++i) {
            publishThrust(thrust);
            ros::Duration(0.01).sleep();
        }
    }

    void forceSensorCallback(const geometry_msgs::WrenchStampedConstPtr& msg)
    {
        std::lock_guard<std::mutex> lk(data_mutex_);
        force_msg_ = *msg;
        force_receive_time_ = ros::Time::now();
        force_catch_ = true;
    }

    void powerInfoCallback(const takasako_sps::PowerInfoConstPtr& msg)
    {
        std::lock_guard<std::mutex> lk(data_mutex_);
        power_current_ = msg->currency;
        power_receive_time_ = ros::Time::now();
    }

    void escTelemCallback(const spinal::ESCTelemetryArrayConstPtr& msg)
    {
        std::lock_guard<std::mutex> lk(data_mutex_);
        esc_telemetry_[0] = msg->esc_telemetry_1;
        esc_telemetry_[1] = msg->esc_telemetry_2;
        esc_telemetry_[2] = msg->esc_telemetry_3;
        esc_telemetry_[3] = msg->esc_telemetry_4;
        esc_msg_time_ = msg->stamp;
        esc_receive_time_ = ros::Time::now();
        esc_catch_ = true;
    }

    void calibrateForceSensor()
    {
        ros::ServiceClient calib_client = nh_.serviceClient<std_srvs::Empty>(force_sensor_calib_service_);
        std_srvs::Empty srv;
        if (calib_client.call(srv)) {
            ROS_INFO("Done force sensor calibration via %s.", force_sensor_calib_service_.c_str());
        } else {
            ROS_WARN("Failed to call force sensor calibration service %s.", force_sensor_calib_service_.c_str());
        }
    }

    void openCsvFile()
    {
        std::time_t now_c = std::time(nullptr);
        std::tm tm = *std::localtime(&now_c);
        std::ostringstream ts;
        ts << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
        csv_filename_ = save_path_ + "/rotor_" + ts.str() + ".csv";
        csv_file_.open(csv_filename_, std::ios::out);
        if (!csv_file_.is_open()) {
            ROS_ERROR("Failed to open rotor response csv: %s.", csv_filename_.c_str());
            save_enable_ = false;
            return;
        }
        csv_file_ << std::fixed << std::setprecision(6);
        writeCsvHeader();
        ROS_INFO("Rotor response csv opened: %s.", csv_filename_.c_str());
    }

    void writeCsvHeader()
    {
        if (!csv_file_.is_open()) return;

        csv_file_ << "timestep"
                  << ",esc_msg_time_ms"
                  << ",esc_receive_time_ms"
                  << ",force_msg_time_ms"
                  << ",force_receive_time_ms"
                  << ",force_x"
                  << ",force_y"
                  << ",force_z"
                  << ",force_norm"
                  << ",torque_x"
                  << ",torque_y"
                  << ",torque_z"
                  << ",power_current";

        for (int i = 0; i < rotor_size_; ++i) {
            if (!rotor_enable_[i]) continue;
            csv_file_ << ",rotor" << i << "_target_thrust"
                      << ",rotor" << i << "_rpm"
                      << ",rotor" << i << "_esc_current"
                      << ",rotor" << i << "_esc_voltage"
                      << ",rotor" << i << "_temperature"
                      << ",rotor" << i << "_crc_error";
        }
        csv_file_ << "\n";
    }

    void writeCsvRow(double timestep)
    {
        if (!save_enable_ || !csv_file_.is_open()) return;

        geometry_msgs::WrenchStamped force_msg;
        ros::Time force_receive_time;
        ros::Time esc_msg_time;
        ros::Time esc_receive_time;
        float power_current = 0.0f;
        std::array<spinal::ESCTelemetry, 4> esc_telemetry;
        {
            std::lock_guard<std::mutex> lk(data_mutex_);
            force_msg = force_msg_;
            force_receive_time = force_receive_time_;
            esc_msg_time = esc_msg_time_;
            esc_receive_time = esc_receive_time_;
            power_current = power_current_;
            esc_telemetry = esc_telemetry_;
        }

        const double fx = force_msg.wrench.force.x;
        const double fy = force_msg.wrench.force.y;
        const double fz = force_msg.wrench.force.z;
        const double force_norm = std::sqrt(fx * fx + fy * fy + fz * fz);

        csv_file_ << timestep
                  << "," << esc_msg_time.toNSec() * 1e-6
                  << "," << esc_receive_time.toNSec() * 1e-6
                  << "," << force_msg.header.stamp.toNSec() * 1e-6
                  << "," << force_receive_time.toNSec() * 1e-6
                  << "," << fx
                  << "," << fy
                  << "," << fz
                  << "," << force_norm
                  << "," << force_msg.wrench.torque.x
                  << "," << force_msg.wrench.torque.y
                  << "," << force_msg.wrench.torque.z
                  << "," << power_current;

        for (int i = 0; i < rotor_size_; ++i) {
            if (!rotor_enable_[i]) continue;
            const spinal::ESCTelemetry telemetry = escTelemetryAt(esc_telemetry, i);
            csv_file_ << "," << target_thrust_[i]
                      << "," << telemetry.rpm
                      << "," << static_cast<double>(telemetry.current) / 100.0
                      << "," << static_cast<double>(telemetry.voltage) / 100.0
                      << "," << static_cast<int>(telemetry.temperature)
                      << "," << static_cast<int>(telemetry.crc_error);
        }
        csv_file_ << "\n";
    }

    spinal::ESCTelemetry escTelemetryAt(const std::array<spinal::ESCTelemetry, 4>& telemetry, int index) const
    {
        if (index >= 0 && index < static_cast<int>(telemetry.size())) {
            return telemetry[index];
        }
        return spinal::ESCTelemetry();
    }

    void logSweepConfig() const
    {
        const double active_dur = std::max(0.0, duration_ - 2.0);
        const double fast_period = std::max(1e-6, std::min(period_, period_ * period_scale_));
        const double slow_period = std::max(fast_period, std::max(period_, period_ * period_scale_));
        const double cycles = estimateSweepCycles(active_dur, fast_period, slow_period);

        ROS_INFO("Rotor response command topic: %s.", thrust_pub_name_.c_str());
        ROS_INFO("Enabled rotors: %s.", enabledRotorString().c_str());
        ROS_INFO("Sweep total duration %.3f s, active duration %.3f s, period %.3f -> %.3f -> %.3f s, estimated cycles %.2f.",
                 duration_, active_dur, fast_period, slow_period, fast_period, cycles);
        ROS_INFO("Thrust target: abs(default %.3f N + delta), delta range +/- %.3f N, clamp [%.3f, %.3f] N.",
                 default_thrust_, thrust_range_, thrust_min_, thrust_max_);
    }

    double estimateSweepCycles(double active_dur, double fast_period, double slow_period) const
    {
        if (active_dur <= 0.0) return 0.0;
        if (std::abs(slow_period - fast_period) < 1e-9) {
            return active_dur / fast_period;
        }
        return active_dur / (slow_period - fast_period) * std::log(slow_period / fast_period);
    }

    std::string enabledRotorString() const
    {
        std::ostringstream ss;
        bool first = true;
        for (int i = 0; i < rotor_size_; ++i) {
            if (!rotor_enable_[i]) continue;
            if (!first) ss << ",";
            ss << i;
            first = false;
        }
        if (first) return "none";
        return ss.str();
    }

    double clamp(double value, double min_value, double max_value) const
    {
        return std::max(min_value, std::min(max_value, value));
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "rotor_response");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    RotorResponse rotor_response(nh, nh_private);
    rotor_response.spin();
    return 0;
}
