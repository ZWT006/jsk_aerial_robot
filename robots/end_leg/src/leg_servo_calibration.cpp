#include <algorithm>
#include <cmath>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64.h>

namespace
{
constexpr double kPi = 3.14159265358979323846;

struct Feedback
{
  double position = 0.0;
  double velocity = 0.0;
  double effort = 0.0;
};

struct Command
{
  double target = 0.0;
  double effort = 0.0;
};

struct Sample
{
  double time;
  double target;
  double position;
  double velocity;
  double effort;
};
}  // namespace

class LegServoCalibration
{
public:
  LegServoCalibration()
    : nh_(), pnh_("~")
  {
    pnh_.param("simulation", simulation_, true);
    pnh_.param("effort_ctrl", effort_ctrl_, false);
    pnh_.param("auto_swing", auto_swing_, false);
    pnh_.param("joint_name", joint_name_, std::string("joint_pitch_servo"));
    pnh_.param("state_topic", state_topic_, std::string("joint_states"));
    pnh_.param("command_topic", command_topic_, std::string("joints_ctrl"));
    pnh_.param("effort_command_topic", effort_command_topic_,
               std::string("servo_controller/joints/controller1/simulation/command"));

    pnh_.param("control_freq", control_freq_, 200.0);
    pnh_.param("duration", duration_, 200.0);
    pnh_.param("decimation", decimation_, 1);
    pnh_.param("feedback_delay_step", feedback_delay_step_, 0);
    pnh_.param("command_delay_step", command_delay_step_, 0);
    pnh_.param("default_position", default_position_, 0.0);
    pnh_.param("joint_range", joint_range_, 0.5);
    pnh_.param("cmd_period", cmd_period_, 0.04);
    pnh_.param("period_scale", period_scale_, 50.0);
    pnh_.param("sweep_mode", sweep_mode_, true);
    pnh_.param("step_mode", step_mode_, false);
    pnh_.param("step_height", step_height_, 0.15);

    pnh_.param("kp", kp_, 4.8309);
    pnh_.param("kd", kd_, 0.30755);
    pnh_.param("effort_limit", effort_limit_, 6.6);
    pnh_.param("state_timeout", state_timeout_, 0.5);
    pnh_.param("lower_limit", lower_limit_, -kPi / 2.0);
    pnh_.param("upper_limit", upper_limit_, kPi / 2.0);
    pnh_.param("enable_save", enable_save_, true);
    pnh_.param("save_path", save_path_, std::string("/tmp"));

    joint_range_ = std::abs(joint_range_);
    step_height_ = std::abs(step_height_);
    effort_limit_ = std::abs(effort_limit_);

    if (control_freq_ <= 0.0 || duration_ <= 0.0 || cmd_period_ <= 0.0 ||
        period_scale_ <= 0.0 || lower_limit_ >= upper_limit_ ||
        decimation_ <= 0 || feedback_delay_step_ < 0 ||
        command_delay_step_ < 0)
      {
        ROS_FATAL("Invalid calibration timing, delay-step, or joint-limit parameter");
        configuration_valid_ = false;
      }
    if (auto_swing_ && effort_ctrl_ && !simulation_)
      {
        ROS_FATAL("effort_ctrl is simulation-only; hardware uses position commands");
        configuration_valid_ = false;
      }
    if (auto_swing_ && step_mode_ && sweep_mode_)
      ROS_WARN("Both step_mode and sweep_mode are true; step_mode has priority");

    state_sub_ = nh_.subscribe(state_topic_, 1,
                               &LegServoCalibration::stateCallback, this,
                               ros::TransportHints().tcpNoDelay());
    if (auto_swing_)
      {
        if (effort_ctrl_)
          effort_pub_ = nh_.advertise<std_msgs::Float64>(effort_command_topic_, 1);
        else
          position_pub_ = nh_.advertise<sensor_msgs::JointState>(command_topic_, 1);
      }
    else
      {
        target_sub_ = nh_.subscribe(command_topic_, 1,
                                    &LegServoCalibration::targetCallback, this,
                                    ros::TransportHints().tcpNoDelay());
      }

    ROS_INFO("end_leg calibration: joint=%s, mode=%s, control=%s, signal=%s",
             joint_name_.c_str(), auto_swing_ ? "auto_swing" : "data_only",
             effort_ctrl_ ? "effort" : "position",
             auto_swing_ ? signalName().c_str() : "none");
    ROS_INFO("feedback=%s, target_topic=%s",
             nh_.resolveName(state_topic_).c_str(),
             nh_.resolveName(command_topic_).c_str());
    ROS_INFO("delay: feedback=%d steps (%.3f ms), command=%d steps (%.3f ms), "
             "control=%.1f Hz, target=%.1f Hz (decimation=%d)",
             feedback_delay_step_, 1000.0 * feedback_delay_step_ / control_freq_,
             command_delay_step_, 1000.0 * command_delay_step_ / control_freq_,
             control_freq_, control_freq_ / decimation_, decimation_);
  }

  bool run()
  {
    if (!configuration_valid_) return false;

    ros::WallRate rate(control_freq_);
    while (ros::ok() && !feedback_received_ && !fatal_error_)
      {
        ros::spinOnce();
        ROS_INFO_THROTTLE(2.0, "Waiting for complete JointState feedback for '%s'",
                          joint_name_.c_str());
        rate.sleep();
      }
    if (!ros::ok() || fatal_error_) return false;

    initializeDelayBuffers();
    samples_.reserve(static_cast<std::size_t>(std::ceil(duration_ * control_freq_)) + 1U);
    start_time_ = ros::WallTime::now();
    ROS_INFO("Feedback is valid; starting %.3f s calibration", duration_);

    while (ros::ok())
      {
        ros::spinOnce();
        if (fatal_error_)
          {
            stopCommand(true);
            saveData();
            return false;
          }

        const ros::WallTime now = ros::WallTime::now();
        const double elapsed = (now - start_time_).toSec();
        if (state_timeout_ > 0.0 &&
            (now - last_feedback_time_).toSec() > state_timeout_)
          {
            ROS_FATAL("JointState feedback timed out for %.3f s",
                      (now - last_feedback_time_).toSec());
            stopCommand(true);
            saveData();
            return false;
          }
        if (elapsed >= duration_) break;

        const Feedback feedback_after_delay = delayedFeedback();
        const double target_before_command_delay = auto_swing_ ?
          publishDelayedCommand(elapsed, feedback_after_delay) :
          observedTarget(feedback_after_delay);

        samples_.push_back({elapsed, target_before_command_delay,
                            feedback_after_delay.position,
                            feedback_after_delay.velocity, feedback_after_delay.effort});
        ROS_INFO_THROTTLE(5.0,
                          "t=%.2f target=%+.4f position=%+.4f velocity=%+.4f effort=%+.4f",
                          elapsed, target_before_command_delay,
                          feedback_after_delay.position, feedback_after_delay.velocity,
                          feedback_after_delay.effort);
        rate.sleep();
      }

    stopCommand(false);
    return saveData();
  }

private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber state_sub_;
  ros::Subscriber target_sub_;
  ros::Publisher position_pub_;
  ros::Publisher effort_pub_;

  bool simulation_ = true;
  bool effort_ctrl_ = false;
  bool auto_swing_ = false;
  bool sweep_mode_ = true;
  bool step_mode_ = false;
  bool enable_save_ = true;
  bool configuration_valid_ = true;
  bool feedback_received_ = false;
  bool fatal_error_ = false;
  bool target_received_ = false;

  std::string joint_name_;
  std::string state_topic_;
  std::string command_topic_;
  std::string effort_command_topic_;
  std::string save_path_;

  double control_freq_ = 200.0;
  double duration_ = 200.0;
  int decimation_ = 1;
  int feedback_delay_step_ = 0;
  int command_delay_step_ = 0;
  double default_position_ = 0.0;
  double joint_range_ = 0.5;
  double cmd_period_ = 0.04;
  double period_scale_ = 50.0;
  double step_height_ = 0.15;
  double kp_ = 4.8309;
  double kd_ = 0.30755;
  double effort_limit_ = 6.6;
  double state_timeout_ = 0.5;
  double lower_limit_ = -kPi / 2.0;
  double upper_limit_ = kPi / 2.0;

  Feedback raw_feedback_;
  double external_target_ = 0.0;
  ros::WallTime start_time_;
  ros::WallTime last_feedback_time_;
  std::deque<Feedback> feedback_buffer_;
  std::deque<Command> command_buffer_;
  std::vector<Sample> samples_;
  double generated_target_ = 0.0;
  int target_update_countdown_ = 0;

  static double clamp(double value, double lower, double upper)
  {
    return std::max(lower, std::min(upper, value));
  }

  void initializeDelayBuffers()
  {
    feedback_buffer_.assign(static_cast<std::size_t>(feedback_delay_step_),
                            raw_feedback_);
    command_buffer_.assign(static_cast<std::size_t>(command_delay_step_),
                           Command{default_position_, 0.0});
    generated_target_ = default_position_;
    target_update_countdown_ = 0;
  }

  Feedback delayedFeedback()
  {
    feedback_buffer_.push_back(raw_feedback_);
    const Feedback feedback = feedback_buffer_.front();
    feedback_buffer_.pop_front();
    return feedback;
  }

  double observedTarget(const Feedback& feedback) const
  {
    return target_received_ ? external_target_ : feedback.position;
  }

  double publishDelayedCommand(double elapsed, const Feedback& feedback)
  {
    Command pending;
    if (target_update_countdown_ == 0)
      {
        generated_target_ = clamp(default_position_ + relativeCommand(elapsed),
                                  lower_limit_, upper_limit_);
        target_update_countdown_ = decimation_;
      }
    --target_update_countdown_;
    pending.target = generated_target_;
    if (effort_ctrl_)
      pending.effort = clamp(kp_ * (pending.target - feedback.position) -
                             kd_ * feedback.velocity,
                             -effort_limit_, effort_limit_);

    command_buffer_.push_back(pending);
    const Command command = command_buffer_.front();
    command_buffer_.pop_front();

    if (effort_ctrl_)
      {
        std_msgs::Float64 msg;
        msg.data = command.effort;
        effort_pub_.publish(msg);
      }
    else
      {
        publishPosition(command.target);
      }
    // The dataset target is intentionally captured before command-delay injection.
    return pending.target;
  }

  std::string signalName() const
  {
    if (step_mode_) return "step";
    if (sweep_mode_) return "sweep";
    return "default";
  }

  double relativeCommand(double elapsed) const
  {
    if (step_mode_)
      return stepWave(elapsed, cmd_period_, step_height_, joint_range_);
    if (sweep_mode_)
      return sweepSinTriangleWave(elapsed, cmd_period_,
                                  period_scale_ * cmd_period_,
                                  duration_, joint_range_);
    return multipleWave(elapsed, cmd_period_, joint_range_);
  }

  static double sinWave(double time, double period, double amplitude)
  {
    return amplitude * std::sin(2.0 * kPi * std::fmod(time, period) / period);
  }

  static double triangleWave(double time, double period, double amplitude)
  {
    double phase = std::fmod(time + period / 4.0, period);
    if (phase < 0.0) phase += period;
    const double half_period = period / 2.0;
    const double slope = 2.0 * amplitude / half_period;
    if (phase < half_period) return -amplitude + slope * phase;
    return amplitude - slope * (phase - half_period);
  }

  static double squareWave(double time, double period, double amplitude)
  {
    return std::fmod(time, period) < period / 2.0 ? amplitude : -amplitude;
  }

  static double trapezoidWave(double time, double period, double amplitude)
  {
    return clamp(triangleWave(time, period, 2.0 * amplitude),
                 -amplitude, amplitude);
  }

  static double sinTriangleWave(double time, double period, double amplitude)
  {
    const double envelope =
      std::abs(triangleWave(time, 32.0 * period, amplitude));
    return envelope * std::sin(2.0 * kPi * std::fmod(time, period) / period);
  }

  static double stepWave(double time, double period,
                         double step_height, double amplitude)
  {
    if (step_height <= 0.0 || amplitude <= 0.0) return 0.0;
    const long half_steps = std::max(1L, static_cast<long>(
      std::ceil(amplitude / step_height - 1e-12)));
    const long cycle = 4L * half_steps;
    long phase = static_cast<long>(std::floor(time / period)) % cycle;
    if (phase < 0) phase += cycle;

    long level = 0;
    if (phase <= half_steps)
      level = phase;
    else if (phase <= 3L * half_steps)
      level = 2L * half_steps - phase;
    else
      level = phase - cycle;
    return clamp(level * step_height, -amplitude, amplitude);
  }

  static double sweepSinTriangleWave(double time, double period_min,
                                     double period_max, double duration,
                                     double amplitude)
  {
    constexpr double padding = 1.0;
    const double active_duration = duration - 2.0 * padding;
    if (time < padding || time > duration - padding || active_duration <= 0.0)
      return 0.0;

    const double active_time = time - padding;
    const double alpha = active_time / active_duration;
    const double progress = 1.0 - std::abs(2.0 * alpha - 1.0);
    const double fast_period = std::max(1e-6, std::min(period_min, period_max));
    const double slow_period = std::max(fast_period, std::max(period_min, period_max));
    const double period_delta = slow_period - fast_period;
    const double half_duration = active_duration / 2.0;
    const double current_period = fast_period + period_delta * progress;

    double phase = 2.0 * kPi * active_time / fast_period;
    if (period_delta >= 1e-9 && half_duration > 1e-9)
      {
        const double slope = period_delta / half_duration;
        const double half_phase =
          2.0 * kPi / slope * std::log(slow_period / fast_period);
        if (active_time <= half_duration)
          phase = 2.0 * kPi / slope * std::log(current_period / fast_period);
        else
          phase = half_phase +
                  2.0 * kPi / slope * std::log(slow_period / current_period);
      }

    return amplitude * std::max(0.0, progress) * std::sin(phase);
  }

  static double multipleWave(double time, double period, double amplitude)
  {
    const double phase = std::fmod(time, 32.0 * period);
    if (phase > 16.0 * period)
      return sinTriangleWave(time, period, amplitude);
    if (phase > 12.0 * period)
      return trapezoidWave(time, period, amplitude);
    if (phase > 8.0 * period)
      return squareWave(time, period, amplitude);
    if (phase > 4.0 * period)
      return triangleWave(time, period, amplitude);
    return sinWave(time, period, amplitude);
  }

  void publishPosition(double position)
  {
    sensor_msgs::JointState msg;
    msg.header.stamp = ros::Time::now();
    msg.name.push_back(joint_name_);
    msg.position.push_back(position);
    position_pub_.publish(msg);
  }

  void stopCommand(bool error_stop)
  {
    if (!auto_swing_) return;
    if (effort_ctrl_)
      {
        std_msgs::Float64 msg;
        msg.data = 0.0;
        effort_pub_.publish(msg);
      }
    else if (feedback_received_)
      {
        publishPosition(error_stop ? raw_feedback_.position : default_position_);
      }
  }

  void targetCallback(const sensor_msgs::JointState::ConstPtr& msg)
  {
    std::size_t index = 0;
    if (!msg->name.empty())
      {
        const auto joint = std::find(msg->name.begin(), msg->name.end(), joint_name_);
        if (joint == msg->name.end())
          {
            ROS_WARN_THROTTLE(2.0, "Target command does not contain joint '%s'",
                              joint_name_.c_str());
            return;
          }
        index = static_cast<std::size_t>(std::distance(msg->name.begin(), joint));
      }
    else if (msg->position.size() != 1)
      {
        ROS_WARN_THROTTLE(2.0, "Cannot map unnamed multi-joint target command");
        return;
      }

    if (index >= msg->position.size() || !std::isfinite(msg->position[index]))
      {
        ROS_WARN_THROTTLE(2.0, "Invalid target position for joint '%s'",
                          joint_name_.c_str());
        return;
      }
    external_target_ = msg->position[index];
    target_received_ = true;
  }

  void stateCallback(const sensor_msgs::JointState::ConstPtr& msg)
  {
    if (fatal_error_) return;

    const auto joint = std::find(msg->name.begin(), msg->name.end(), joint_name_);
    if (joint == msg->name.end())
      {
        ROS_FATAL("Joint '%s' is absent from %s",
                  joint_name_.c_str(), nh_.resolveName(state_topic_).c_str());
        fatal_error_ = true;
        return;
      }

    const std::size_t index =
      static_cast<std::size_t>(std::distance(msg->name.begin(), joint));
    if (index >= msg->position.size())
      {
        ROS_FATAL("JointState.position is missing for joint '%s'", joint_name_.c_str());
        fatal_error_ = true;
        return;
      }
    if (index >= msg->velocity.size())
      {
        ROS_FATAL("JointState.velocity is missing for joint '%s'; finite-difference fallback is disabled",
                  joint_name_.c_str());
        fatal_error_ = true;
        return;
      }
    if (index >= msg->effort.size())
      {
        ROS_FATAL("JointState.effort is missing for joint '%s'", joint_name_.c_str());
        fatal_error_ = true;
        return;
      }

    const Feedback feedback{msg->position[index], msg->velocity[index],
                            msg->effort[index]};
    if (!std::isfinite(feedback.position) || !std::isfinite(feedback.velocity) ||
        !std::isfinite(feedback.effort))
      {
        ROS_FATAL("Non-finite JointState feedback for joint '%s'", joint_name_.c_str());
        fatal_error_ = true;
        return;
      }

    raw_feedback_ = feedback;
    if (!target_received_) external_target_ = feedback.position;
    last_feedback_time_ = ros::WallTime::now();
    feedback_received_ = true;
  }

  bool saveData() const
  {
    if (!enable_save_)
      {
        ROS_INFO("Data saving is disabled; collected %zu samples", samples_.size());
        return true;
      }

    std::time_t now = std::time(nullptr);
    std::tm local_time;
    localtime_r(&now, &local_time);
    std::ostringstream timestamp;
    timestamp << std::put_time(&local_time, "%Y-%m-%d_%H-%M-%S");

    std::string directory = save_path_.empty() ? "." : save_path_;
    if (directory.back() != '/') directory.push_back('/');
    const std::string mode = auto_swing_ ?
      (effort_ctrl_ ? "effort_" : "position_") : "data_";
    const std::string filename = directory + "end_leg_" + mode +
                                 timestamp.str() + ".csv";

    std::ofstream file(filename);
    if (!file.is_open())
      {
        ROS_ERROR("Failed to open calibration output: %s", filename.c_str());
        return false;
      }

    file << "time_s,target_rad,position_rad,velocity_rad_s,effort_Nm\n";
    file << std::fixed << std::setprecision(9);
    for (const Sample& sample : samples_)
      file << sample.time << ',' << sample.target << ',' << sample.position << ','
           << sample.velocity << ',' << sample.effort << '\n';

    if (!file.good())
      {
        ROS_ERROR("Failed while writing calibration output: %s", filename.c_str());
        return false;
      }

    ROS_INFO("Saved %zu samples to %s", samples_.size(), filename.c_str());
    return true;
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "leg_servo_calibration");
  LegServoCalibration calibration;
  return calibration.run() ? 0 : 1;
}
