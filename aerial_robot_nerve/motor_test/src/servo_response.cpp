#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Float64.h>
#include <xmlrpcpp/XmlRpcValue.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
constexpr double kPi = 3.14159265358979323846;

std::string lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

std::string trimSlash(std::string s)
{
  while (!s.empty() && s.front() == '/') s.erase(s.begin());
  while (!s.empty() && s.back() == '/') s.pop_back();
  return s;
}

double clamp(double value, double lo, double hi)
{
  return std::min(std::max(value, lo), hi);
}

double xmlDouble(XmlRpc::XmlRpcValue& value, double fallback)
{
  if (value.getType() == XmlRpc::XmlRpcValue::TypeInt)
    return static_cast<int>(value);
  if (value.getType() == XmlRpc::XmlRpcValue::TypeDouble)
    return static_cast<double>(value);
  if (value.getType() == XmlRpc::XmlRpcValue::TypeBoolean)
    return static_cast<bool>(value) ? 1.0 : 0.0;
  if (value.getType() == XmlRpc::XmlRpcValue::TypeString)
    {
      try
        {
          return std::stod(static_cast<std::string>(value));
        }
      catch (const std::exception&)
        {
        }
    }
  return fallback;
}

bool xmlBool(XmlRpc::XmlRpcValue& value, bool fallback)
{
  if (value.getType() == XmlRpc::XmlRpcValue::TypeBoolean)
    return static_cast<bool>(value);
  if (value.getType() == XmlRpc::XmlRpcValue::TypeInt)
    return static_cast<int>(value) != 0;
  if (value.getType() == XmlRpc::XmlRpcValue::TypeDouble)
    return static_cast<double>(value) != 0.0;
  if (value.getType() == XmlRpc::XmlRpcValue::TypeString)
    {
      const std::string s = lower(static_cast<std::string>(value));
      if (s == "true" || s == "1") return true;
      if (s == "false" || s == "0") return false;
    }
  return fallback;
}

std::string xmlString(XmlRpc::XmlRpcValue& value)
{
  if (value.getType() == XmlRpc::XmlRpcValue::TypeString)
    return static_cast<std::string>(value);
  return std::string();
}

std::vector<std::string> stringParamList(ros::NodeHandle& nh,
                                         const std::vector<std::string>& keys,
                                         bool* found = nullptr)
{
  if (found) *found = false;
  for (const auto& key : keys)
    {
      XmlRpc::XmlRpcValue value;
      if (!nh.getParam(key, value)) continue;
      if (found) *found = true;

      std::vector<std::string> out;
      if (value.getType() == XmlRpc::XmlRpcValue::TypeString)
        {
          out.push_back(static_cast<std::string>(value));
          return out;
        }
      if (value.getType() != XmlRpc::XmlRpcValue::TypeArray) return out;

      for (int i = 0; i < value.size(); ++i)
        {
          std::string s = xmlString(value[i]);
          if (!s.empty()) out.push_back(s);
        }
      return out;
    }
  return {};
}

std::vector<double> doubleParamList(ros::NodeHandle& nh,
                                    const std::vector<std::string>& keys,
                                    size_t count, double fallback,
                                    bool* found = nullptr)
{
  if (found) *found = false;
  for (const auto& key : keys)
    {
      XmlRpc::XmlRpcValue value;
      if (!nh.getParam(key, value)) continue;
      if (found) *found = true;

      std::vector<double> out;
      if (value.getType() != XmlRpc::XmlRpcValue::TypeArray)
        {
          out.assign(count, xmlDouble(value, fallback));
          return out;
        }
      for (int i = 0; i < value.size(); ++i)
        out.push_back(xmlDouble(value[i], fallback));
      out.resize(count, fallback);
      return out;
    }
  return std::vector<double>(count, fallback);
}

std::vector<bool> boolParamList(ros::NodeHandle& nh,
                                const std::vector<std::string>& keys,
                                size_t count, bool fallback,
                                bool* found = nullptr)
{
  if (found) *found = false;
  for (const auto& key : keys)
    {
      XmlRpc::XmlRpcValue value;
      if (!nh.getParam(key, value)) continue;
      if (found) *found = true;

      std::vector<bool> out;
      if (value.getType() != XmlRpc::XmlRpcValue::TypeArray)
        {
          out.assign(count, xmlBool(value, fallback));
          return out;
        }
      for (int i = 0; i < value.size(); ++i)
        out.push_back(xmlBool(value[i], fallback));
      out.resize(count, fallback);
      return out;
    }
  return std::vector<bool>(count, fallback);
}

std::string stringParam(ros::NodeHandle& nh, const std::vector<std::string>& keys,
                        const std::string& fallback)
{
  for (const auto& key : keys)
    {
      std::string value;
      if (nh.getParam(key, value)) return value;
    }
  return fallback;
}

double doubleParam(ros::NodeHandle& nh, const std::vector<std::string>& keys,
                   double fallback)
{
  for (const auto& key : keys)
    {
      XmlRpc::XmlRpcValue value;
      if (nh.getParam(key, value)) return xmlDouble(value, fallback);
    }
  return fallback;
}

bool controllerIndex(const std::string& key, int& index)
{
  const std::string prefix = "controller";
  if (key.compare(0, prefix.size(), prefix) != 0) return false;
  if (key.size() == prefix.size()) return false;

  index = 0;
  for (size_t i = prefix.size(); i < key.size(); ++i)
    {
      if (!std::isdigit(static_cast<unsigned char>(key[i]))) return false;
      index = index * 10 + key[i] - '0';
    }
  return true;
}

double xmlMemberDouble(XmlRpc::XmlRpcValue& value, const std::string& key,
                       double fallback)
{
  if (!value.hasMember(key)) return fallback;
  return xmlDouble(value[key], fallback);
}
}  // namespace

class ServoResponse
{
public:
  ServoResponse(ros::NodeHandle nh, ros::NodeHandle nhp)
    : nh_(nh), nhp_(nhp)
  {
    loadParams();
    if (!valid_) return;

    pos_.assign(axis_count_, 0.0);
    vel_.assign(axis_count_, 0.0);
    seen_.assign(axis_count_, false);
    state_names_.resize(axis_count_);
    target_.assign(axis_count_, 0.0);

    rebuildNameIndex();
    state_sub_ = nh_.subscribe(topic(state_topic_), 1, &ServoResponse::stateCallback,
                               this, ros::TransportHints().tcpNoDelay());

    if (!effort_ctrl_)
      command_pub_ = nh_.advertise<sensor_msgs::JointState>(topic(command_topic_), 1);

    if (publish_debug_ && !debug_topic_.empty())
      debug_pub_ = nh_.advertise<sensor_msgs::JointState>(topic(debug_topic_), 1);

    if (effort_ctrl_) valid_ = setupEffortPublishers();

    ROS_INFO("servo_response: ns='%s' state='%s' command='%s' axes=%zu signal=%s",
             robot_ns_.c_str(), topic(state_topic_).c_str(), topic(command_topic_).c_str(),
             axis_count_, signal_type_.c_str());
  }

  bool run()
  {
    if (!valid_) return false;
    if (!waitForState()) return false;

    ros::WallRate rate(control_freq_);
    ros::WallTime start = ros::WallTime::now();
    int last_report = -1;

    while (ros::ok())
      {
        const double t = (ros::WallTime::now() - start).toSec();
        publishCommand(t);

        const int sec = static_cast<int>(std::floor(t));
        if (sec != last_report)
          {
            ROS_INFO("servo_response: %.2f s, samples=%zu", t, rows_.size());
            last_report = sec;
          }

        if (duration_ > 0.0 && t >= duration_)
          {
            if (enable_save_ && !saveCsv()) return false;
            ROS_INFO("servo_response: finished after %.2f s", t);
            return true;
          }

        ros::spinOnce();
        rate.sleep();
      }
    return true;
  }

private:
  struct State
  {
    std::vector<double> pos;
    std::vector<double> vel;
    std::vector<std::string> names;
    ros::Time stamp;
    ros::Time receive_stamp;
  };

  ros::NodeHandle nh_;
  ros::NodeHandle nhp_;
  ros::Subscriber state_sub_;
  ros::Publisher command_pub_;
  ros::Publisher debug_pub_;
  std::vector<ros::Publisher> effort_pubs_;

  std::mutex mutex_;
  std::vector<double> pos_;
  std::vector<double> vel_;
  std::vector<bool> seen_;
  std::vector<std::string> state_names_;
  std::unordered_map<std::string, size_t> name_index_;
  ros::Time state_stamp_;
  ros::Time receive_stamp_;

  bool valid_ = true;
  std::string robot_ns_;
  std::string state_topic_ = "joint_states";
  std::string command_topic_ = "gimbals_ctrl";
  std::string debug_topic_;
  std::string servo_group_ = "gimbals";
  std::vector<std::string> axis_names_;
  std::vector<bool> enabled_;
  std::vector<double> default_pos_;
  std::vector<double> kp_;
  std::vector<double> kd_;
  std::vector<double> effort_limit_;
  size_t axis_count_ = 4;

  double control_freq_ = 200.0;
  double duration_ = 10.0;
  double position_limit_ = kPi;
  double cmd_period_ = 2.0;
  double period_scale_ = 1.0;
  double step_height_ = 0.1;
  std::string signal_type_ = "multi";

  bool enable_save_ = false;
  bool publish_debug_ = true;
  bool effort_ctrl_ = false;
  bool csv_header_ = true;
  bool load_effort_gains_ = true;
  std::string save_path_ = "/tmp";
  std::string file_prefix_ = "servo_response";

  std::vector<double> target_;
  std::vector<std::vector<double>> rows_;

  void loadParams()
  {
    nhp_.param("robot_ns", robot_ns_, std::string());
    if (robot_ns_.empty())
      nhp_.param("robots_ns", robot_ns_, std::string());

    state_topic_ = stringParam(nhp_, {"state_topic", "joint_state_topic"}, state_topic_);
    command_topic_ = stringParam(nhp_, {"command_topic", "servo_command_topic"},
                                 command_topic_);
    debug_topic_ = stringParam(nhp_, {"debug_topic"}, command_topic_ + "_debug");
    servo_group_ = stringParam(nhp_, {"servo_group", "effort_group"}, servo_group_);

    axis_names_ = stringParamList(nhp_, {"axis_names", "gimbal_names", "joint_names"});

    int axis_count = axis_names_.empty() ? 4 : static_cast<int>(axis_names_.size());
    if (!nhp_.getParam("axis_count", axis_count))
      nhp_.param("gimbal_size", axis_count, axis_count);
    if (axis_count < 1)
      {
        ROS_ERROR("servo_response: axis_count must be positive");
        valid_ = false;
        return;
      }
    axis_count_ = static_cast<size_t>(axis_count);
    if (!axis_names_.empty() && axis_names_.size() != axis_count_)
      {
        ROS_WARN("servo_response: axis_names size overrides axis_count (%zu != %zu)",
                 axis_names_.size(), axis_count_);
        axis_count_ = axis_names_.size();
      }

    bool found_enabled = false;
    enabled_ = boolParamList(nhp_, {"enabled_axes"}, axis_count_, true, &found_enabled);
    if (!found_enabled)
      {
        enabled_.assign(axis_count_, true);
        for (size_t i = 0; i < axis_count_; ++i)
          {
            const std::string old_key = "enable_gimbal_" + std::to_string(i);
            const std::string key = "enable_axis_" + std::to_string(i);
            bool value = enabled_[i];
            if (nhp_.getParam(key, value) || nhp_.getParam(old_key, value))
              enabled_[i] = value;
          }
      }

    default_pos_ = doubleParamList(nhp_, {"default_position", "default_gimbal"},
                                   axis_count_, 0.0);

    position_limit_ = doubleParam(nhp_, {"position_limit", "gimbal_range"},
                                  position_limit_);
    control_freq_ = doubleParam(nhp_, {"control_freq"}, control_freq_);
    duration_ = doubleParam(nhp_, {"duration"}, duration_);
    cmd_period_ = doubleParam(nhp_, {"cmd_period"}, cmd_period_);
    period_scale_ = doubleParam(nhp_, {"period_scale"}, period_scale_);
    step_height_ = doubleParam(nhp_, {"step_height"}, step_height_);

    nhp_.param("enable_save", enable_save_, enable_save_);
    nhp_.param("publish_debug", publish_debug_, publish_debug_);
    nhp_.param("effort_ctrl", effort_ctrl_, effort_ctrl_);
    nhp_.param("csv_header", csv_header_, csv_header_);
    nhp_.param("load_effort_gains", load_effort_gains_, load_effort_gains_);
    nhp_.param("save_path", save_path_, save_path_);
    nhp_.param("file_prefix", file_prefix_, file_prefix_);

    kp_ = doubleParamList(nhp_, {"kp"}, axis_count_, 0.285);
    kd_ = doubleParamList(nhp_, {"kd"}, axis_count_, 0.019);
    effort_limit_ = doubleParamList(nhp_, {"effort_limit"}, axis_count_,
                                    std::numeric_limits<double>::infinity());

    signal_type_ = stringParam(nhp_, {"signal_type", "waveform"}, std::string());
    if (signal_type_.empty())
      {
        bool sweep = false, step = false;
        nhp_.param("sweep_mode", sweep, false);
        nhp_.param("step_mode", step, false);
        if (step) signal_type_ = "step";
        else if (sweep) signal_type_ = "sweep";
        else signal_type_ = "multi";
      }
    signal_type_ = lower(signal_type_);
    if (signal_type_ == "stepset" || signal_type_ == "step_set")
      signal_type_ = "step";
    else if (signal_type_ == "ident" || signal_type_ == "identset" ||
             signal_type_ == "ident_set")
      signal_type_ = "sweep";
    else if (signal_type_ == "multiple" || signal_type_ == "multiplewave" ||
             signal_type_ == "val" || signal_type_ == "valset" ||
             signal_type_ == "validation")
      signal_type_ = "multi";

    if (signal_type_ != "step" && signal_type_ != "sweep" &&
        signal_type_ != "multi")
      {
        ROS_ERROR("servo_response: unknown signal_type '%s' (use step, sweep, or multi)",
                  signal_type_.c_str());
        valid_ = false;
      }

    if (control_freq_ <= 0.0 || cmd_period_ <= 0.0 || position_limit_ < 0.0)
      {
        ROS_ERROR("servo_response: control_freq and cmd_period must be positive; position_limit must be non-negative");
        valid_ = false;
      }
  }

  std::string scopedName(const std::string& name) const
  {
    if (name.empty() || name.front() == '/') return name;
    const std::string ns = trimSlash(robot_ns_);
    if (ns.empty()) return name;
    return "/" + ns + "/" + name;
  }

  std::string topic(const std::string& name) const
  {
    return scopedName(name);
  }

  void rebuildNameIndex()
  {
    name_index_.clear();
    for (size_t i = 0; i < axis_names_.size(); ++i)
      name_index_[axis_names_[i]] = i;
  }

  bool setupEffortPublishers()
  {
    effort_pubs_.resize(axis_count_);

    bool topics_found = false;
    std::vector<std::string> topics =
      stringParamList(nhp_, {"effort_topics"}, &topics_found);
    if (topics_found && !topics.empty())
      {
        if (topics.size() != axis_count_)
          {
            ROS_ERROR("servo_response: effort_topics size (%zu) != axis_count (%zu)",
                      topics.size(), axis_count_);
            return false;
          }
        for (size_t i = 0; i < axis_count_; ++i)
          effort_pubs_[i] = nh_.advertise<std_msgs::Float64>(topic(topics[i]), 1);
        return true;
      }

    XmlRpc::XmlRpcValue group_xml;
    if (nh_.getParam(scopedName("servo_controller/" + servo_group_), group_xml) &&
        group_xml.getType() == XmlRpc::XmlRpcValue::TypeStruct)
      return setupEffortFromServoParams(group_xml);

    ROS_WARN("servo_response: no servo_controller/%s params; using controller order",
             servo_group_.c_str());
    for (size_t i = 0; i < axis_count_; ++i)
      {
        const std::string t = "servo_controller/" + servo_group_ + "/controller" +
                              std::to_string(i + 1) + "/simulation/command";
        effort_pubs_[i] = nh_.advertise<std_msgs::Float64>(topic(t), 1);
      }
    return true;
  }

  void loadGroupEffortGains(XmlRpc::XmlRpcValue& group_xml)
  {
    if (!load_effort_gains_ || !group_xml.hasMember("simulation")) return;
    XmlRpc::XmlRpcValue& sim = group_xml["simulation"];
    if (sim.hasMember("pid"))
      {
        XmlRpc::XmlRpcValue& pid = sim["pid"];
        const double kp = xmlMemberDouble(pid, "p", kp_[0]);
        const double kd = xmlMemberDouble(pid, "d", kd_[0]);
        std::fill(kp_.begin(), kp_.end(), kp);
        std::fill(kd_.begin(), kd_.end(), kd);
      }
    const double limit = xmlMemberDouble(sim, "effort_limit", effort_limit_[0]);
    std::fill(effort_limit_.begin(), effort_limit_.end(), limit);
  }

  void loadControllerEffortGains(XmlRpc::XmlRpcValue& ctrl, size_t index)
  {
    if (!load_effort_gains_ || !ctrl.hasMember("simulation")) return;
    XmlRpc::XmlRpcValue& sim = ctrl["simulation"];
    if (sim.hasMember("pid"))
      {
        XmlRpc::XmlRpcValue& pid = sim["pid"];
        kp_[index] = xmlMemberDouble(pid, "p", kp_[index]);
        kd_[index] = xmlMemberDouble(pid, "d", kd_[index]);
      }
    effort_limit_[index] = xmlMemberDouble(sim, "effort_limit", effort_limit_[index]);
  }

  bool setupEffortFromServoParams(XmlRpc::XmlRpcValue& group_xml)
  {
    loadGroupEffortGains(group_xml);

    std::vector<bool> found(axis_count_, false);
    std::vector<std::pair<int, std::string>> ordered_topics;

    for (auto it = group_xml.begin(); it != group_xml.end(); ++it)
      {
        const std::string key = static_cast<std::string>(it->first);
        int ctrl_index = 0;
        if (!controllerIndex(key, ctrl_index)) continue;

        XmlRpc::XmlRpcValue& ctrl = it->second;
        const std::string t = "servo_controller/" + servo_group_ + "/" +
                              key + "/simulation/command";

        if (axis_names_.empty())
          {
            ordered_topics.push_back({ctrl_index, t});
            continue;
          }

        if (!ctrl.hasMember("name")) continue;
        const std::string name = static_cast<std::string>(ctrl["name"]);
        const auto name_it = name_index_.find(name);
        if (name_it == name_index_.end()) continue;

        const size_t axis = name_it->second;
        loadControllerEffortGains(ctrl, axis);
        effort_pubs_[axis] = nh_.advertise<std_msgs::Float64>(topic(t), 1);
        found[axis] = true;
      }

    if (axis_names_.empty())
      {
        std::sort(ordered_topics.begin(), ordered_topics.end());
        for (size_t i = 0; i < axis_count_ && i < ordered_topics.size(); ++i)
          {
            effort_pubs_[i] =
              nh_.advertise<std_msgs::Float64>(topic(ordered_topics[i].second), 1);
            found[i] = true;
          }
      }

    bool ok = true;
    for (size_t i = 0; i < axis_count_; ++i)
      {
        if (found[i]) continue;
        ROS_ERROR("servo_response: no effort controller for axis %zu (%s)",
                  i, axisName(i).c_str());
        ok = false;
      }
    return ok;
  }

  bool waitForState()
  {
    ROS_INFO("servo_response: waiting for %s", topic(state_topic_).c_str());
    ros::Rate rate(50.0);
    while (ros::ok())
      {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (readyLocked()) return true;
        }
        ros::spinOnce();
        rate.sleep();
      }
    return false;
  }

  bool readyLocked() const
  {
    bool has_enabled_axis = false;
    for (size_t i = 0; i < axis_count_; ++i)
      {
        if (!enabled_[i]) continue;
        has_enabled_axis = true;
        if (!seen_[i]) return false;
      }
    if (has_enabled_axis) return true;
    return std::any_of(seen_.begin(), seen_.end(), [](bool v) { return v; });
  }

  State snapshot()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    State s;
    s.pos = pos_;
    s.vel = vel_;
    s.names = state_names_;
    s.stamp = state_stamp_;
    s.receive_stamp = receive_stamp_;
    return s;
  }

  std::string axisName(size_t i) const
  {
    if (i < axis_names_.size() && !axis_names_[i].empty()) return axis_names_[i];
    if (i < state_names_.size() && !state_names_[i].empty()) return state_names_[i];
    return "axis_" + std::to_string(i);
  }

  std::vector<std::string> commandNames(const State& state) const
  {
    std::vector<std::string> names(axis_count_);
    for (size_t i = 0; i < axis_count_; ++i)
      {
        if (i < axis_names_.size() && !axis_names_[i].empty())
          names[i] = axis_names_[i];
        else if (i < state.names.size() && !state.names[i].empty())
          names[i] = state.names[i];
        else
          names[i] = "axis_" + std::to_string(i);
      }
    return names;
  }

  void publishCommand(double t)
  {
    target_ = signal(t);

    const State state = snapshot();
    const std::vector<std::string> names = commandNames(state);

    sensor_msgs::JointState cmd;
    cmd.header.stamp = ros::Time::now();
    cmd.name = names;
    cmd.position.resize(axis_count_);
    if (effort_ctrl_) cmd.effort.resize(axis_count_);

    for (size_t i = 0; i < axis_count_; ++i)
      {
        if (!enabled_[i]) target_[i] = 0.0;
        target_[i] = clamp(target_[i], -position_limit_, position_limit_);
        cmd.position[i] = default_pos_[i] + target_[i];

        if (effort_ctrl_)
          {
            double effort = kp_[i] * (cmd.position[i] - state.pos[i]) -
                            kd_[i] * state.vel[i];
            if (!std::isfinite(effort)) effort = 0.0;
            if (std::isfinite(effort_limit_[i]))
              effort = clamp(effort, -effort_limit_[i], effort_limit_[i]);
            cmd.effort[i] = effort;

            std_msgs::Float64 msg;
            msg.data = effort;
            effort_pubs_[i].publish(msg);
          }
      }

    if (!effort_ctrl_) command_pub_.publish(cmd);
    if (debug_pub_) debug_pub_.publish(cmd);
    if (enable_save_) appendSample(t, state, cmd.position);
  }

  void appendSample(double t, const State& state, const std::vector<double>& cmd)
  {
    std::vector<double> row;
    row.reserve(3 + 2 * axis_count_);
    row.push_back(t);
    row.push_back(state.stamp.toNSec() * 1e-6);
    row.push_back(state.receive_stamp.toNSec() * 1e-6);
    row.insert(row.end(), state.pos.begin(), state.pos.end());
    row.insert(row.end(), cmd.begin(), cmd.end());
    rows_.push_back(row);
  }

  std::vector<double> signal(double t)
  {
    double value = 0.0;
    if (signal_type_ == "step")
      value = step(t);
    else if (signal_type_ == "sweep")
      value = sweep(t);
    else
      value = multi(t);

    return std::vector<double>(axis_count_, value);
  }

  double sine(double t, double period, double amp) const
  {
    return amp * std::sin(2.0 * kPi * std::fmod(t, period) / period);
  }

  double triangle(double t, double period, double amp) const
  {
    double phase = std::fmod(t + 0.25 * period, period);
    if (phase < 0.0) phase += period;
    if (phase < 0.5 * period)
      return -amp + 4.0 * amp * phase / period;
    return amp - 4.0 * amp * (phase - 0.5 * period) / period;
  }

  double square(double t, double period, double amp) const
  {
    return std::fmod(t, period) < 0.5 * period ? amp : -amp;
  }

  double trapezoid(double t, double period, double amp) const
  {
    return clamp(triangle(t, period, 2.0 * amp), -amp, amp);
  }

  double sinTriangle(double t, double period, double amp) const
  {
    const double env = std::abs(triangle(t, 32.0 * period, amp));
    return env * std::sin(2.0 * kPi * std::fmod(t, period) / period);
  }

  // Validation set: replay common wave shapes in fixed order. Only cmd_period
  // is user-facing; the sequence is intentionally fixed for model validation.
  double multi(double t) const
  {
    const double phase = std::fmod(t, 32.0 * cmd_period_);
    if (phase > 16.0 * cmd_period_) return sinTriangle(t, cmd_period_, position_limit_);
    if (phase > 12.0 * cmd_period_) return trapezoid(t, cmd_period_, position_limit_);
    if (phase > 8.0 * cmd_period_) return square(t, cmd_period_, position_limit_);
    if (phase > 4.0 * cmd_period_) return triangle(t, cmd_period_, position_limit_);
    return sine(t, cmd_period_, position_limit_);
  }

  // Step set: hold each level for cmd_period, move by step_height, and bounce
  // inside the symmetric work range.
  double step(double t) const
  {
    if (step_height_ <= 0.0) return 0.0;
    int steps = static_cast<int>(std::ceil(2.0 * position_limit_ / step_height_));
    steps = std::max(steps, 1);

    const long cycle = 2L * steps;
    long phase = static_cast<long>(std::floor(t / cmd_period_)) % cycle;
    if (phase < 0) phase += cycle;

    const long level = phase <= steps ? phase : cycle - phase;
    return clamp(-position_limit_ + level * step_height_,
                 -position_limit_, position_limit_);
  }

  // Identification set: beetle_omni SweepSinWave. The sine period sweeps
  // short->long->short while the amplitude envelope grows and shrinks once.
  double sweep(double t) const
  {
    const double pad = 1.0;
    const double active = duration_ - 2.0 * pad;
    if (t < pad || t > duration_ - pad || active <= 0.0) return 0.0;

    const double ta = t - pad;
    const double alpha = ta / active;
    const double progress = 1.0 - std::abs(2.0 * alpha - 1.0);
    const double fast = std::max(1e-6, std::min(cmd_period_, period_scale_ * cmd_period_));
    const double slow = std::max(fast, std::max(cmd_period_, period_scale_ * cmd_period_));
    const double delta = slow - fast;
    const double half = 0.5 * active;
    const double period = fast + delta * progress;

    double phase = 2.0 * kPi * ta / fast;
    if (delta > 1e-9 && half > 1e-9)
      {
        const double slope = delta / half;
        const double phase_half = 2.0 * kPi * std::log(slow / fast) / slope;
        if (ta <= half)
          phase = 2.0 * kPi * std::log(period / fast) / slope;
        else
          phase = phase_half + 2.0 * kPi * std::log(slow / period) / slope;
      }

    return position_limit_ * progress * std::sin(phase);
  }

  bool saveCsv()
  {
    std::string filename = save_path_;
    if (!filename.empty() && filename.back() != '/') filename += "/";

    std::time_t now = std::time(nullptr);
    std::tm tm = *std::localtime(&now);
    std::ostringstream stamp;
    stamp << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    filename += file_prefix_ + "_" + stamp.str() + ".csv";

    std::ofstream ofs(filename);
    if (!ofs)
      {
        ROS_ERROR("servo_response: failed to open %s", filename.c_str());
        return false;
      }

    const State state = snapshot();
    const std::vector<std::string> names = commandNames(state);
    ofs << std::fixed << std::setprecision(6);
    if (csv_header_)
      {
        ofs << "time,state_stamp_ms,receive_stamp_ms";
        for (const auto& name : names) ofs << ",pos_" << name;
        for (const auto& name : names) ofs << ",cmd_" << name;
        ofs << "\n";
      }

    for (const auto& row : rows_)
      {
        for (size_t i = 0; i < row.size(); ++i)
          {
            if (i) ofs << ",";
            ofs << row[i];
          }
        ofs << "\n";
      }
    ROS_INFO("servo_response: saved %zu samples to %s", rows_.size(), filename.c_str());
    return true;
  }

  void stateCallback(const sensor_msgs::JointState::ConstPtr& msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);

    const bool has_vel = msg->velocity.size() == msg->name.size();
    double dt = 0.0;
    if (!state_stamp_.isZero())
      dt = (msg->header.stamp - state_stamp_).toSec();

    if (!axis_names_.empty())
      {
        for (size_t i = 0; i < msg->name.size(); ++i)
          {
            if (i >= msg->position.size()) break;
            const auto it = name_index_.find(msg->name[i]);
            if (it == name_index_.end()) continue;

            const size_t axis = it->second;
            if (has_vel)
              vel_[axis] = msg->velocity[i];
            else if (seen_[axis] && dt > 1e-6)
              vel_[axis] = (msg->position[i] - pos_[axis]) / dt;
            pos_[axis] = msg->position[i];
            state_names_[axis] = msg->name[i];
            seen_[axis] = true;
          }
      }
    else
      {
        const size_t n = std::min(axis_count_, msg->position.size());
        for (size_t i = 0; i < n; ++i)
          {
            if (has_vel)
              vel_[i] = msg->velocity[i];
            else if (seen_[i] && dt > 1e-6)
              vel_[i] = (msg->position[i] - pos_[i]) / dt;
            pos_[i] = msg->position[i];
            if (i < msg->name.size()) state_names_[i] = msg->name[i];
            seen_[i] = true;
          }
      }

    state_stamp_ = msg->header.stamp;
    receive_stamp_ = ros::Time::now();
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "servo_response");
  ros::NodeHandle nh;
  ros::NodeHandle nhp("~");

  ServoResponse node(nh, nhp);
  return node.run() ? 0 : 1;
}
