//
// Created by wentao-zhang on 2026-07-11.
//
// Whole-body target-state tracking RL controller for DRAGON.
// Deployment counterpart of robots/dragon/policy/dragon_track_env.py (DragonTrackEnv).
//
// Observation (per policy step, 40 Hz):
//   [ target: K x (rel_pos(3) + rel_rot6d(6) + target_joint(6)) |
//     joint_pos(6) | joint_vel(6)*dof_vel (or last_joint_pos) |
//     gimbal_pos(8) | gimbal_vel(8)*dof_vel (or last_gimbal_pos) |
//     root_rot6d(6) | lin_vel_b(3)*lin_vel | ang_vel_b(3)*ang_vel | last_action(18) ]
// Action: [ joint(6)*joint_act | gimbal(8)*gimbal_act | paired per-EDF thrust(4)*thrust_act + hover ]
//

#include "dragon/control/dragon_tracker_rlagent.h"
#include <ros/package.h>
#include <xmlrpcpp/XmlRpcValue.h>
#include <tf/transform_datatypes.h>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>

namespace
{

tf::Vector3 rotateByQuatInv(const tf::Quaternion& q, const tf::Vector3& v)
{
  return tf::Matrix3x3(q.inverse()) * v;
}

double normalizeAngle(double angle)
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

// shortest-path slerp (tf/Bullet slerp does not flip the sign of q1 by itself)
tf::Quaternion safeSlerp(const tf::Quaternion& q0, tf::Quaternion q1, double t)
{
  if (q0.dot(q1) < 0.0) q1 = tf::Quaternion(-q1.x(), -q1.y(), -q1.z(), -q1.w());
  return q0.slerp(q1, t).normalized();
}

// Reads an ordered list-of-strings param (e.g. 'joint_names'); empty when absent.
std::vector<std::string> parseStringListParam(ros::NodeHandle& nh, const std::string& key,
                                              const std::vector<std::string>& default_value)
{
  XmlRpc::XmlRpcValue xml;
  if (!nh.getParam(key, xml) || xml.getType() != XmlRpc::XmlRpcValue::TypeArray)
    return default_value;
  std::vector<std::string> result;
  result.reserve(xml.size());
  for (int i = 0; i < xml.size(); ++i)
    result.push_back(static_cast<std::string>(xml[i]));
  return result;
}

std::string resolvePackagePath(const std::string& path)
{
  if (!path.empty() && path.front() == '/') return path;
  return ros::package::getPath("dragon") + "/" + path;
}

bool isControllerKey(const std::string& key)
{
  const std::string prefix = "controller";
  if (key.compare(0, prefix.size(), prefix) != 0) return false;
  for (size_t i = prefix.size(); i < key.size(); ++i)
    if (!std::isdigit(static_cast<unsigned char>(key[i]))) return false;
  return key.size() > prefix.size();
}

double getXmlRpcDouble(XmlRpc::XmlRpcValue& value, const std::string& key, double fallback)
{
  if (!value.hasMember(key)) return fallback;
  XmlRpc::XmlRpcValue& v = value[key];
  if (v.getType() == XmlRpc::XmlRpcValue::TypeInt) return static_cast<int>(v);
  if (v.getType() == XmlRpc::XmlRpcValue::TypeDouble) return static_cast<double>(v);
  return fallback;
}

double clampDouble(double value, double lower, double upper)
{
  return std::min(std::max(value, lower), upper);
}

int extractFirstInteger(const std::string& text)
{
  for (size_t i = 0; i < text.size(); ++i)
    {
      if (!std::isdigit(static_cast<unsigned char>(text[i]))) continue;
      size_t end = i + 1;
      while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end]))) ++end;
      return std::stoi(text.substr(i, end - i));
    }
  return -1;
}

std::string jointShortName(const std::string& name)
{
  char axis = '?';
  if (name.find("yaw") != std::string::npos) axis = 'Y';
  else if (name.find("pitch") != std::string::npos) axis = 'P';

  int id = extractFirstInteger(name);
  if (axis == '?' || id < 0) return name;
  return std::string(1, axis) + std::to_string(id);
}

std::string gimbalShortName(const std::string& name)
{
  char axis = '?';
  if (name.find("roll") != std::string::npos) axis = 'R';
  else if (name.find("pitch") != std::string::npos) axis = 'P';
  else if (name.find("yaw") != std::string::npos) axis = 'Y';

  int id = extractFirstInteger(name);
  if (axis == '?' || id < 0) return name;
  return "G" + std::string(1, axis) + std::to_string(id);
}

}  // namespace

namespace aerial_robot_control
{

DragonTrackerRLAgent::DragonTrackerRLAgent() : ControlBase()
{
}

DragonTrackerRLAgent::~DragonTrackerRLAgent()
{
  if (input_name_) allocator_.Free(const_cast<char*>(input_name_));
  if (output_name_) allocator_.Free(const_cast<char*>(output_name_));
}

void DragonTrackerRLAgent::initialize(ros::NodeHandle nh, ros::NodeHandle nhp,
                                      boost::shared_ptr<aerial_robot_model::RobotModel> robot_model,
                                      boost::shared_ptr<aerial_robot_estimation::StateEstimator> estimator,
                                      boost::shared_ptr<aerial_robot_navigation::BaseNavigator> navigator,
                                      double ctrl_loop_du)
{
  ControlBase::initialize(nh, nhp, robot_model, estimator, navigator, ctrl_loop_du);

  ros::NodeHandle rl_nh(nh_, "rlagent");

  std::string onnx_model_path;
  if (!rl_nh.getParam("onnx_model_path", onnx_model_path) || onnx_model_path.empty())
    {
      ROS_ERROR("[RL-Tracker] Parameter rlagent/onnx_model_path is not set!");
      return;
    }
  onnx_model_path = resolvePackagePath(onnx_model_path);
  ROS_INFO("[RL-Tracker] ONNX model path: %s", onnx_model_path.c_str());

  getParam<int>(rl_nh, "decimation", decimation_, 1);
  getParam<bool>(rl_nh, "ideal_obs", ideal_obs_, false);
  getParam<bool>(rl_nh, "ideal_pos_obs", ideal_pos_obs_, true);
  getParam<bool>(rl_nh, "ideal_quat_obs", ideal_quat_obs_, true);
  getParam<bool>(rl_nh, "ideal_lin_vel_obs", ideal_lin_vel_obs_, true);
  getParam<bool>(rl_nh, "ideal_ang_vel_obs", ideal_ang_vel_obs_, true);
  getParam<bool>(rl_nh, "replace_dof_vel_with_last_pos",
                 replace_dof_vel_with_last_pos_, false);
  getParam<std::string>(rl_nh, "ideal_obs_topic", ideal_obs_topic_, "uav/baselink/odom");
  getParam<int>(rl_nh, "ideal_delay", ideal_delay_, 0);
  ideal_delay_ = std::max(0, ideal_delay_);
  getParam<bool>(rl_nh, "rlagent_verbose", verbose_, false);
  getParam<bool>(rl_nh, "rlagent_debug", debug_, false);
  getParam<bool>(rl_nh, "rlagent_forward_info", forward_info_, false);

  getParam<std::string>(rl_nh, "target_state_topic", target_state_topic_, "full_state_target");
  getParam<int>(rl_nh, "target_obs_horizon", target_obs_horizon_, 5);
  target_obs_horizon_ = std::max(target_obs_horizon_, 1);
  getParam<int>(rl_nh, "history_length", history_length_, 1);
  history_length_ = std::max(history_length_, 1);

  getParam<bool>(rl_nh, "enable_thrust", enable_thrust_, false);
  getParam<bool>(rl_nh, "enable_joint", enable_joint_, false);
  getParam<bool>(rl_nh, "enable_gimbal", enable_gimbal_, false);
  getParam<bool>(rl_nh, "joint_effort_ctrl", joint_effort_ctrl_, false);
  getParam<bool>(rl_nh, "gimbal_effort_ctrl", gimbal_effort_ctrl_, false);
  getParam<bool>(rl_nh, "action_clamp", action_clamp_, false);
  getParam<double>(rl_nh, "thrust_hover", thrust_hover_, 8.5);
  getParam<double>(rl_nh, "thrust_limit", thrust_limit_, 16.0);
  getParam<double>(rl_nh, "thrust_scale", thrust_scale_default_, 1.0);
  if (thrust_scale_default_ > 1.0) thrust_scale_default_ = 1.0;
  getParam<double>(rl_nh, "joint_limit", joint_limit_, 1.57);
  getParam<double>(rl_nh, "gimbal_roll_limit", gimbal_roll_limit_, M_PI);
  getParam<double>(rl_nh, "gimbal_pitch_limit", gimbal_pitch_limit_, M_PI);
  getParam<int>(rl_nh, "joint_target_delay_steps", joint_target_delay_steps_, 0);
  getParam<int>(rl_nh, "gimbal_target_delay_steps", gimbal_target_delay_steps_, 0);
  getParam<int>(rl_nh, "thrust_target_delay_steps", thrust_target_delay_steps_, 0);

  // actuator layouts: must match the training env's joint/gimbal/pair ordering
  joint_names_ = parseStringListParam(rl_nh, "joint_names",
      {"joint1_yaw", "joint2_pitch", "joint1_pitch", "joint2_yaw", "joint3_pitch", "joint3_yaw"});
  gimbal_names_ = parseStringListParam(rl_nh, "gimbal_names",
      {"gimbal2_roll", "gimbal2_pitch", "gimbal1_roll", "gimbal3_roll",
       "gimbal1_pitch", "gimbal3_pitch", "gimbal4_roll", "gimbal4_pitch"});
  thrust_pair_names_ = parseStringListParam(rl_nh, "thrust_pair_names",
      {"edf2", "edf1", "edf3", "edf4"});
  joint_size_ = static_cast<int>(joint_names_.size());
  gimbal_size_ = static_cast<int>(gimbal_names_.size());
  pair_size_ = static_cast<int>(thrust_pair_names_.size());
  if (!buildGimbalPositionLimits(gimbal_roll_limit_, gimbal_pitch_limit_,
                                 gimbal_position_limits_))
    return;

  obs_terms_ = parseStringListParam(rl_nh, "observations",
      {"target_obs_history", "joint_pos", "joint_vel", "gimbal_pos", "gimbal_vel",
       "root_rot_vec", "lin_vel", "ang_vel", "last_action"});
  target_obs_terms_ = parseStringListParam(rl_nh, "target_obs",
      {"root_pos", "root_rot_vec", "joint_pos"});
  history_terms_ = parseStringListParam(rl_nh, "history_terms", {});
  action_groups_ = parseStringListParam(rl_nh, "actions", {"joint", "gimbal", "thrust"});

  // pair name "edfN" -> base_thrust index N-1 (bag convention: base_thrust[i] == thrust{i+1})
  pair_module_index_.clear();
  for (const auto& name : thrust_pair_names_)
    {
      size_t digit_pos = name.find_first_of("0123456789");
      if (digit_pos == std::string::npos)
        {
          ROS_ERROR("[RL-Tracker] thrust pair name '%s' contains no module index!", name.c_str());
          return;
        }
      int module = std::stoi(name.substr(digit_pos)) - 1;
      if (module < 0 || module >= motor_num_)
        {
          ROS_ERROR("[RL-Tracker] thrust pair '%s' -> module %d out of range (motor_num=%d)!",
                    name.c_str(), module, motor_num_);
          return;
        }
      pair_module_index_.push_back(module);
    }

  // default positions (training used all-zero); optional name->value dict in YAML
  joint_default_pos_.assign(joint_size_, 0.0);
  gimbal_default_pos_.assign(gimbal_size_, 0.0);
  XmlRpc::XmlRpcValue defaults_xml;
  if (rl_nh.getParam("joint_default", defaults_xml))
    for (int i = 0; i < joint_size_; ++i)
      if (defaults_xml.hasMember(joint_names_[i]))
        joint_default_pos_[i] = static_cast<double>(defaults_xml[joint_names_[i]]);
  if (rl_nh.getParam("gimbal_default", defaults_xml))
    for (int i = 0; i < gimbal_size_; ++i)
      if (defaults_xml.hasMember(gimbal_names_[i]))
        gimbal_default_pos_[i] = static_cast<double>(defaults_xml[gimbal_names_[i]]);

  // scales
  XmlRpc::XmlRpcValue scales_xml;
  getParam<XmlRpc::XmlRpcValue>(rl_nh, "scales", scales_xml, XmlRpc::XmlRpcValue());
  for (auto it = scales_xml.begin(); it != scales_xml.end(); ++it)
    scales_[static_cast<std::string>(it->first)] = static_cast<double>(it->second);
  for (const auto& key : {"dof_vel", "lin_vel", "ang_vel", "joint_act", "gimbal_act",
                          "thrust_act", "clip_observation", "clip_action"})
    {
      if (scales_.find(key) == scales_.end())
        {
          ROS_ERROR("[RL-Tracker] Missing scale '%s' under rlagent/scales in YAML!", key);
          return;
        }
    }

  // state buffers
  joint_pos_.assign(joint_size_, 0.0);
  joint_vel_.assign(joint_size_, 0.0);
  gimbal_pos_.assign(gimbal_size_, 0.0);
  gimbal_vel_.assign(gimbal_size_, 0.0);
  last_joint_pos_.assign(joint_size_, 0.0);
  last_gimbal_pos_.assign(gimbal_size_, 0.0);
  target_joint_.assign(joint_size_, 0.0);
  target_gimbal_.assign(gimbal_size_, 0.0);
  target_thrust_.assign(pair_size_, 0.0);
  target_frames_.assign(target_obs_horizon_, RefFrame{tf::Vector3(0, 0, 0),
                                                      tf::Quaternion(0, 0, 0, 1),
                                                      std::vector<double>(joint_size_, 0.0)});

  // ONNX runtime
  if (verbose_) ROS_INFO("[RL-Tracker] Loading agent model......");
  try
    {
      initRLAgent(onnx_model_path);
    }
  catch (const std::exception& e)
    {
      ROS_ERROR("[RL-Tracker] Failed to load ONNX model %s: %s", onnx_model_path.c_str(), e.what());
      return;
    }

  size_t expected_action = 0;
  for (const auto& group : action_groups_)
    expected_action += actionGroupDim(group);
  if (action_size_ != expected_action)
    {
      ROS_ERROR("[RL-Tracker] ONNX action size (%zu) != configured actions size (%zu)!",
                action_size_, expected_action);
      return;
    }
  single_obs_size_ = 0;
  for (const auto& term : history_terms_) single_obs_size_ += static_cast<int>(obsTermDim(term));
  if (!history_terms_.empty())
    {
      history_observations_.assign(history_length_,
                                   std::vector<float>(single_obs_size_, 0.0f));
    }

  size_t expected_obs = 0;
  for (const auto& term : obs_terms_)
    {
      const size_t dim = obsTermDim(term);
      const bool in_history = std::find(history_terms_.begin(), history_terms_.end(), term) != history_terms_.end();
      expected_obs += in_history ? dim * static_cast<size_t>(history_length_) : dim;
    }
  if (obs_size_ != expected_obs)
    {
      ROS_ERROR("[RL-Tracker] ONNX obs size (%zu) != configured observations size (%zu)! "
                "Check observations/history_terms/target_obs_horizon/action layout.",
                obs_size_, expected_obs);
      return;
    }

  // ROS I/O
  joint_state_sub_ = nh_.subscribe("joint_states", 1, &DragonTrackerRLAgent::jointStateCallback, this);
  configureIdealObservationSubscriber();
  target_state_sub_ = nh_.subscribe(target_state_topic_, 10, &DragonTrackerRLAgent::targetStateCallback, this);

  thrust_pub_ = nh_.advertise<spinal::FourAxisCommand>("four_axes/command", 1);
  enable_all_arm_sub_ = nh_.subscribe("teleop_command/start", 1, &DragonTrackerRLAgent::enableJointAndGimbalCallback, this);
  enable_all_tracking_sub_ = nh_.subscribe("teleop_command/start_tracking", 1, &DragonTrackerRLAgent::enableJointAndGimbalCallback, this);
  force_landing_sub_ = nh_.subscribe("teleop_command/force_landing", 1, &DragonTrackerRLAgent::forceLandingCallback, this);
  halt_sub_ = nh_.subscribe("teleop_command/halt", 1, &DragonTrackerRLAgent::haltCallback, this);
  joint_pub_ = nh_.advertise<sensor_msgs::JointState>("joints_ctrl", 1);
  gimbal_pub_ = nh_.advertise<sensor_msgs::JointState>("gimbals_ctrl", 1);
  thrust_debug_pub_ = nh_.advertise<spinal::FourAxisCommand>("rlagent/thrust_cmd_debug", 1);
  joint_debug_pub_ = nh_.advertise<sensor_msgs::JointState>("rlagent/joint_cmd_debug", 1);
  gimbal_debug_pub_ = nh_.advertise<sensor_msgs::JointState>("rlagent/gimbal_cmd_debug", 1);
  obs_debug_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("rlagent/observation_debug", 1);

  reload_policy_srv_ = nh_.advertiseService("rlagent/reload_policy",
                                            &DragonTrackerRLAgent::reloadPolicyCallback, this);
  reload_params_srv_ = nh_.advertiseService("rlagent/reload_params",
                                            &DragonTrackerRLAgent::reloadParamsCallback, this);

  // command messages
  thrust_cmd_.base_thrust = std::vector<float>(motor_num_, 0.0);
  joint_cmd_.name = joint_names_;
  joint_cmd_.position.assign(joint_size_, 0.0);
  gimbal_cmd_.name = gimbal_names_;
  gimbal_cmd_.position.assign(gimbal_size_, 0.0);

  if (joint_effort_ctrl_ &&
      !loadEffortControlParams("joints", joint_names_, joint_kp_, joint_kd_,
                               joint_effort_limit_, joint_effort_pubs_))
    {
      ROS_ERROR("[RL-Tracker] Failed to load joint effort control parameters.");
      return;
    }
  if (gimbal_effort_ctrl_ &&
      !loadEffortControlParams("gimbals", gimbal_names_, gimbal_kp_, gimbal_kd_,
                               gimbal_effort_limit_, gimbal_effort_pubs_))
    {
      ROS_ERROR("[RL-Tracker] Failed to load gimbal effort control parameters.");
      return;
    }

  last_infer_report_time_ = ros::Time::now();
  if (verbose_) ROS_INFO("[RL-Tracker] Initialized successfully.");
}

// ================================ RL agent ===================================

void DragonTrackerRLAgent::initRLAgent(std::string model_path)
{
  ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "dragon_tracker_rlagent");
  sess_opts_ = std::make_unique<Ort::SessionOptions>();
  sess_opts_->SetIntraOpNumThreads(2);
  sess_opts_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_ = std::make_unique<Ort::Session>(*ort_env_, model_path.c_str(), *sess_opts_);

  Ort::AllocatorWithDefaultOptions allocator;
  input_name_ = session_->GetInputNameAllocated(0, allocator).release();
  output_name_ = session_->GetOutputNameAllocated(0, allocator).release();

  auto in_dims = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  auto out_dims = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  auto dims_to_size = [](const std::vector<int64_t>& dims) {
    size_t prod = 1;
    for (auto d : dims) if (d > 0) prod *= static_cast<size_t>(d);
    return prod;
  };
  obs_size_ = dims_to_size(in_dims);
  action_size_ = dims_to_size(out_dims);
  if (verbose_)
    ROS_INFO("[RL-Tracker] Model input size: %zu, output size: %zu", obs_size_, action_size_);

  observation_.assign(obs_size_, 0.0f);
  action_.assign(action_size_, 0.0f);
  last_action_.assign(action_size_, 0.0f);
}

void DragonTrackerRLAgent::updateReference()
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  if (target_state_fifo_.empty())
    {
      target_state_catch_ = false;
      target_frames_.clear();
      return;
    }

  target_frames_.assign(target_state_fifo_.begin(), target_state_fifo_.end());
  while (static_cast<int>(target_frames_.size()) < target_obs_horizon_)
    target_frames_.push_back(target_frames_.back());
  if (static_cast<int>(target_frames_.size()) > target_obs_horizon_)
    target_frames_.erase(target_frames_.begin(),
                         target_frames_.end() - target_obs_horizon_);

  // The old CSV playback path toggled tracking_ from services.  The current
  // FullStateTarget path has no such service, so derive the display state from
  // the actual target window that enters the policy observation.
  tracking_ = false;
  if (target_frames_.size() > 1)
    {
      const RefFrame& ref = target_frames_.front();
      for (size_t i = 1; i < target_frames_.size(); ++i)
        {
          const RefFrame& f = target_frames_[i];
          if ((f.pos - ref.pos).length2() > 1e-8)
            {
              tracking_ = true;
              break;
            }
          const double quat_dot = std::abs(ref.quat.dot(f.quat));
          if (1.0 - std::min(1.0, quat_dot) > 1e-6)
            {
              tracking_ = true;
              break;
            }
          const size_t joint_count = std::min(ref.joint_pos.size(), f.joint_pos.size());
          for (size_t j = 0; j < joint_count; ++j)
            {
              if (std::abs(f.joint_pos[j] - ref.joint_pos[j]) <= 1e-4) continue;
              tracking_ = true;
              break;
            }
          if (tracking_) break;
        }
    }

  target_state_catch_ = true;
}

bool DragonTrackerRLAgent::update()
{
  if (!session_)
    {
      ROS_ERROR_THROTTLE(5.0, "[RL-Tracker] Agent not initialized (no ONNX session); no command sent.");
      return false;
    }
  ros::Time now = ros::Time::now();
  if (step_count_ % decimation_ == 0)
    {
      updateReference();
      buildObservation();
      // always run inference once observations are available (so it can be checked on
      // the ground); sendCmd() decides per channel whether the output is published
      policyForward();
    }
  step_count_++;
  sendCmd();

  double elapsed = (now - last_infer_report_time_).toSec();
  if (elapsed >= 2.0 && infer_count_ > 0)
    {
      if (forward_info_)
        {
          double avg_ms = static_cast<double>(infer_total_ns_) / infer_count_ / 1e6;
          char buf[256];
          std::ostringstream info;
          snprintf(buf, sizeof(buf),
                   "\033[1;32m[RL-Tracker]\033[0m ORT %.2fs: calls=%lu avg=%.3fms min=%.3fms max=%.3fms | "
                   "target fifo %zu/%d %s",
                   elapsed, static_cast<unsigned long>(infer_count_), avg_ms,
                   infer_min_ns_ / 1e6, infer_max_ns_ / 1e6,
                   target_state_fifo_.size(), target_obs_horizon_, tracking_ ? "TRACK" : "HOLD");
          info << buf;

          if (debug_)
            {
              const double pos_norm = pos_error_.length();
              const double rpy_norm = std::sqrt(rpy_error_.x() * rpy_error_.x() +
                                                rpy_error_.y() * rpy_error_.y() +
                                                rpy_error_.z() * rpy_error_.z());
              snprintf(buf, sizeof(buf),
                       "\n  root err: xyz [%+.3f %+.3f %+.3f] |norm %.3f  "
                       "rpy [%+.3f %+.3f %+.3f] |norm %.3f",
                       pos_error_.x(), pos_error_.y(), pos_error_.z(), pos_norm,
                       rpy_error_.x(), rpy_error_.y(), rpy_error_.z(), rpy_norm);
              info << buf;

              info << "\n  joint err:";
              if (!target_frames_.empty())
                {
                  for (int j = 0; j < joint_size_; ++j)
                    {
                      snprintf(buf, sizeof(buf), " %s %+.3f",
                               jointShortName(joint_names_[j]).c_str(),
                               target_frames_[0].joint_pos[j] - joint_pos_[j]);
                      info << buf;
                    }
                  snprintf(buf, sizeof(buf), " |norm %.3f", joint_error_);
                  info << buf;
                }
              else
                {
                  info << " n/a";
                }

              double gimbal_err_sq = 0.0;
              info << "\n  gimbal err:";
              for (int j = 0; j < gimbal_size_; ++j)
                {
                  const double err = target_gimbal_[j] - gimbal_pos_[j];
                  gimbal_err_sq += err * err;
                  snprintf(buf, sizeof(buf), " %s %+.3f",
                           gimbalShortName(gimbal_names_[j]).c_str(), err);
                  info << buf;
                }
              snprintf(buf, sizeof(buf), " |norm %.3f", std::sqrt(gimbal_err_sq));
              info << buf;

              info << "\n  rotor:";
              for (int i = 0; i < static_cast<int>(thrust_cmd_.base_thrust.size()); ++i)
                {
                  snprintf(buf, sizeof(buf), " R%d %.2f", i + 1, thrust_cmd_.base_thrust[i]);
                  info << buf;
                }
            }

          ROS_INFO("%s", info.str().c_str());
        }
      infer_count_ = 0;
      infer_total_ns_ = 0;
      infer_max_ns_ = 0;
      infer_min_ns_ = std::numeric_limits<uint64_t>::max();
      last_infer_report_time_ = now;
    }
  return ControlBase::update();
}

void DragonTrackerRLAgent::reset()
{
  ControlBase::reset();
  std::lock_guard<std::mutex> lock(data_mutex_);
  step_count_ = 0;
  tracking_ = false;
  catch_obs_ = false;
  last_dof_pos_initialized_ = false;
  target_joint_list_.clear();
  target_gimbal_list_.clear();
  target_thrust_list_.clear();

  // re-seed the action so the first commands hold the measured configuration with zero
  // thrust (RSI-style consistency: action = (state - default) / scale)
  seedActionFromState();

  last_infer_report_time_ = ros::Time::now();
  if (verbose_) ROS_INFO("[RL-Tracker] Reset RL agent.");
}

void DragonTrackerRLAgent::seedActionFromState()
{
  action_.assign(action_size_, 0.0f);
  if (joint_catch_)
    for (int i = 0; i < joint_size_; ++i)
      action_[i] = (joint_pos_[i] - joint_default_pos_[i]) / scales_["joint_act"];
  if (gimbal_catch_)
    for (int i = 0; i < gimbal_size_; ++i)
      action_[joint_size_ + i] = (gimbal_pos_[i] - gimbal_default_pos_[i]) / scales_["gimbal_act"];
  for (int i = 0; i < pair_size_; ++i)
    action_[joint_size_ + gimbal_size_ + i] = -thrust_hover_ / scales_["thrust_act"];
  last_action_ = action_;
  action_seeded_ = joint_catch_ && gimbal_catch_;
}

bool DragonTrackerRLAgent::usesIdealObservationSource() const
{
  return ideal_obs_ &&
         (ideal_pos_obs_ || ideal_quat_obs_ || ideal_lin_vel_obs_ || ideal_ang_vel_obs_);
}

void DragonTrackerRLAgent::configureIdealObservationSubscriber()
{
  if (!usesIdealObservationSource())
    {
      if (ideal_odom_subscribed_) odom_sub_.shutdown();
      ideal_odom_subscribed_ = false;
      active_ideal_obs_topic_.clear();
      odom_catch_ = false;
      lin_vel_list_.clear();
      ang_vel_list_.clear();
      return;
    }

  if (ideal_obs_topic_.empty())
    {
      if (ideal_odom_subscribed_) odom_sub_.shutdown();
      ideal_odom_subscribed_ = false;
      active_ideal_obs_topic_.clear();
      odom_catch_ = false;
      lin_vel_list_.clear();
      ang_vel_list_.clear();
      ROS_ERROR("[RL-Tracker] ideal_obs is enabled, but rlagent/ideal_obs_topic is empty!");
      return;
    }

  if (!ideal_odom_subscribed_ || active_ideal_obs_topic_ != ideal_obs_topic_)
    {
      odom_sub_.shutdown();
      odom_sub_ = nh_.subscribe(ideal_obs_topic_, 1, &DragonTrackerRLAgent::odomCallback, this);
      ideal_odom_subscribed_ = true;
      active_ideal_obs_topic_ = ideal_obs_topic_;
      odom_catch_ = false;
      lin_vel_list_.clear();
      ang_vel_list_.clear();
      ROS_INFO("[RL-Tracker] Ideal observation source: %s "
               "(pos=%d quat=%d lin_vel=%d ang_vel=%d delay=%d)",
               ideal_obs_topic_.c_str(), ideal_pos_obs_, ideal_quat_obs_,
               ideal_lin_vel_obs_, ideal_ang_vel_obs_, ideal_delay_);
    }
}

void DragonTrackerRLAgent::buildObservation()
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  if (!joint_catch_ || !gimbal_catch_)
    {
      ROS_WARN_THROTTLE(2.0, "[RL-Tracker] No joint/gimbal state received yet!");
      catch_obs_ = false;
      return;
    }
  const bool use_ideal_obs = usesIdealObservationSource();
  if (use_ideal_obs && !odom_catch_)
    {
      ROS_WARN_THROTTLE(2.0, "[RL-Tracker] No odometry received yet!");
      catch_obs_ = false;
      return;
    }
  if (!target_state_catch_ || target_frames_.empty())
    {
      ROS_WARN_THROTTLE(2.0, "[RL-Tracker] No target state received yet!");
      catch_obs_ = false;
      return;
    }

  // On the first valid frame (and after reset/config enable), use the current
  // positions as the previous positions so the replacement slots do not start
  // with an artificial all-zero sample.
  if (replace_dof_vel_with_last_pos_ && !last_dof_pos_initialized_)
    {
      last_joint_pos_ = joint_pos_;
      last_gimbal_pos_ = gimbal_pos_;
      last_dof_pos_initialized_ = true;
    }

  // ---- fc pose / twist (dragon baselink == fc, the training IMU link) ----
  fc_pos_ = estimator_->getPos(Frame::BASELINK, estimate_mode_);
  tf::Matrix3x3 estimator_rot = estimator_->getOrientation(Frame::BASELINK, estimate_mode_);
  estimator_rot.getRotation(fc_quat_);
  tf::Vector3 lin_vel_world = estimator_->getVel(Frame::BASELINK, estimate_mode_);
  ang_vel_body_ = estimator_->getAngularVel(Frame::BASELINK, estimate_mode_);

  if (use_ideal_obs)
    {
      tf::Vector3 ideal_pos(odom_msg_.pose.pose.position.x, odom_msg_.pose.pose.position.y,
                            odom_msg_.pose.pose.position.z);
      tf::Quaternion ideal_quat(odom_msg_.pose.pose.orientation.x, odom_msg_.pose.pose.orientation.y,
                                odom_msg_.pose.pose.orientation.z, odom_msg_.pose.pose.orientation.w);
      // dragon odom convention: twist.linear world frame, twist.angular body frame
      tf::Vector3 ideal_lin_vel_world(odom_msg_.twist.twist.linear.x, odom_msg_.twist.twist.linear.y,
                                      odom_msg_.twist.twist.linear.z);
      tf::Vector3 ideal_ang_vel_body(odom_msg_.twist.twist.angular.x, odom_msg_.twist.twist.angular.y,
                                     odom_msg_.twist.twist.angular.z);
      if (ideal_delay_ > 0 && (ideal_lin_vel_obs_ || ideal_ang_vel_obs_))
        {
          lin_vel_list_.push_back(ideal_lin_vel_world);
          ang_vel_list_.push_back(ideal_ang_vel_body);
          ideal_lin_vel_world = lin_vel_list_.front();
          ideal_ang_vel_body = ang_vel_list_.front();
          if (static_cast<int>(lin_vel_list_.size()) > ideal_delay_ + 1) lin_vel_list_.pop_front();
          if (static_cast<int>(ang_vel_list_.size()) > ideal_delay_ + 1) ang_vel_list_.pop_front();
        }

      if (ideal_pos_obs_) fc_pos_ = ideal_pos;
      if (ideal_quat_obs_) fc_quat_ = ideal_quat;
      if (ideal_lin_vel_obs_) lin_vel_world = ideal_lin_vel_world;
      if (ideal_ang_vel_obs_) ang_vel_body_ = ideal_ang_vel_body;
    }
  fc_quat_.normalize();
  lin_vel_body_ = rotateByQuatInv(fc_quat_, lin_vel_world);

  // ---- assemble observation according to the YAML term list ----
  std::vector<float> obs;
  obs.reserve(obs_size_);
  if (!history_terms_.empty())
    {
      std::vector<float> current_history_obs;
      current_history_obs.reserve(single_obs_size_);
      for (const auto& term : history_terms_) appendObsTerm(term, current_history_obs);
      for (int k = history_length_ - 1; k > 0; --k)
        history_observations_[k] = history_observations_[k - 1];
      history_observations_[0] = current_history_obs;
      for (const auto& history_obs : history_observations_)
        obs.insert(obs.end(), history_obs.begin(), history_obs.end());
      for (const auto& term : obs_terms_)
        if (std::find(history_terms_.begin(), history_terms_.end(), term) == history_terms_.end())
          appendObsTerm(term, obs);
    }
  else
    {
      for (const auto& term : obs_terms_) appendObsTerm(term, obs);
    }

  if (obs.size() != obs_size_)
    {
      ROS_ERROR_THROTTLE(2.0, "[RL-Tracker] Built observation size (%zu) != model input (%zu)!",
                         obs.size(), obs_size_);
      catch_obs_ = false;
      return;
    }
  observation_ = std::move(obs);
  catch_obs_ = true;

  // Advance the position history only after a complete observation has been
  // accepted. Thus every replacement slot refers to the preceding policy step,
  // independent of the joint-state callback rate and controller decimation.
  if (replace_dof_vel_with_last_pos_)
    {
      last_joint_pos_ = joint_pos_;
      last_gimbal_pos_ = gimbal_pos_;
    }

  // debug tracking errors against the first look-ahead frame
  const RefFrame& f0 = target_frames_[0];
  pos_error_ = f0.pos - fc_pos_;
  double target_roll = 0.0, target_pitch = 0.0, target_yaw = 0.0;
  double current_roll = 0.0, current_pitch = 0.0, current_yaw = 0.0;
  tf::Matrix3x3(f0.quat).getRPY(target_roll, target_pitch, target_yaw);
  tf::Matrix3x3(fc_quat_).getRPY(current_roll, current_pitch, current_yaw);
  rpy_error_ = tf::Vector3(normalizeAngle(target_roll - current_roll),
                           normalizeAngle(target_pitch - current_pitch),
                           normalizeAngle(target_yaw - current_yaw));
  yaw_error_ = rpy_error_.z();
  double joint_err_sq = 0.0;
  for (int j = 0; j < joint_size_; ++j)
    {
      double e = f0.joint_pos[j] - joint_pos_[j];
      joint_err_sq += e * e;
    }
  joint_error_ = std::sqrt(joint_err_sq);
}

void DragonTrackerRLAgent::appendObsTerm(const std::string& term, std::vector<float>& out)
{
  if (term == "target_obs_history" || term == "target_obs")
    {
      for (const auto& f : target_frames_)
        {
          tf::Vector3 rel_pos = rotateByQuatInv(fc_quat_, f.pos - fc_pos_);
          tf::Quaternion rel_quat = (fc_quat_.inverse() * f.quat).normalized();
          tf::Matrix3x3 rel_rot(rel_quat);
          for (const auto& target_term : target_obs_terms_)
            {
              if (target_term == "root_pos")
                {
                  out.push_back(rel_pos.x());
                  out.push_back(rel_pos.y());
                  out.push_back(rel_pos.z());
                }
              else if (target_term == "root_rot_vec" || target_term == "root_quat")
                {
                  for (int r = 0; r < 2; ++r)
                    for (int c = 0; c < 3; ++c) out.push_back(rel_rot[r][c]);
                }
              else if (target_term == "joint_pos" || target_term == "joint")
                {
                  for (int j = 0; j < joint_size_; ++j)
                    out.push_back(f.joint_pos[j] - joint_default_pos_[j]);
                }
              else
                {
                  ROS_WARN_THROTTLE(5.0, "[RL-Tracker] Unknown target_obs term '%s', skipping.",
                                    target_term.c_str());
                }
            }
        }
  }
  else if (term == "joint_pos")
    {
      for (int i = 0; i < joint_size_; ++i) out.push_back(joint_pos_[i] - joint_default_pos_[i]);
    }
  else if (term == "joint_vel")
    {
      if (replace_dof_vel_with_last_pos_)
        for (int i = 0; i < joint_size_; ++i)
          out.push_back(joint_pos_[i] -last_joint_pos_[i]);
      else
        for (int i = 0; i < joint_size_; ++i)
          out.push_back(joint_vel_[i] * scales_["dof_vel"]);
    }
  else if (term == "gimbal_pos")
    {
      for (int i = 0; i < gimbal_size_; ++i) out.push_back(gimbal_pos_[i] - gimbal_default_pos_[i]);
    }
  else if (term == "gimbal_vel")
    {
      if (replace_dof_vel_with_last_pos_)
        for (int i = 0; i < gimbal_size_; ++i)
          out.push_back(gimbal_pos_[i] - last_gimbal_pos_[i]);
      else
        for (int i = 0; i < gimbal_size_; ++i)
          out.push_back(gimbal_vel_[i] * scales_["dof_vel"]);
    }
  else if (term == "root_rot_vec")
    {
      tf::Matrix3x3 root_rot(fc_quat_);
      for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c) out.push_back(root_rot[r][c]);
    }
  else if (term == "lin_vel")
    {
      out.push_back(lin_vel_body_.x() * scales_["lin_vel"]);
      out.push_back(lin_vel_body_.y() * scales_["lin_vel"]);
      out.push_back(lin_vel_body_.z() * scales_["lin_vel"]);
    }
  else if (term == "ang_vel")
    {
      out.push_back(ang_vel_body_.x() * scales_["ang_vel"]);
      out.push_back(ang_vel_body_.y() * scales_["ang_vel"]);
      out.push_back(ang_vel_body_.z() * scales_["ang_vel"]);
    }
  else if (term == "last_action")
    {
      out.insert(out.end(), last_action_.begin(), last_action_.end());
    }
  else
    {
      ROS_WARN_THROTTLE(5.0, "[RL-Tracker] Unknown observation term '%s', skipping.", term.c_str());
    }
}

size_t DragonTrackerRLAgent::obsTermDim(const std::string& term) const
{
  if (term == "target_obs_history" || term == "target_obs")
    {
      size_t per_frame = 0;
      for (const auto& target_term : target_obs_terms_)
        {
          if (target_term == "root_pos") per_frame += 3;
          else if (target_term == "root_rot_vec" || target_term == "root_quat") per_frame += 6;
          else if (target_term == "joint_pos" || target_term == "joint") per_frame += static_cast<size_t>(joint_size_);
          else ROS_WARN("[RL-Tracker] Unknown target_obs term '%s' when computing dimension.",
                        target_term.c_str());
        }
      return static_cast<size_t>(target_obs_horizon_) * per_frame;
    }
  if (term == "joint_pos") return static_cast<size_t>(joint_size_);
  if (term == "joint_vel") return static_cast<size_t>(joint_size_);
  if (term == "gimbal_pos") return static_cast<size_t>(gimbal_size_);
  if (term == "gimbal_vel") return static_cast<size_t>(gimbal_size_);
  if (term == "root_rot_vec") return 6;
  if (term == "lin_vel") return 3;
  if (term == "ang_vel") return 3;
  if (term == "last_action") return action_size_;
  ROS_WARN("[RL-Tracker] Unknown observation term '%s' when computing dimension.", term.c_str());
  return 0;
}

size_t DragonTrackerRLAgent::actionGroupDim(const std::string& group) const
{
  if (group == "joint") return static_cast<size_t>(joint_size_);
  if (group == "gimbal") return static_cast<size_t>(gimbal_size_);
  if (group == "thrust") return static_cast<size_t>(pair_size_);
  ROS_WARN("[RL-Tracker] Unknown action group '%s' when computing dimension.", group.c_str());
  return 0;
}

bool DragonTrackerRLAgent::buildGimbalPositionLimits(
  double roll_limit, double pitch_limit, std::vector<double>& limits) const
{
  if (!std::isfinite(roll_limit) || roll_limit < 0.0 ||
      !std::isfinite(pitch_limit) || pitch_limit < 0.0)
    {
      ROS_ERROR("[RL-Tracker] Gimbal position limits must be finite and non-negative "
                "(roll=%g, pitch=%g).", roll_limit, pitch_limit);
      return false;
    }

  std::vector<double> mapped_limits;
  mapped_limits.reserve(gimbal_names_.size());
  for (const auto& name : gimbal_names_)
    {
      const bool is_roll = name.find("roll") != std::string::npos;
      const bool is_pitch = name.find("pitch") != std::string::npos;
      if (is_roll == is_pitch)
        {
          ROS_ERROR("[RL-Tracker] Cannot select roll/pitch limit for gimbal '%s'.",
                    name.c_str());
          return false;
        }
      mapped_limits.push_back(is_roll ? roll_limit : pitch_limit);
    }

  limits = std::move(mapped_limits);
  return true;
}

bool DragonTrackerRLAgent::loadEffortControlParams(const std::string& group_name,
                                                   const std::vector<std::string>& names,
                                                   std::vector<double>& kp,
                                                   std::vector<double>& kd,
                                                   std::vector<double>& effort_limit,
                                                   std::vector<ros::Publisher>& effort_pubs)
{
  XmlRpc::XmlRpcValue group_xml;
  if (!nh_.getParam("servo_controller/" + group_name, group_xml) ||
      group_xml.getType() != XmlRpc::XmlRpcValue::TypeStruct)
    {
      ROS_ERROR("[RL-Tracker] Missing servo_controller/%s parameters for effort control.",
                group_name.c_str());
      return false;
    }

  if (!group_xml.hasMember("simulation") ||
      group_xml["simulation"].getType() != XmlRpc::XmlRpcValue::TypeStruct)
    {
      ROS_ERROR("[RL-Tracker] Missing servo_controller/%s/simulation parameters "
                "for effort control.", group_name.c_str());
      return false;
    }
  XmlRpc::XmlRpcValue& group_sim = group_xml["simulation"];
  for (const auto& key : {"mit_kp_si", "mit_kd_si", "effort_limit"})
    if (!group_sim.hasMember(key))
      {
        ROS_ERROR("[RL-Tracker] Missing servo_controller/%s/simulation/%s for effort control.",
                  group_name.c_str(), key);
        return false;
      }

  const double default_kp = getXmlRpcDouble(group_sim, "mit_kp_si", 0.0);
  const double default_kd = getXmlRpcDouble(group_sim, "mit_kd_si", 0.0);
  const double default_limit = getXmlRpcDouble(group_sim, "effort_limit", 0.0);

  kp.assign(names.size(), default_kp);
  kd.assign(names.size(), default_kd);
  effort_limit.assign(names.size(), default_limit);
  effort_pubs.resize(names.size());
  std::vector<bool> found(names.size(), false);

  for (auto it = group_xml.begin(); it != group_xml.end(); ++it)
    {
      const std::string controller_key = static_cast<std::string>(it->first);
      if (!isControllerKey(controller_key)) continue;
      XmlRpc::XmlRpcValue& ctrl = it->second;
      if (!ctrl.hasMember("name")) continue;
      const std::string joint_name = static_cast<std::string>(ctrl["name"]);
      auto name_it = std::find(names.begin(), names.end(), joint_name);
      if (name_it == names.end()) continue;
      const size_t idx = static_cast<size_t>(std::distance(names.begin(), name_it));

      if (ctrl.hasMember("simulation"))
        {
          XmlRpc::XmlRpcValue& sim = ctrl["simulation"];
          kp[idx] = getXmlRpcDouble(sim, "mit_kp_si", kp[idx]);
          kd[idx] = getXmlRpcDouble(sim, "mit_kd_si", kd[idx]);
          effort_limit[idx] = getXmlRpcDouble(sim, "effort_limit", effort_limit[idx]);
        }

      const std::string topic = "servo_controller/" + group_name + "/" +
                                controller_key + "/simulation/command";
      effort_pubs[idx] = nh_.advertise<std_msgs::Float64>(topic, 1);
      found[idx] = true;
      ROS_INFO("[RL-Tracker] %s effort controller: %s -> %s (kp=%.3f kd=%.3f limit=%.3f)",
               group_name.c_str(), joint_name.c_str(), topic.c_str(),
               kp[idx], kd[idx], effort_limit[idx]);
    }

  bool ok = true;
  for (size_t i = 0; i < names.size(); ++i)
    {
      if (!found[i])
        {
          ROS_ERROR("[RL-Tracker] Could not find effort controller for %s/%s",
                    group_name.c_str(), names[i].c_str());
          ok = false;
        }
    }
  return ok;
}

void DragonTrackerRLAgent::policyForward()
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  if (!catch_obs_)
    {
      ROS_WARN_THROTTLE(2.0, "[RL-Tracker] No valid observation yet, skip policyForward()");
      return;
    }
  const float clip_obs = static_cast<float>(scales_["clip_observation"]);
  for (auto& v : observation_)
    {
      if (!std::isfinite(v)) v = 0.0f;
      v = std::min(std::max(v, -clip_obs), clip_obs);
    }

  std::vector<int64_t> input_shape = {1, static_cast<int64_t>(observation_.size())};
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      mem_info, observation_.data(), observation_.size(), input_shape.data(), input_shape.size());

  const char* input_names[] = {input_name_};
  const char* output_names[] = {output_name_};
  Ort::RunOptions run_options;
  run_options.SetRunLogVerbosityLevel(0);

  auto t0 = std::chrono::high_resolution_clock::now();
  auto output_tensors = session_->Run(run_options, input_names, &input_tensor, 1, output_names, 1);
  auto t1 = std::chrono::high_resolution_clock::now();
  uint64_t infer_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  infer_count_ += 1;
  infer_total_ns_ += infer_ns;
  infer_max_ns_ = std::max(infer_max_ns_, infer_ns);
  infer_min_ns_ = std::min(infer_min_ns_, infer_ns);

  float* out_ptr = output_tensors.front().GetTensorMutableData<float>();
  size_t out_len = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
  if (out_len != action_size_)
    {
      ROS_ERROR_THROTTLE(2.0, "[RL-Tracker] ONNX output length (%zu) != action size (%zu)!",
                         out_len, action_size_);
      return;
    }
  bool out_bad = false;
  for (size_t i = 0; i < out_len; ++i)
    if (!std::isfinite(out_ptr[i])) { out_ptr[i] = 0.0f; out_bad = true; }
  if (out_bad)
    ROS_ERROR_THROTTLE(1.0, "[RL-Tracker] Model output contained non-finite values (zeroed)!");

  const float clip_act = static_cast<float>(scales_["clip_action"]);
  for (size_t i = 0; i < action_size_; ++i)
    action_[i] = std::min(std::max(out_ptr[i], -clip_act), clip_act);
  catch_obs_ = false;
}

void DragonTrackerRLAgent::applyJointAction(const float* action_slice)
{
  for (int i = 0; i < joint_size_; ++i)
    {
      double target = action_slice[i] * scales_["joint_act"] + joint_default_pos_[i];
      target_joint_[i] = action_clamp_
        ? clampDouble(target, -joint_limit_, joint_limit_)
        : target;
    }
  if (joint_target_delay_steps_ > 0)
    {
      target_joint_list_.push_back(target_joint_);
      target_joint_ = target_joint_list_.front();
      if (static_cast<int>(target_joint_list_.size()) > joint_target_delay_steps_ + 1)
        target_joint_list_.pop_front();
    }
  // Leave effort empty in position mode: servo_bridge forwards any non-empty
  // effort as goal current, which caps CbP servos at zero torque.
  joint_cmd_.effort.resize(joint_effort_ctrl_ ? joint_size_ : 0, 0.0);
  for (int i = 0; i < joint_size_; ++i)
    {
      joint_cmd_.position[i] = target_joint_[i];
      if (joint_effort_ctrl_)
        {
          double effort = joint_kp_[i] * (target_joint_[i] - joint_pos_[i]) -
                          joint_kd_[i] * joint_vel_[i];
          if (!std::isfinite(effort)) effort = 0.0;
          if (std::isfinite(joint_effort_limit_[i]))
            effort = clampDouble(effort, -joint_effort_limit_[i], joint_effort_limit_[i]);
          joint_cmd_.effort[i] = static_cast<float>(effort);
        }
    }
}

void DragonTrackerRLAgent::applyGimbalAction(const float* action_slice)
{
  for (int i = 0; i < gimbal_size_; ++i)
    {
      double target = action_slice[i] * scales_["gimbal_act"] + gimbal_default_pos_[i];
      const double limit = gimbal_position_limits_[i];
      target_gimbal_[i] = action_clamp_
        ? clampDouble(target, -limit, limit)
        : target;
    }
  if (gimbal_target_delay_steps_ > 0)
    {
      target_gimbal_list_.push_back(target_gimbal_);
      target_gimbal_ = target_gimbal_list_.front();
      if (static_cast<int>(target_gimbal_list_.size()) > gimbal_target_delay_steps_ + 1)
        target_gimbal_list_.pop_front();
    }
  // Leave effort empty in position mode: servo_bridge forwards any non-empty
  // effort as goal current, which caps CbP servos at zero torque.
  gimbal_cmd_.effort.resize(gimbal_effort_ctrl_ ? gimbal_size_ : 0, 0.0);
  for (int i = 0; i < gimbal_size_; ++i)
    {
      gimbal_cmd_.position[i] = target_gimbal_[i];
      if (gimbal_effort_ctrl_)
        {
          double effort = gimbal_kp_[i] * (target_gimbal_[i] - gimbal_pos_[i]) -
                          gimbal_kd_[i] * gimbal_vel_[i];
          if (!std::isfinite(effort)) effort = 0.0;
          if (std::isfinite(gimbal_effort_limit_[i]))
            effort = clampDouble(effort, -gimbal_effort_limit_[i], gimbal_effort_limit_[i]);
          gimbal_cmd_.effort[i] = static_cast<float>(effort);
        }
    }
}

void DragonTrackerRLAgent::applyThrustAction(const float* action_slice)
{
  for (int i = 0; i < pair_size_; ++i)
    {
      double per_rotor_thrust = action_slice[i] * scales_["thrust_act"] + thrust_hover_;
      if (!std::isfinite(per_rotor_thrust)) per_rotor_thrust = 0.0;
      per_rotor_thrust = clampDouble(per_rotor_thrust, 0.0, thrust_limit_);
      target_thrust_[i] = per_rotor_thrust * thrust_scale_default_;
    }
  if (thrust_target_delay_steps_ > 0)
    {
      target_thrust_list_.push_back(target_thrust_);
      target_thrust_ = target_thrust_list_.front();
      if (static_cast<int>(target_thrust_list_.size()) > thrust_target_delay_steps_ + 1)
        target_thrust_list_.pop_front();
    }
  for (int i = 0; i < pair_size_; ++i)
    {
      // Training action is per-rotor thrust. The JSK/DRAGON base_thrust command
      // addresses one dual-EDF module, so send the paired total force.
      thrust_cmd_.base_thrust[pair_module_index_[i]] = 2.0f * target_thrust_[i];
    }
}

void DragonTrackerRLAgent::sendCmd()
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  last_action_ = action_;

  ros::Time now = ros::Time::now();
  joint_cmd_.header.stamp = now;
  gimbal_cmd_.header.stamp = now;

  size_t action_offset = 0;
  for (const auto& group : action_groups_)
    {
      if (group == "joint")
        {
          applyJointAction(&action_[action_offset]);
          action_offset += static_cast<size_t>(joint_size_);
        }
      else if (group == "gimbal")
        {
          applyGimbalAction(&action_[action_offset]);
          action_offset += static_cast<size_t>(gimbal_size_);
        }
      else if (group == "thrust")
        {
          applyThrustAction(&action_[action_offset]);
          action_offset += static_cast<size_t>(pair_size_);
        }
      else
        {
          ROS_WARN_THROTTLE(5.0, "[RL-Tracker] Unknown action group '%s', skipping.",
                            group.c_str());
        }
    }
  if (action_offset != action_size_)
    ROS_WARN_THROTTLE(5.0, "[RL-Tracker] Parsed action size (%zu) != ONNX output size (%zu)",
                      action_offset, action_size_);

  // `enable_gimbal_` intentionally gates thrust as a hardware safety interlock:
  // remote should enable joints/gimbals first, then arm, then takeoff.
  const bool control_started = control_timestamp_ > 0.0;
  const bool joint_ready = enable_joint_ && action_seeded_;
  const bool gimbal_ready = enable_gimbal_ && action_seeded_;
  const bool thrust_ready = enable_thrust_ && control_started && action_seeded_ && enable_gimbal_;

  if (joint_ready)
    {
      if (joint_effort_ctrl_)
        {
          for (int i = 0; i < joint_size_; ++i)
            {
              std_msgs::Float64 msg;
              msg.data = joint_cmd_.effort[i];
              joint_effort_pubs_[i].publish(msg);
            }
        }
      else
        {
          joint_pub_.publish(joint_cmd_);
        }
    }
  if (gimbal_ready)
    {
      if (gimbal_effort_ctrl_)
        {
          for (int i = 0; i < gimbal_size_; ++i)
            {
              std_msgs::Float64 msg;
              msg.data = gimbal_cmd_.effort[i];
              gimbal_effort_pubs_[i].publish(msg);
            }
        }
      else
        {
          gimbal_pub_.publish(gimbal_cmd_);
        }
    }
  if (thrust_ready)
    {
      thrust_pub_.publish(thrust_cmd_);
    }
  else if (enable_thrust_)
    {
      ROS_INFO_THROTTLE(2.0,
                        "[RL-Tracker] Thrust gated: navi_state=%d control_started=%d "
                        "control_timestamp=%.3f action_seeded=%d enable_gimbal=%d",
                        static_cast<int>(navigator_->getNaviState()),
                        control_started, control_timestamp_, action_seeded_, enable_gimbal_);
    }

  if (debug_) publishDebug();
}

void DragonTrackerRLAgent::publishDebug()
{
  joint_debug_pub_.publish(joint_cmd_);
  gimbal_debug_pub_.publish(gimbal_cmd_);
  thrust_debug_pub_.publish(thrust_cmd_);

  std_msgs::Float32MultiArray obs_msg;
  obs_msg.data = observation_;
  obs_debug_pub_.publish(obs_msg);
}

// ================================ callbacks ==================================

void DragonTrackerRLAgent::jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  const bool has_velocity = msg->velocity.size() == msg->name.size();
  double dt = -1.0;
  if (!has_velocity && joint_state_stamp_.toSec() > 0.0)
    dt = (msg->header.stamp - joint_state_stamp_).toSec();

  bool joint_seen = false, gimbal_seen = false;
  for (size_t i = 0; i < msg->name.size(); ++i)
    {
      if (i >= msg->position.size()) break;
      for (int j = 0; j < joint_size_; ++j)
        {
          if (msg->name[i] != joint_names_[j]) continue;
          if (has_velocity) joint_vel_[j] = msg->velocity[i];
          else if (dt > 1e-6) joint_vel_[j] = (msg->position[i] - joint_pos_[j]) / dt;
          joint_pos_[j] = msg->position[i];
          joint_seen = true;
        }
      for (int j = 0; j < gimbal_size_; ++j)
        {
          if (msg->name[i] != gimbal_names_[j]) continue;
          if (has_velocity) gimbal_vel_[j] = msg->velocity[i];
          else if (dt > 1e-6) gimbal_vel_[j] = (msg->position[i] - gimbal_pos_[j]) / dt;
          gimbal_pos_[j] = msg->position[i];
          gimbal_seen = true;
        }
    }
  joint_state_stamp_ = msg->header.stamp;
  if (joint_seen) joint_catch_ = true;
  if (gimbal_seen) gimbal_catch_ = true;
  // seed the action from the first full measured state; it only serves as a fallback
  // command until the policy starts forwarding
  if (!action_seeded_ && joint_catch_ && gimbal_catch_) seedActionFromState();
}

void DragonTrackerRLAgent::odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  odom_msg_ = *msg;
  odom_catch_ = true;
}

void DragonTrackerRLAgent::enableJointAndGimbalCallback(
  const std_msgs::Empty::ConstPtr& /*msg*/)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  enable_joint_ = true;
  enable_gimbal_ = true;
  ROS_INFO("[RL-Tracker] Joint and gimbal commands enabled by teleop.");
}

void DragonTrackerRLAgent::forceLandingCallback(const std_msgs::Empty::ConstPtr& /*msg*/)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  enable_joint_ = false;
  enable_gimbal_ = false;
  ROS_WARN("[RL-Tracker] Force landing: joint/gimbal commands disabled.");
}

void DragonTrackerRLAgent::haltCallback(const std_msgs::Empty::ConstPtr& /*msg*/)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  enable_joint_ = false;
  enable_gimbal_ = false;
  ROS_WARN("[RL-Tracker] Halt: joint/gimbal commands disabled.");
}

void DragonTrackerRLAgent::targetStateCallback(const aerial_robot_msgs::FullStateTarget::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);

  RefFrame frame;
  frame.pos = tf::Vector3(msg->root_state.pose.pose.position.x,
                          msg->root_state.pose.pose.position.y,
                          msg->root_state.pose.pose.position.z);
  frame.quat = tf::Quaternion(msg->root_state.pose.pose.orientation.x,
                              msg->root_state.pose.pose.orientation.y,
                              msg->root_state.pose.pose.orientation.z,
                              msg->root_state.pose.pose.orientation.w);
  if (frame.quat.length2() < 1e-12)
    frame.quat = tf::Quaternion(0.0, 0.0, 0.0, 1.0);
  else
    frame.quat.normalize();

  frame.joint_pos.assign(joint_size_, 0.0);
  if (!msg->joint_state.name.empty())
    {
      for (int j = 0; j < joint_size_; ++j)
        {
          auto it = std::find(msg->joint_state.name.begin(), msg->joint_state.name.end(), joint_names_[j]);
          if (it == msg->joint_state.name.end())
            {
              ROS_WARN_THROTTLE(2.0, "[RL-Tracker] FullStateTarget lacks joint '%s'",
                                joint_names_[j].c_str());
              frame.joint_pos[j] = joint_default_pos_[j];
              continue;
            }
          const size_t idx = static_cast<size_t>(std::distance(msg->joint_state.name.begin(), it));
          frame.joint_pos[j] = idx < msg->joint_state.position.size() ?
              msg->joint_state.position[idx] : joint_default_pos_[j];
        }
    }
  else
    {
      for (int j = 0; j < joint_size_; ++j)
        frame.joint_pos[j] = j < static_cast<int>(msg->joint_state.position.size()) ?
            msg->joint_state.position[j] : joint_default_pos_[j];
      if (msg->joint_state.position.size() < static_cast<size_t>(joint_size_))
        ROS_WARN_THROTTLE(2.0, "[RL-Tracker] FullStateTarget joint_state has %zu positions, need %d.",
                          msg->joint_state.position.size(), joint_size_);
    }

  target_state_fifo_.push_back(frame);
  while (static_cast<int>(target_state_fifo_.size()) > target_obs_horizon_)
    target_state_fifo_.pop_front();
  target_state_stamp_ = msg->header.stamp;
  target_state_catch_ = true;
}


bool DragonTrackerRLAgent::reloadPolicyCallback(aerial_robot_model::ReloadPolicy::Request& req,
                                                aerial_robot_model::ReloadPolicy::Response& res)
{
  if (req.policy_name.empty())
    {
      res.success = false;
      res.message = "policy_name is empty";
      return true;
    }
  std::string model_path = resolvePackagePath(req.policy_name);
  std::ifstream f(model_path.c_str());
  if (!f.good())
    {
      res.success = false;
      res.message = "Policy file not found: " + model_path;
      ROS_WARN("[RL-Tracker] %s", res.message.c_str());
      return true;
    }

  std::lock_guard<std::mutex> lock(data_mutex_);
  size_t old_obs_size = obs_size_, old_action_size = action_size_;
  if (input_name_) { allocator_.Free(const_cast<char*>(input_name_)); input_name_ = nullptr; }
  if (output_name_) { allocator_.Free(const_cast<char*>(output_name_)); output_name_ = nullptr; }
  session_.reset();
  try
    {
      initRLAgent(model_path);
    }
  catch (const std::exception& e)
    {
      res.success = false;
      res.message = std::string("Failed to load policy: ") + e.what();
      ROS_ERROR("[RL-Tracker] %s", res.message.c_str());
      return true;
    }
  if (obs_size_ != old_obs_size || action_size_ != old_action_size)
    {
      res.success = false;
      res.message = "New policy I/O (" + std::to_string(obs_size_) + "/" +
                    std::to_string(action_size_) + ") differs from previous (" +
                    std::to_string(old_obs_size) + "/" + std::to_string(old_action_size) + ")!";
      ROS_ERROR("[RL-Tracker] %s", res.message.c_str());
      return true;
    }

  catch_obs_ = false;
  seedActionFromState();

  res.success = true;
  res.message = "Policy reloaded: " + req.policy_name;
  ROS_INFO("[RL-Tracker] %s", res.message.c_str());
  return true;
}

bool DragonTrackerRLAgent::reloadParamsCallback(std_srvs::Trigger::Request& /*req*/,
                                                std_srvs::Trigger::Response& res)
{
  ros::NodeHandle rl_nh(nh_, "rlagent");
  std::lock_guard<std::mutex> lock(data_mutex_);

  getParam<bool>(rl_nh, "enable_thrust", enable_thrust_, enable_thrust_);
  getParam<bool>(rl_nh, "enable_joint", enable_joint_, enable_joint_);
  getParam<bool>(rl_nh, "enable_gimbal", enable_gimbal_, enable_gimbal_);
  getParam<bool>(rl_nh, "joint_effort_ctrl", joint_effort_ctrl_, joint_effort_ctrl_);
  getParam<bool>(rl_nh, "gimbal_effort_ctrl", gimbal_effort_ctrl_, gimbal_effort_ctrl_);
  bool new_action_clamp = action_clamp_;
  getParam<bool>(rl_nh, "action_clamp", new_action_clamp, new_action_clamp);
  getParam<bool>(rl_nh, "rlagent_verbose", verbose_, verbose_);
  getParam<bool>(rl_nh, "rlagent_debug", debug_, debug_);
  getParam<bool>(rl_nh, "rlagent_forward_info", forward_info_, forward_info_);
  const bool old_replace_dof_vel_with_last_pos = replace_dof_vel_with_last_pos_;
  getParam<bool>(rl_nh, "replace_dof_vel_with_last_pos",
                 replace_dof_vel_with_last_pos_, replace_dof_vel_with_last_pos_);
  if (replace_dof_vel_with_last_pos_ != old_replace_dof_vel_with_last_pos)
    last_dof_pos_initialized_ = false;
  getParam<double>(rl_nh, "thrust_hover", thrust_hover_, thrust_hover_);
  getParam<double>(rl_nh, "thrust_limit", thrust_limit_, thrust_limit_);
  getParam<double>(rl_nh, "joint_limit", joint_limit_, joint_limit_);

  double new_gimbal_roll_limit = gimbal_roll_limit_;
  double new_gimbal_pitch_limit = gimbal_pitch_limit_;
  getParam<double>(rl_nh, "gimbal_roll_limit", new_gimbal_roll_limit,
                   new_gimbal_roll_limit);
  getParam<double>(rl_nh, "gimbal_pitch_limit", new_gimbal_pitch_limit,
                   new_gimbal_pitch_limit);
  std::vector<double> new_gimbal_position_limits;
  if (!buildGimbalPositionLimits(new_gimbal_roll_limit, new_gimbal_pitch_limit,
                                 new_gimbal_position_limits))
    {
      res.success = false;
      res.message = "Invalid gimbal roll/pitch position limits; previous limits retained.";
      return true;
    }
  action_clamp_ = new_action_clamp;
  gimbal_roll_limit_ = new_gimbal_roll_limit;
  gimbal_pitch_limit_ = new_gimbal_pitch_limit;
  gimbal_position_limits_ = std::move(new_gimbal_position_limits);

  double new_thrust_scale = thrust_scale_default_;
  getParam<double>(rl_nh, "thrust_scale", new_thrust_scale, new_thrust_scale);
  thrust_scale_default_ = std::min(new_thrust_scale, 1.0);

  if (joint_effort_ctrl_)
    {
      loadEffortControlParams("joints", joint_names_, joint_kp_, joint_kd_,
                              joint_effort_limit_, joint_effort_pubs_);
    }
  if (gimbal_effort_ctrl_)
    {
      loadEffortControlParams("gimbals", gimbal_names_, gimbal_kp_, gimbal_kd_,
                              gimbal_effort_limit_, gimbal_effort_pubs_);
    }

  res.success = true;
  res.message = "Params reloaded. enable_thrust=" + std::to_string(enable_thrust_) +
                " enable_joint=" + std::to_string(enable_joint_) +
                " enable_gimbal=" + std::to_string(enable_gimbal_) +
                " joint_effort_ctrl=" + std::to_string(joint_effort_ctrl_) +
                " gimbal_effort_ctrl=" + std::to_string(gimbal_effort_ctrl_) +
                " action_clamp=" + std::to_string(action_clamp_) +
                " replace_dof_vel_with_last_pos=" +
                std::to_string(replace_dof_vel_with_last_pos_) +
                " thrust_scale=" + std::to_string(thrust_scale_default_) +
                " joint_limit=" + std::to_string(joint_limit_) +
                " gimbal_roll_limit=" + std::to_string(gimbal_roll_limit_) +
                " gimbal_pitch_limit=" + std::to_string(gimbal_pitch_limit_);
  ROS_INFO("[RL-Tracker] %s", res.message.c_str());
  return true;
}

}  // namespace aerial_robot_control

/* plugin registration */
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(aerial_robot_control::DragonTrackerRLAgent, aerial_robot_control::ControlBase);
