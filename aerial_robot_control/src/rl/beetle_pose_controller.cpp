//
// Created by wentao-zhang on 2025/12/04.
//

#include "aerial_robot_control/rl/beetle_pose_controller.h"

namespace aerial_robot_control
{

BeetlePoseController::BeetlePoseController(): ControlBase()
{
}

void BeetlePoseController::initialize(ros::NodeHandle nh, ros::NodeHandle nhp,
                                      boost::shared_ptr<aerial_robot_model::RobotModel> robot_model,
                                      boost::shared_ptr<aerial_robot_estimation::StateEstimator> estimator,
                                      boost::shared_ptr<aerial_robot_navigation::BaseNavigator> navigator,
                                      double ctrl_loop_du)
{
  ControlBase::initialize(nh, nhp, robot_model, estimator, navigator, ctrl_loop_du);
  // 1. read the ONNX model path from parameter server
  ros::NodeHandle rl_nh(nh_, "rlagent");
  std::string onnx_model_path;
  if (!rl_nh.getParam("onnx_model_path", onnx_model_path))
  {
    ROS_ERROR(
        "Parameter ~onnx_model_path is not loaded. "
        "You must specify the onnx_model_path in the launch file.");
    return;
  }
  if (onnx_model_path.empty())
  {
    ROS_ERROR("The onnx_model_path is empty!");
    return;
  }
  else
  {
    ROS_INFO("ONNX model path: %s", onnx_model_path.c_str());
  }
  // 2. initialize ONNX Runtime
  initRLAgent(onnx_model_path);

  double goal_angle_R, goal_angle_P, goal_angle_Y;
  getParam<double>(rl_nh,"control_freq", control_hz_, 200.0);
  getParam<int>(rl_nh,"decimation", decimation_, 4);
  getParam<bool>(rl_nh,"rlagent_verbose", verbose_, false);
  getParam<int>(rl_nh,"gimbal_target_delay_steps", gimbal_target_delay_steps_, 0);
  getParam<int>(rl_nh,"gimbal_obs_delay_steps", gimbal_obs_delay_steps_, 0);
  getParam<bool>(rl_nh,"enable_thrust", enable_thrust_, false);
  // getParam<bool>(rl_nh,"lock_thrust", lock_thrust_, false);
  getParam<bool>(rl_nh,"enable_gimbal", enable_gimbal_, false);
  // getParam<bool>(rl_nh,"lock_gimbal", lock_gimbal_, false);
  getParam<double>(rl_nh,"goal_pos_x", desired_pose_.pose.position.x, 0.0);
  getParam<double>(rl_nh,"goal_pos_y", desired_pose_.pose.position.y, 0.0);
  getParam<double>(rl_nh,"goal_pos_z", desired_pose_.pose.position.z, 0.5);
  getParam<double>(rl_nh,"goal_angle_R", goal_angle_R, 0.0);
  getParam<double>(rl_nh,"goal_angle_P", goal_angle_P, 0.0);
  getParam<double>(rl_nh,"goal_angle_Y", goal_angle_Y, 0.0);
  getParam<double>(rl_nh,"thrust_default", thrust_default_, 0.0);
  // getParam<int>("gimbal_size", gimbal_size_, robot_model_->getJointNum() - robot_model_->getRotorNum());
  // getParam<int>("thrust_size", thrust_size_, robot_model_->getRotorNum());
  getParam<int>(rl_nh,"gimbal_size", gimbal_size_, 4);
  getParam<int>(rl_nh,"thrust_size", thrust_size_, 4);
  tf::Quaternion quat = tf::createQuaternionFromRPY(goal_angle_R, goal_angle_P, goal_angle_Y);
  desired_pose_.pose.orientation.x = quat.x();
  desired_pose_.pose.orientation.y = quat.y();
  desired_pose_.pose.orientation.z = quat.z();
  desired_pose_.pose.orientation.w = quat.w();

  target_gimbal_.resize(gimbal_size_);
  target_thrust_.resize(thrust_size_);
  // 3. initialize ros 
  goal_sub_   = nh_.subscribe("/desired_3D_pose", 1, &BeetlePoseController::goalCallback, this);
  gimbal_sub_ = nh_.subscribe("joint_states", 1, &BeetlePoseController::gimbalCallback, this);
  thrust_pub_ = nh_.advertise<spinal::FourAxisCommand>("four_axes/command", 1);
  thrust_debug_pub_ = nh_.advertise<spinal::FourAxisCommand>("four_axes/command_debug", 1);
  gimbal_pub_ = nh_.advertise<sensor_msgs::JointState>("gimbals_ctrl", 1);
  gimbal_debug_pub_ = nh_.advertise<sensor_msgs::JointState>("gimbals_ctrl_debug", 1);
  obs_debug_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("observation_debug", 1);

   /* reset control input */
  thrust_cmd_.base_thrust = std::vector<float>(thrust_size_, 0.0);

  if (!gimbal_cmd_.name.empty()) gimbal_cmd_.name.clear();
  if (!gimbal_cmd_.position.empty()) gimbal_cmd_.position.clear();
  for (int i = 0; i < gimbal_size_; i++)
  {
    gimbal_cmd_.name.emplace_back("gimbal" + std::to_string(i + 1));
    gimbal_cmd_.position.push_back(0.0);
  }
}

bool BeetlePoseController::update()
{
  ros::Time now = ros::Time::now();
  step_count_++;
  if (step_count_ % decimation_ == 0)
  {
    buildObservation();
    policyForward();
  }
  sendCmd();
  double elapsed = (now - last_infer_report_time_).toSec();
  if (elapsed >= 2.0 && infer_count_ > 0)
  {
    double avg_ns = static_cast<double>(infer_total_ns_) / static_cast<double>(infer_count_);
    double avg_ms = avg_ns / 1e6;
    double min_ms = static_cast<double>(infer_min_ns_) / 1e6;
    double max_ms = static_cast<double>(infer_max_ns_) / 1e6;
    double avg_period_ms = (elapsed / static_cast<double>(infer_count_)) * 1000.0;
    if (verbose_)
      ROS_INFO(
          "[RL-Agent]\n"
          "ORT Inference (last %.2fs): calls=%lu\n"
          "  avg_time=%.3f ms, avg_period=%.3f ms\n"
          "  min=%.3f ms, max=%.3f ms\n"
          "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
          elapsed, static_cast<unsigned long>(infer_count_), avg_ms, avg_period_ms, min_ms, max_ms);
    // reset counters
    infer_count_ = 0;
    infer_total_ns_ = 0;
    infer_max_ns_ = 0;
    infer_min_ns_ = std::numeric_limits<uint64_t>::max();
    last_infer_report_time_ = now;
  }
  return ControlBase::update();
}

void BeetlePoseController::reset()
{
  ControlBase::reset();
  step_count_ = 0;
  if (verbose_)
    ROS_INFO("[RL-Agent]Reset RL Agent.");
  last_infer_report_time_ = ros::Time::now();
  // Needs to be done: reset the mpc_solver with data
}

BeetlePoseController::~BeetlePoseController()
{
    // free allocated C strings from GetInputNameAllocated / GetOutputNameAllocated
    if (input_name_) allocator_.Free(const_cast<char*>(input_name_));
    if (output_name_) allocator_.Free(const_cast<char*>(output_name_));
}

void BeetlePoseController::policyForward()
{
  // Create input tensor object from data values
  std::vector<int64_t> input_shape = {1, static_cast<int64_t>(obs_.size())};
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, obs_.data(), obs_.size(), input_shape.data(), input_shape.size());

  const char* input_names[] = {input_name_};
  const char* output_names[] = {output_name_};
  
  Ort::RunOptions run_options;
    run_options.SetRunLogVerbosityLevel(0);
    auto infer_t0 = std::chrono::high_resolution_clock::now();
    auto output_tensors = session_->Run(run_options, input_names, &input_tensor, 1, output_names, 1);
    auto infer_t1 = std::chrono::high_resolution_clock::now();
    uint64_t infer_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(infer_t1 - infer_t0).count();
    // update stats
    infer_count_ += 1;
    infer_total_ns_ += infer_ns;
    if (infer_ns > infer_max_ns_) infer_max_ns_ = infer_ns;
    if (infer_ns < infer_min_ns_) infer_min_ns_ = infer_ns;
    // assume single output tensor of shape (1, action_size)
    float* out_ptr = output_tensors.front().GetTensorMutableData<float>();
    size_t out_len = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

    if (out_len == 0) {
      ROS_ERROR("[RL-Agent]ONNX runtime returned zero-length output");
      return;
    }

    // Check for NaN/Inf in output immediately
    bool out_bad = false;
    for (size_t i = 0; i < out_len; ++i) {
      if (!std::isfinite(out_ptr[i])) { out_bad = true; break; }
    }
    if (out_len != action_size_) {
      ROS_WARN("[RL-Agent]ONNX runtime output length (%zu) differs from expected action_size_ (%zu)", out_len, action_size_);
      out_bad = true;
    }
    if (out_bad) {
      std::ostringstream ss;
      ss << "[RL-Agent]Model output contains non-finite values: [";
      for (size_t i = 0; i < std::min<size_t>(out_len, 8); ++i) {
          ss << std::fixed << std::setprecision(6) << out_ptr[i] << (i + 1 < std::min<size_t>(out_len, 8) ? ", " : "");
      }
      ss << (out_len > 8 ? ", ..." : "") << "]";
      ROS_ERROR_STREAM(ss.str());
      // optionally replace non-finite outputs with zeros to avoid downstream crash
      for (size_t i = 0; i < out_len; ++i) if (!std::isfinite(out_ptr[i])) out_ptr[i] = 0.0f;
    }

    // copy to vector
    std::vector<float> action(out_ptr, out_ptr + out_len);
    if (action_size_ != out_len) {
      ROS_WARN("[RL-Agent]Model output length (%zu) differs from stored action_size_ (%zu). Updating action_size_.", out_len, action_size_);
      action_size_ = out_len;
      last_action_.assign(action_size_, 0.0f);
    }
}

void BeetlePoseController::initRLAgent(std::string model_path)
{
    // ONNX Runtime init
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rlagent");
    sess_opts_ = std::make_unique<Ort::SessionOptions>();
    sess_opts_->SetIntraOpNumThreads(2);
    sess_opts_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *sess_opts_);

    // retrieve input/output names (simple, first input/output)
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_->GetInputNameAllocated(0, allocator).release();
    output_name_ = session_->GetOutputNameAllocated(0, allocator).release();

    // Query model input/output shapes and set expected sizes
    {
        auto in_type_info = session_->GetInputTypeInfo(0);
        auto in_tensor_info = in_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> in_dims = in_tensor_info.GetShape();
        if (in_dims.size() == 2) {
            obs_size_ = static_cast<size_t>(in_dims[1]);
        } else if (in_dims.size() == 1) {
            obs_size_ = static_cast<size_t>(in_dims[0]);
        } else {
            ROS_WARN("[RL-Agent]Unexpected model input dims size=%zu, using product", in_dims.size());
            size_t prod = 1;
            for (auto d : in_dims) if (d > 0) prod *= static_cast<size_t>(d);
            obs_size_ = prod;
        }

        auto out_type_info = session_->GetOutputTypeInfo(0);
        auto out_tensor_info = out_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> out_dims = out_tensor_info.GetShape();
        if (out_dims.size() == 2) {
            action_size_ = static_cast<size_t>(out_dims[1]);
        } else if (out_dims.size() == 1) {
            action_size_ = static_cast<size_t>(out_dims[0]);
        } else {
            size_t prod = 1;
            for (auto d : out_dims) if (d > 0) prod *= static_cast<size_t>(d);
            action_size_ = prod;
        }
    }
}

void BeetlePoseController::buildObservation()
{

}

void BeetlePoseController::sendCmd()
{
  if (gimbal_size_ + thrust_size_ != action_size_) {
    if (verbose_)
      ROS_ERROR("[RL-Agent] Gimbal size (%d) + thrust size (%d) != action size (%zu)", gimbal_size_, thrust_size_, action_size_);
    return;
  }
  for (size_t i = 0; i < action_size_; ++i) {
    last_action_[i] = action_[i];
  }
  for (size_t i = 0; i < gimbal_size_; ++i) {
    target_gimbal_[i] = action_[i] * 0.25f + gimbal_default_pos_[i];
    gimbal_cmd_.position[i] = target_gimbal_[i];
  }
  for (size_t i = 0; i < thrust_size_; ++i) {
    target_thrust_[i] = action_[gimbal_size_ + i] * 1.25f + thrust_default_;
    thrust_cmd_.base_thrust[i] = target_thrust_[i];
  }
  gimbal_cmd_.header.stamp = ros::Time::now();
  if (enable_gimbal_)
    gimbal_pub_.publish(gimbal_cmd_);
  if (enable_thrust_)
    thrust_pub_.publish(thrust_cmd_);
}

void BeetlePoseController::goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  desired_pose_ = *msg;
}

void BeetlePoseController::gimbalCallback(const sensor_msgs::JointState::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  for (size_t i=0, j=0; i < msg->name.size(); i++) {
    if (msg->name[i].find("gimbal") != std::string::npos && j < gimbal_size_) {
      gimbal_pos_.at(j) = msg->position[i];
      j++;
    }
  }
  gimbal_catch_ = true;
}



}  // namespace aerial_robot_control

/* plugin registration */
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(aerial_robot_control::BeetlePoseController, aerial_robot_control::ControlBase);