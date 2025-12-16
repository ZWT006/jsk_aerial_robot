//
// Created by wentao-zhang on 2025/12/04.
//

#include "beetle_omni/beetle_pose_rlagent.h"
#include <ros/package.h>
#include <xmlrpcpp/XmlRpcValue.h>

#define PRINT_FREQUENCY 200

geometry_msgs::Transform inverseTransform(const geometry_msgs::Transform &Transform)
{
    geometry_msgs::Transform inverse_transform;
    tf::Quaternion quaternion(Transform.rotation.x, Transform.rotation.y, Transform.rotation.z, Transform.rotation.w);
    tf::Vector3 translation_vector(Transform.translation.x, Transform.translation.y, Transform.translation.z);
    tf::Quaternion inverse_quaternion = quaternion.inverse();

    tf::Matrix3x3 quaternion_matrix(inverse_quaternion);
    tf::Vector3 inverse_tanslation_vector = quaternion_matrix * translation_vector;
    inverse_tanslation_vector = -inverse_tanslation_vector; // Invert the translation vector

    inverse_transform.translation.x = inverse_tanslation_vector.x();
    inverse_transform.translation.y = inverse_tanslation_vector.y();
    inverse_transform.translation.z = inverse_tanslation_vector.z();
    inverse_transform.rotation.x = inverse_quaternion.x();
    inverse_transform.rotation.y = inverse_quaternion.y();
    inverse_transform.rotation.z = inverse_quaternion.z();
    inverse_transform.rotation.w = inverse_quaternion.w();
    return inverse_transform;
}

geometry_msgs::Transform forwardTransform(const geometry_msgs::Transform &A2B, const geometry_msgs::Transform & B2C)
{
    tf::Quaternion q_A2B(A2B.rotation.x, A2B.rotation.y, A2B.rotation.z, A2B.rotation.w);
    tf::Quaternion q_B2C(B2C.rotation.x, B2C.rotation.y, B2C.rotation.z, B2C.rotation.w);
    tf::Quaternion q_A2C = q_A2B * q_B2C;
    tf::Vector3 t_A2B(A2B.translation.x, A2B.translation.y, A2B.translation.z);
    tf::Vector3 t_B2C(B2C.translation.x, B2C.translation.y, B2C.translation.z);
    tf::Matrix3x3 R_A2B(q_A2B);
    tf::Vector3 t_A2C = t_A2B + R_A2B * t_B2C;

    geometry_msgs::Transform forward_transform;
    forward_transform.translation.x = t_A2C.x();
    forward_transform.translation.y = t_A2C.y();
    forward_transform.translation.z = t_A2C.z();
    forward_transform.rotation.x = q_A2C.x();
    forward_transform.rotation.y = q_A2C.y();
    forward_transform.rotation.z = q_A2C.z();
    forward_transform.rotation.w = q_A2C.w();

    return forward_transform;
} 

tf::Vector3 rotate_by_quat_inv(const tf::Quaternion& q, const tf::Vector3& v) {
    tf::Quaternion qinv = q.inverse();
    tf::Matrix3x3 R(qinv);
    return R * v;
}

namespace aerial_robot_control
{

BeetlePoseRLAgent::BeetlePoseRLAgent(): ControlBase()
{
}

void BeetlePoseRLAgent::initialize(ros::NodeHandle nh, ros::NodeHandle nhp,
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
    ROS_ERROR("[RL-Agent] "
        "Parameter ~onnx_model_path is not loaded. "
        "You must specify the onnx_model_path in the launch file.");
    return;
  }
  if (onnx_model_path.empty())
  {
    ROS_ERROR("[RL-Agent]The onnx_model_path is empty!");
    return;
  }
  else
  {
    std::string pkg_path = ros::package::getPath("beetle_omni");
    onnx_model_path = pkg_path + "/" + onnx_model_path;
    ROS_INFO("[RL-Agent] ONNX model path: %s", onnx_model_path.c_str());
  }

  double goal_angle_R, goal_angle_P, goal_angle_Y;
  std::string odom_topic;
  XmlRpc::XmlRpcValue gimbal_default_xml, scales_xml;
  getParam<double>(rl_nh,"control_freq", control_hz_, 200.0);
  getParam<int>(rl_nh,"decimation", decimation_, 4);
  getParam<bool>(rl_nh,"ideal_obs", ideal_obs_, false);
  getParam<std::string>(rl_nh,"ideal_obs_topic", odom_topic, "uav/cog/odom");
  getParam<int>(rl_nh,"ideal_delay", ideal_delay_, 4);  // 100Hz 
  getParam<bool>(rl_nh,"rlagent_verbose", verbose_, false);
  getParam<bool>(rl_nh,"rlagent_debug", debug_, false);
  getParam<bool>(rl_nh,"rlagent_forward_info", forward_info_, false);
  getParam<int>(rl_nh,"gimbal_target_delay_steps", gimbal_target_delay_steps_, 0);
  getParam<int>(rl_nh,"gimbal_obs_delay_steps", gimbal_obs_delay_steps_, 0);
  getParam<bool>(rl_nh,"enable_thrust", enable_thrust_, false);
  getParam<double>(rl_nh,"thrust_scale", thrust_scale_, 1.0);
  getParam<double>(rl_nh,"thrust_tau", thrust_tau_, 0.05);
  getParam<int>(rl_nh,"thrust_target_delay_steps", thrust_target_delay_steps_, 0);
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
  getParam<double>(rl_nh,"thrust_scale", thrust_scale_, 0.0);
  if (thrust_scale_ > 1.0)
    thrust_scale_ = 1.0;
  getParam<XmlRpc::XmlRpcValue>(rl_nh,"gimbal_default", gimbal_default_xml, XmlRpc::XmlRpcValue());
  getParam<XmlRpc::XmlRpcValue>(rl_nh,"scales", scales_xml, XmlRpc::XmlRpcValue());
  getParam<int>(rl_nh,"gimbal_size", gimbal_size_, 4);
  getParam<int>(rl_nh,"thrust_size", thrust_size_, 4);
  getParam<double>(rl_nh,"gimbal_kp", gimbal_kp_, 5.0);
  getParam<double>(rl_nh,"gimbal_kd", gimbal_kd_, 0.1);
  getParam<bool>(rl_nh,"gimbal_effort_ctrl", gimbal_effort_ctrl_, false);
  tf::Quaternion quat = tf::createQuaternionFromRPY(goal_angle_R, goal_angle_P, goal_angle_Y);
  desired_pose_.pose.orientation.x = quat.x();
  desired_pose_.pose.orientation.y = quat.y();
  desired_pose_.pose.orientation.z = quat.z();
  desired_pose_.pose.orientation.w = quat.w();
  std::cout << "Desired \n";
  std::cout << "Pos : " << desired_pose_.pose.position.x << ", " << desired_pose_.pose.position.y << ", " << desired_pose_.pose.position.z << "\n";
  std::cout << "Ori : " << desired_pose_.pose.orientation.x << ", " << desired_pose_.pose.orientation.y << ", " << desired_pose_.pose.orientation.z << ", " << desired_pose_.pose.orientation.w << "\n";

  for (auto it = gimbal_default_xml.begin(); it != gimbal_default_xml.end(); ++it)
  {
    std::string name = static_cast<std::string>(it->first);
    double value = static_cast<double>(it->second);
    gimbal_default_pos_.push_back(value);
    gimbal_names_.push_back(name);  
    if (verbose_)
      ROS_INFO("[RL-Agent] Gimbal default: %s = %.3f", name.c_str(), value);
  }

  for (auto it = scales_xml.begin(); it != scales_xml.end(); ++it)
  {
    std::string name = static_cast<std::string>(it->first);
    double value = static_cast<double>(it->second);
    scales[name] = value;
    if (verbose_)
      ROS_INFO("[RL-Agent] Scale: %s = %.3f", name.c_str(), value);
  }

  // 2. initialize ONNX Runtime
  if (verbose_)
    ROS_INFO("[RL-Agent] Loading Agent Model......");
  initRLAgent(onnx_model_path);
  if (verbose_)
    ROS_INFO("[RL-Agent] Agent Model initialized.");

  if (gimbal_size_ + thrust_size_ != action_size_)
  {
    ROS_ERROR("[RL-Agent] The action size (%lu) does not match gimbal_size (%d) + thrust_size (%d)!\n"
              "Please check the parameter settings.",
              action_size_, gimbal_size_, thrust_size_);
    return;
  }
  if (gimbal_size_ != static_cast<int>(gimbal_default_pos_.size()))
  {
    ROS_ERROR("[RL-Agent] The gimbal_size (%d) does not match the size of gimbal_default (%lu)!\n"
              "Please check the parameter settings.",
              gimbal_size_, gimbal_default_pos_.size());
    return;
  }
  target_gimbal_.resize(gimbal_size_);
  target_thrust_.resize(thrust_size_);
  target_gimbal_.assign(gimbal_size_, 0.0);
  target_thrust_.assign(thrust_size_, 0.0);
  last_action_.resize(action_size_);
  gimbal_pos_.resize(gimbal_size_, 0.0);
  gimbal_vel_.resize(gimbal_size_, 0.0);
  gimbal_pos_.assign(gimbal_size_, 0.0);
  gimbal_vel_.assign(gimbal_size_, 0.0);

  target_gimbal_list_.clear();
  if (gimbal_target_delay_steps_ > 0) {
      // pre-fill with current target_gimbal_ so initial outputs are stable
      for (int i = 0; i < gimbal_target_delay_steps_; ++i) target_gimbal_list_.push_back(target_gimbal_);
  }
  gimbal_pos_list_.clear();
  if (gimbal_obs_delay_steps_ > 0) {
      // pre-fill with current gimbal_pos_ so initial observations are stable
      for (int i = 0; i < gimbal_obs_delay_steps_; ++i) gimbal_pos_list_.push_back(gimbal_pos_);
  }
  target_thrust_list_.clear();
  if (thrust_target_delay_steps_ > 0) {
      // pre-fill with current target_thrust_ so initial outputs are stable
      for (int i = 0; i < thrust_target_delay_steps_; ++i) target_thrust_list_.push_back(target_thrust_);
  }

  ang_vel_list_.clear();
  lin_vel_list_.clear();
  if (ideal_delay_ > 0) {
    tf::Vector3 zero_vec(0.0, 0.0, 0.0);
    // pre-fill with current target_gimbal_ so initial outputs are stable
    for (int i = 0; i < ideal_delay_; ++i) ang_vel_list_.push_back(zero_vec);
    for (int i = 0; i < ideal_delay_; ++i) lin_vel_list_.push_back(zero_vec);
  }
  // 3. initialize ros 
  goal_sub_   = nh_.subscribe("/desired_3D_pose", 1, &BeetlePoseRLAgent::goalCallback, this);
  gimbal_sub_ = nh_.subscribe("joint_states", 1, &BeetlePoseRLAgent::gimbalCallback, this);
  odom_sub_ = nh_.subscribe(odom_topic, 1, &BeetlePoseRLAgent::odomCallback, this);
  thrust_pub_ = nh_.advertise<spinal::FourAxisCommand>("four_axes/command", 1);
  thrust_debug_pub_ = nh_.advertise<spinal::FourAxisCommand>("four_axes/command_debug", 1);
  gimbal_pub_ = nh_.advertise<sensor_msgs::JointState>("gimbals_ctrl", 1);
  gimbal_effort_pub1_ = nh_.advertise<std_msgs::Float64>("servo_controller/gimbals/controller1/simulation/command", 1);
  gimbal_effort_pub2_ = nh_.advertise<std_msgs::Float64>("servo_controller/gimbals/controller2/simulation/command", 1);
  gimbal_effort_pub3_ = nh_.advertise<std_msgs::Float64>("servo_controller/gimbals/controller3/simulation/command", 1);
  gimbal_effort_pub4_ = nh_.advertise<std_msgs::Float64>("servo_controller/gimbals/controller4/simulation/command", 1);
  gimbal_debug_pub_ = nh_.advertise<sensor_msgs::JointState>("gimbals_ctrl_debug", 1);
  obs_debug_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("observation_debug", 1);

   /* reset control input */
  thrust_cmd_.base_thrust = std::vector<float>(thrust_size_, 0.0);

  if (!gimbal_cmd_.name.empty()) gimbal_cmd_.name.clear();
  if (!gimbal_cmd_.position.empty()) gimbal_cmd_.position.clear();
  for (int i = 0; i < gimbal_size_; i++)
  {
    gimbal_cmd_.name.emplace_back(gimbal_names_[i]);
    gimbal_cmd_.position.push_back(gimbal_default_pos_[i]);
    gimbal_cmd_.effort.push_back(0.0);
  }
  if (verbose_)
    ROS_INFO("[RL-Agent] Initialized Successfully.");
}

bool BeetlePoseRLAgent::update()
{
  // std::cout << "-------- [RL Agent] update() called --------" << std::endl;
  ros::Time now = ros::Time::now();
  if (step_count_ % decimation_ == 0)
  {
    buildObservation();
    policyForward();
  }
  step_count_++;
  sendCmd();
  double elapsed = (now - last_infer_report_time_).toSec();
  if (elapsed >= 2.0 && infer_count_ > 0)
  {
    double avg_ns = static_cast<double>(infer_total_ns_) / static_cast<double>(infer_count_);
    double avg_ms = avg_ns / 1e6;
    double min_ms = static_cast<double>(infer_min_ns_) / 1e6;
    double max_ms = static_cast<double>(infer_max_ns_) / 1e6;
    double avg_period_ms = (elapsed / static_cast<double>(infer_count_)) * 1000.0;
    if (forward_info_)
      ROS_INFO(
          "[RL-Agent]\n"
          "ORT Inference (last %.2fs): calls=%lu\n"
          "  avg_time=%.3f ms, avg_period=%.3f ms\n"
          "  min_time=%.3f ms, max_time=%.3f ms\n"
          "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
          "Pos Error: [%.3f, %.3f, %.3f]\n"
          "Ang Error: [%.3f, %.3f, %.3f]\n"
          // "Pos Body : [%.3f, %.3f, %.3f]\n"
          // "Ang Body : [%.3f, %.3f, %.3f]\n"
          // "Pos Goal : [%.3f, %.3f, %.3f]\n"
          // "Ang Goal : [%.3f, %.3f, %.3f]\n"
          "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
          elapsed,
          static_cast<unsigned long>(infer_count_),
          avg_ms,
          avg_period_ms,
          min_ms,
          max_ms,
          pos_error.x(), pos_error.y(), pos_error.z(),
          ang_error.x(), ang_error.y(), ang_error.z()
          // pos_body.x(), pos_body.y(), pos_body.z(),
          // ang_body.x(), ang_body.y(), ang_body.z(),
          // pos_goal.x(), pos_goal.y(), pos_goal.z(),
          // ang_goal.x(), ang_goal.y(), ang_goal.z()
       );
    // reset counters
    infer_count_ = 0;
    infer_total_ns_ = 0;
    infer_max_ns_ = 0;
    infer_min_ns_ = std::numeric_limits<uint64_t>::max();
    last_infer_report_time_ = now;
  }
  return ControlBase::update();
  // std::cout << "-------- [RL Agent] update() finished --------" << std::endl;
}

void BeetlePoseRLAgent::reset()
{
  ControlBase::reset();
  step_count_ = 0;
  if (verbose_)
    ROS_INFO("[RL-Agent]Reset RL Agent.");
  last_infer_report_time_ = ros::Time::now();
  // Needs to be done: reset the mpc_solver with data
  action_.assign(action_size_, 0.0f);
  for (size_t i = 4; i < action_.size(); ++i)
    action_[i] = -thrust_default_/scales["thrust"];
  sendCmd();
}

BeetlePoseRLAgent::~BeetlePoseRLAgent()
{
  ROS_INFO("-------- BeetlePoseRLAgent destructor called --------");
    // free allocated C strings from GetInputNameAllocated / GetOutputNameAllocated
    if (input_name_) allocator_.Free(const_cast<char*>(input_name_));
    if (output_name_) allocator_.Free(const_cast<char*>(output_name_));
}

void BeetlePoseRLAgent::policyForward()
{
  // std::cout << "-------- [RL Agent] policyForward() called --------" << std::endl;
  // Create input tensor object from data values
  if (!catch_obs_) {
    ROS_WARN_THROTTLE(2.0, "[RL-Agent] No valid observation constructed yet, skipping policyForward()");
    action_.assign(action_size_, 0.0f);
    return;
  }
  // Clamp observation values to finite numbers
  for (size_t i = 0; i < observation_.size(); ++i) {
    if (observation_[i] > scales["clip_observation"]) 
      observation_[i] = scales["clip_observation"];
    else if (observation_[i] < -scales["clip_observation"]) 
      observation_[i] = -scales["clip_observation"];
  }
  std::vector<int64_t> input_shape = {1, static_cast<int64_t>(observation_.size())};
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, observation_.data(), observation_.size(), input_shape.data(), input_shape.size());

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
    ROS_WARN("[RL-Agent]Model output length (%zu) differs from stored action_size_ (%zu). set zero for default", out_len, action_size_);
    action_.assign(action_size_, 0.0f);
  }
  else {
    action_ = action;
  }
  // Clamp action values to finite numbers
  for (size_t i = 0; i < action_.size(); ++i) {
    if (action_[i] > scales["clip_action"]) 
      action_[i] = scales["clip_action"];
    else if (action_[i] < -scales["clip_action"]) 
      action_[i] = -scales["clip_action"];
  }
  catch_obs_ = false;
  // std::cout << "-------- [RL Agent] policyForward() finished --------" << std::endl;
}

void BeetlePoseRLAgent::initRLAgent(std::string model_path)
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
    if (verbose_)
      ROS_INFO("[RL-Agent] Model input size: %zu", obs_size_);
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
    if (verbose_)
      ROS_INFO("[RL-Agent] Model output size: %zu", action_size_);
  }
  observation_.resize(obs_size_, 0.0f);
  action_.resize(action_size_, 0.0f);
  last_action_.resize(action_size_, 0.0f);
}

void BeetlePoseRLAgent::buildObservation()
{
  // std::cout << "-------- [RL Agent] buildObservation() called --------" << std::endl;
  std::lock_guard<std::mutex> lk(data_mutex_);
  if (!gimbal_catch_ ) {
    ROS_WARN_THROTTLE(2.0, "[RL-Agent] No gimbal state received yet!");
    catch_obs_ = false;
    return;
  }
  if (ideal_obs_ && !odom_catch_) {
    ROS_WARN_THROTTLE(2.0, "[RL-Agent] No Odometry received yet!");
    catch_obs_ = false;
    return;
  }
  // body quaternion and pos
  tf::Quaternion body_quat;
  if (ideal_obs_)
    body_quat = tf::Quaternion(odom_msg_.pose.pose.orientation.x,
                            odom_msg_.pose.pose.orientation.y,
                            odom_msg_.pose.pose.orientation.z,
                            odom_msg_.pose.pose.orientation.w);
  else {
    tf::Matrix3x3 body_mat = estimator_->getOrientation(Frame::BASELINK, estimate_mode_);
    body_mat.getRotation(body_quat);
  }
  // normalize body_quat
  body_quat.normalize();

  tf::Vector3 body_pos;
  if (ideal_obs_)
    body_pos = tf::Vector3(odom_msg_.pose.pose.position.x,
                          odom_msg_.pose.pose.position.y,
                          odom_msg_.pose.pose.position.z);
  else
    body_pos = estimator_->getPos(Frame::BASELINK, estimate_mode_);

  pos_body = body_pos;
  double body_roll, body_pitch, body_yaw;
  tf::Matrix3x3 body_rot_mat(body_quat);
  body_rot_mat.getRPY(body_roll, body_pitch, body_yaw);
  ang_body = tf::Vector3(body_roll, body_pitch, body_yaw);

  // linear velocity of COM in body frame
  tf::Vector3 lin_vel_world;
  if (ideal_obs_) {
    // handle ideal delay buffer
    lin_vel_world = tf::Vector3(odom_msg_.twist.twist.linear.x,
                  odom_msg_.twist.twist.linear.y,
                  odom_msg_.twist.twist.linear.z);
    if (ideal_delay_ > 0) {
      lin_vel_list_.push_back(lin_vel_world);
      lin_vel_world = lin_vel_list_.front();
      if (static_cast<int>(lin_vel_list_.size()) > ideal_delay_ + 1) {
        lin_vel_list_.erase(lin_vel_list_.begin());
      }
    }
  }
  else
    lin_vel_world  = estimator_->getVel(Frame::COG, estimate_mode_);
  
  tf::Vector3 lin_vel_body = rotate_by_quat_inv(body_quat, lin_vel_world);

  tf::Vector3 ang_vel_body;
  if (ideal_obs_) {
    ang_vel_body = tf::Vector3(odom_msg_.twist.twist.angular.x,
                  odom_msg_.twist.twist.angular.y,
                  odom_msg_.twist.twist.angular.z);
    if (ideal_delay_ > 0) {
      ang_vel_list_.push_back(ang_vel_body);
      ang_vel_body = ang_vel_list_.front();
      if (static_cast<int>(ang_vel_list_.size()) > ideal_delay_ + 1) {
        ang_vel_list_.erase(ang_vel_list_.begin()); 
      }
    }
  }
  else
    ang_vel_body = estimator_->getAngularVel(Frame::COG, estimate_mode_);
  // tf::Vector3 ang_vel_body = rotate_by_quat_inv(body_quat, ang_vel_world);
  // gravity projection (rotate world [0,0,-1] into body)
  tf::Vector3 gravity_b = rotate_by_quat_inv(body_quat, tf::Vector3(0.0, 0.0, -1.0));

  // goal pos in body frame
  tf::Quaternion goal_quat(desired_pose_.pose.orientation.x,
                    desired_pose_.pose.orientation.y,
                    desired_pose_.pose.orientation.z,
                    desired_pose_.pose.orientation.w);
  // normalize goal_quat
  goal_quat.normalize();
  tf::Vector3 goal_world(desired_pose_.pose.position.x,
                          desired_pose_.pose.position.y,
                          desired_pose_.pose.position.z);
  // compute goal in body frame: q_inv * (goal - body_pos)

  pos_goal = goal_world;
  double goal_roll, goal_pitch, goal_yaw;
  tf::Matrix3x3 goal_rot_mat(goal_quat);
  goal_rot_mat.getRPY(goal_roll, goal_pitch, goal_yaw);
  ang_goal = tf::Vector3(goal_roll, goal_pitch, goal_yaw);

  tf::Vector3 t02_minus_t01 = goal_world - body_pos;
  tf::Vector3 goal_pos = rotate_by_quat_inv(body_quat, t02_minus_t01);
  goal_quat = body_quat.inverse() * goal_quat;

  pos_error = goal_pos;
  double error_roll, error_pitch, error_yaw;
  tf::Matrix3x3 error_rot_mat(goal_quat);
  error_rot_mat.getRPY(error_roll, error_pitch, error_yaw);
  ang_error = tf::Vector3(error_roll, error_pitch, error_yaw);

  if (gimbal_obs_delay_steps_ > 0) {
      gimbal_pos_list_.push_back(gimbal_pos_);
      gimbal_pos_ = gimbal_pos_list_.front();
      if (static_cast<int>(gimbal_pos_list_.size()) > gimbal_obs_delay_steps_ + 1) {
          gimbal_pos_list_.erase(gimbal_pos_list_.begin());
      }
  }

  // root_rot_vec and goal_rot_vec: approximate by taking first 2x3 of rotation matrix (6 elements)
  tf::Matrix3x3 Rb(body_quat), Rg(goal_quat);
  // take rows 0..1 and cols 0..2 -> 2x3 flattened == 6 elements
  std::vector<float> root_rot_vec;
  std::vector<float> goal_rot_vec;
  for (int r = 0; r < 2; ++r) {
      for (int c = 0; c < 3; ++c) {
          root_rot_vec.push_back(Rb[r][c]);
          goal_rot_vec.push_back(Rg[r][c]);
      }
  }
  std::vector<float>temp_obs;
  temp_obs.reserve(obs_size_);
  // lin vel (3) 3
  temp_obs.push_back(lin_vel_body.x() * scales["lin_vel"]); temp_obs.push_back(lin_vel_body.y() * scales["lin_vel"]); temp_obs.push_back(lin_vel_body.z() * scales["lin_vel"]);
  // ang vel (3) 6
  temp_obs.push_back(ang_vel_body.x() * scales["agn_vel"]); temp_obs.push_back(ang_vel_body.y() * scales["agn_vel"]); temp_obs.push_back(ang_vel_body.z() * scales["agn_vel"]);
  // gravity (3) 9
  temp_obs.push_back(gravity_b.x()); temp_obs.push_back(gravity_b.y()); temp_obs.push_back(gravity_b.z());
  // goal pos (3) 12
  temp_obs.push_back(goal_pos.x()); temp_obs.push_back(goal_pos.y()); temp_obs.push_back(goal_pos.z());
  // gimbal dof (4) 16
  for (size_t i = 0; i < gimbal_size_; ++i) temp_obs.push_back(gimbal_pos_[i]);
  // root_rot_vec (6) 22
  temp_obs.insert(temp_obs.end(), root_rot_vec.begin(), root_rot_vec.end());
  // goal_rot_vec (6) 28
  temp_obs.insert(temp_obs.end(), goal_rot_vec.begin(), goal_rot_vec.end());
  // last_action (8) 36
  temp_obs.insert(temp_obs.end(), last_action_.begin(), last_action_.end());
  // verify obs36 length (should be 36)
  if (temp_obs.size() != obs_size_) {
    ROS_WARN("[RL-Agent] Observation size (%zu) does not match expected obs_size_ (%zu)", observation_.size(), obs_size_);
  }
  else {
    // trim if larger
    observation_.assign(temp_obs.begin(), temp_obs.begin() + obs_size_);
  }
  catch_obs_ = true;
  // std::cout << "-------- [RL Agent`] buildObservation() finished --------" << std::endl;
}

void BeetlePoseRLAgent::sendCmd()
{
  for (size_t i = 0; i < action_size_; ++i) {
    last_action_[i] = action_[i];
  }
  for (size_t i = 0; i < gimbal_size_; ++i) {
    target_gimbal_[i] = action_[i] * scales["gimbal_act"] + gimbal_default_pos_[i];
  }
  if (gimbal_target_delay_steps_ > 0) {
      target_gimbal_list_.push_back(target_gimbal_);
      target_gimbal_ = target_gimbal_list_.front();
      if (static_cast<int>(target_gimbal_list_.size()) > gimbal_target_delay_steps_ + 1) {
          target_gimbal_list_.erase(target_gimbal_list_.begin());
      }
  }
  for (size_t i = 0; i < gimbal_size_; ++i) {
    gimbal_cmd_.position[i] = target_gimbal_[i];
  }
  if (gimbal_effort_ctrl_) {
    // compute effort command using PD control
    for (size_t i = 0; i < gimbal_size_; ++i) {
      double pos_err = target_gimbal_[i] - gimbal_pos_[i];
      double vel_err = - gimbal_vel_[i]; // assume zero velocity feedback for now
      gimbal_cmd_.effort[i] = static_cast<float>(gimbal_kp_ * pos_err + gimbal_kd_ * vel_err);
    }
  }
  if (thrust_tau_ > 0.0)  {
    for (size_t i = 0; i < thrust_size_; ++i) {
      double a = std::exp(- (1.0 / 200.0) / thrust_tau_);
      double thrust_input = action_[gimbal_size_ + i] * scales["thrust_act"] + thrust_default_;
      double thrust_old = target_thrust_[i];
      target_thrust_[i] = a * thrust_old + (1 - a) * thrust_input;
    }
  }
  else {
    for (size_t i = 0; i < thrust_size_; ++i) {
      target_thrust_[i] = action_[gimbal_size_ + i] * scales["thrust_act"] + thrust_default_;
    }
  }
  if (thrust_target_delay_steps_ > 0) {
      target_thrust_list_.push_back(target_thrust_);
      target_thrust_ = target_thrust_list_.front();
      if (static_cast<int>(target_thrust_list_.size()) > thrust_target_delay_steps_ + 1) {
          target_thrust_list_.erase(target_thrust_list_.begin());
      }
  }
  for (size_t i = 0; i < thrust_size_; ++i) {
      if (!std::isfinite(target_thrust_[i])) {
        std::cout << "BAD target_thrust_ at i=" << i << std::endl;
        target_thrust_[i] = thrust_default_;
      }
    thrust_cmd_.base_thrust[i] = target_thrust_[i] * thrust_scale_ * thrust_scale_;
  }
  gimbal_cmd_.header.stamp = ros::Time::now();
  if (enable_gimbal_){
    if (gimbal_effort_ctrl_)
    {
      std_msgs::Float64 effort_msg;
      effort_msg.data = gimbal_cmd_.effort[0];
      gimbal_effort_pub1_.publish(effort_msg);
      effort_msg.data = gimbal_cmd_.effort[1];
      gimbal_effort_pub2_.publish(effort_msg);
      effort_msg.data = gimbal_cmd_.effort[2];
      gimbal_effort_pub3_.publish(effort_msg);
      effort_msg.data = gimbal_cmd_.effort[3];
      gimbal_effort_pub4_.publish(effort_msg);
    }
    else
      gimbal_pub_.publish(gimbal_cmd_);
  }
  if (enable_thrust_ && control_timestamp_ > 0.0)
    thrust_pub_.publish(thrust_cmd_);
  if (debug_) {
    gimbal_debug_pub_.publish(gimbal_cmd_);
    thrust_debug_pub_.publish(thrust_cmd_);
    std_msgs::Float32MultiArray obs_msg;
    obs_msg.data = observation_;
    obs_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
    obs_msg.layout.dim[0].label = "observation";
    obs_msg.layout.dim[0].size = observation_.size();
    obs_msg.layout.dim[0].stride = observation_.size();
    obs_debug_pub_.publish(obs_msg);
  }
  // if (debug_ && step_count_ % PRINT_FREQUENCY == 0)
  //   std::cout << "-------- [RL Agent] sendCmd() finished --------" << std::endl;
}

void BeetlePoseRLAgent::goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
  // ROS_INFO("-------- [RL Agent] goalCallback() called --------");
  std::lock_guard<std::mutex> lock(data_mutex_);
  if (msg->pose.position.z < 0.3) {
    ROS_WARN("[RL-Agent] Received goal z position too low (%.3f), just skip update", msg->pose.position.z);
    return;
  }
  ROS_INFO("[RL-Agent] New goal received: [%.3f, %.3f, %.3f]", msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  desired_pose_ = *msg;
}

void BeetlePoseRLAgent::gimbalCallback(const sensor_msgs::JointState::ConstPtr& msg)
{
  // if (debug_ && step_count_ % PRINT_FREQUENCY == 0)
  //   std::cout << "-------- [RL Agent] gimbalCallback() called --------" << std::endl;
  std::lock_guard<std::mutex> lock(data_mutex_);
  for (size_t i=0, j=0; i < msg->name.size(); i++) {
    if (msg->name[i].find(gimbal_cmd_.name[j]) != std::string::npos && j < gimbal_size_) {
      gimbal_pos_[j] = msg->position[i] - gimbal_default_pos_[j];
      if (gimbal_effort_ctrl_)
        gimbal_vel_[j] = msg->velocity[i];
      j++;
    }
  }
  gimbal_catch_ = true;
}

void BeetlePoseRLAgent::imuCallback(const spinal::Imu::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  imu_msg_ = *msg;
  imu_catch_ = true;
}

void BeetlePoseRLAgent::odomCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  odom_msg_ = *msg;
  odom_catch_ = true;
}

  }  // namespace aerial_robot_control

/* plugin registration */
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(aerial_robot_control::BeetlePoseRLAgent, aerial_robot_control::ControlBase);