//
// Created by li-jinjie on 2025/10/9.
//

#include "aerial_robot_control/rl/base_rl_agent.h"

namespace aerial_robot_control
{
namespace rlagent
{

void BaseRLAgent::initialize(ros::NodeHandle nh, ros::NodeHandle nhp,
                             boost::shared_ptr<aerial_robot_model::RobotModel> robot_model,
                             boost::shared_ptr<aerial_robot_estimation::StateEstimator> estimator,
                         boost::shared_ptr<aerial_robot_navigation::BaseNavigator> navigator, double ctrl_loop_du)
{
  ControlBase::initialize(nh, nhp, robot_model, estimator, navigator, ctrl_loop_du);
  try
  {
    // 1. read the ONNX model path from parameter server
    std::string onnx_model_path;
    if (!nh_.getParam("onnx_model_path", onnx_model_path))
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

    getParam<bool>("control_freq", control_hz_, 200.0);
    getParam<int>("decimation", decimation_, 4);
    getParam<bool>("rlagent_verbose", verbose_, false);
    
    // 2. initialize ONNX Runtime
    initRLAgent(onnx_model_path);
  }
}

bool BaseRLAgent::update()
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
  if (elapsed >= 2.0 && infer_count_ > 0) {
    double avg_ns = static_cast<double>(infer_total_ns_) / static_cast<double>(infer_count_);
    double avg_ms = avg_ns / 1e6;
    double min_ms = static_cast<double>(infer_min_ns_) / 1e6;
    double max_ms = static_cast<double>(infer_max_ns_) / 1e6;
    double avg_period_ms = (elapsed / static_cast<double>(infer_count_)) * 1000.0;
    if (verbose_)
      ROS_INFO("[RL-Agent]\n"
        "ORT Inference (last %.2fs): calls=%lu\n"
        "  avg_time=%.3f ms, avg_period=%.3f ms\n"
        "  min=%.3f ms, max=%.3f ms\n"
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
        elapsed,
        static_cast<unsigned long>(infer_count_),
        avg_ms,
        avg_period_ms,
        min_ms,
        max_ms
      );
    // reset counters
    infer_count_ = 0;
    infer_total_ns_ = 0;
    infer_max_ns_ = 0;
    infer_min_ns_ = std::numeric_limits<uint64_t>::max();
    last_infer_report_time_ = now;
  }
  return ControlBase::update();
}

void BaseRLAgent::reset()
{
  ControlBase::reset();
  step_count_ = 0;
  if (verbose_)
    ROS_INFO("[RL-Agent]Reset RL Agent.");
  last_infer_report_time_ = ros::Time::now();
  // Needs to be done: reset the mpc_solver with data
}

BaseRLAgent::~BaseRLAgent()
{
    // free allocated C strings from GetInputNameAllocated / GetOutputNameAllocated
    if (input_name_) allocator_.Free(const_cast<char*>(input_name_));
    if (output_name_) allocator_.Free(const_cast<char*>(output_name_));
}

void BaseRLAgent::policyForward()
{
  // Create input tensor object from data values
  std::vector<int64_t> input_shape = {1, static_cast<int64_t>(obs.size())};
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, obs.data(), obs.size(), input_shape.data(), input_shape.size());

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

void BaseRLAgent::initRLAgent(std::string model_path)
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
            ROS_WARN("Unexpected model input dims size=%zu, using product", in_dims.size());
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


}  // namespace rlagent
}  // namespace aerial_robot_control
