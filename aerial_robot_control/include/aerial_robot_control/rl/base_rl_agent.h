//
// Created by wentao-zhang on 25-12-04.
//

#ifndef BASE_RL_AGENT_CONTROLLER_H
#define BASE_RL_AGENT_CONTROLLER_H

#include "aerial_robot_control/control/base/base.h"
#include <onnxruntime_cxx_api.h>
#include <mutex>  // for std::mutex
#include <chrono> // for high_resolution_clock

namespace aerial_robot_control
{
namespace rlagent
{

/**
 * @brief The BaseMPC class defines the common interface of NMPC controllers, especially for the ROS communication.
 */
class BaseRLAgent : public ControlBase
{
public:
  BaseRLAgent() = default;
  ~BaseRLAgent() override = default;
  // ----------- lifecycle -----------
  void initialize(ros::NodeHandle nh, ros::NodeHandle nhp,
                  boost::shared_ptr<aerial_robot_model::RobotModel> robot_model,
                  boost::shared_ptr<aerial_robot_estimation::StateEstimator> estimator,
                  boost::shared_ptr<aerial_robot_navigation::BaseNavigator> navigator, double ctrl_loop_du) override;
  bool update() override;
  void reset() override;

protected:
  // ----------- general parameters -----------
  std::vector<float> observation_;
  std::vector<float> action_;
  double control_hz_;
  size_t obs_size_;
  size_t action_size_;
  int step_count_ = 0;
  int decimation_ = 4;
  bool verbose_ = false;

  // ----------- RL Agent functions -----------
  virtual void initRLAgent(std::string model_path);
  // void initRLAgent(std::string model_path);
  
  virtual void buildObservation() = 0;
  virtual void policyForward();

  virtual void sendCmd() = 0;

private:
  // ----------- ONNX Runtime -----------
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::SessionOptions> sess_opts_;
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;
  const char* input_name_ = nullptr;
  const char* output_name_ = nullptr;

  // ONNX inference timing stats
  size_t infer_count_ = 0;
  uint64_t infer_total_ns_ = 0;
  uint64_t infer_max_ns_ = 0;
  uint64_t infer_min_ns_ = std::numeric_limits<uint64_t>::max();
  ros::Time last_infer_report_time_;
};

}  // namespace rlagent
}  // namespace aerial_robot_control

#endif  // BASE_RL_AGENT_CONTROLLER_H