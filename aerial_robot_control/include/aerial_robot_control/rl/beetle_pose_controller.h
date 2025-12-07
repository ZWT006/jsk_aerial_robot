//
// Created by wentao-zhang on 25-12-04.
//

#ifndef BEETLE_RL_CONTROLLER_H
#define BEETLE_RL_CONTROLLER_H

#include "aerial_robot_control/control/base/base.h"
#include <onnxruntime_cxx_api.h>
#include <mutex>  // for std::mutex
#include <chrono> // for high_resolution_clock
#include <cmath> // for std::isfinite
#include <math.h> // for M_PI

#include <deque>
#include <std_msgs/Float32MultiArray.h>
#include <spinal/FourAxisCommand.h>
#include <spinal/Imu.h>
#include <sensor_msgs/JointState.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Transform.h>
#include <geometry_msgs/TransformStamped.h>


namespace aerial_robot_control
{

  class BeetlePoseController : public ControlBase
  {
  public:
    BeetlePoseController();
    ~BeetlePoseController() override;
    void initialize(ros::NodeHandle nh,
                            ros::NodeHandle nhp,
                            boost::shared_ptr<aerial_robot_model::RobotModel> robot_model,
                            boost::shared_ptr<aerial_robot_estimation::StateEstimator> estimator,
                            boost::shared_ptr<aerial_robot_navigation::BaseNavigator> navigator,
                            double ctrl_loop_du) override;
    bool update() override;
    void reset() override;
  protected:
    // ----------- RL Agent functions -----------
    void initRLAgent(std::string model_path);
    // void initRLAgent(std::string model_path);
    
    void buildObservation();
    void policyForward();

    void sendCmd();

    // ----------- ROS Callbacks -----------
    void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void gimbalCallback(const sensor_msgs::JointState::ConstPtr& msg);

  private:
    ros::Subscriber goal_sub_, gimbal_sub_;
    ros::Publisher thrust_pub_, thrust_debug_pub_, gimbal_pub_, gimbal_debug_pub_;
    ros::Publisher obs_debug_pub_;
    spinal::FourAxisCommand thrust_cmd_;
    sensor_msgs::JointState gimbal_cmd_;
    // ----------- Controller parameters -----------
    std::mutex data_mutex_;
    spinal::Imu imu_msg_;
    bool imu_catch_ = false;
    nav_msgs::Odometry odom_msg_;
    bool odom_catch_ = false;
    sensor_msgs::JointState gimbal_msg_;
    bool gimbal_catch_ = false;
    geometry_msgs::PoseStamped desired_pose_;
    geometry_msgs::Transform cog2root_T;
    geometry_msgs::Transform world2cog_T;
    geometry_msgs::Transform world2root_T;

    // ----------- general parameters -----------
    std::vector<float> obs_;
    std::vector<float> action_;
    double control_hz_;
    size_t obs_size_;
    size_t action_size_;
    int step_count_ = 0;
    int decimation_ = 4;
    bool verbose_ = false;

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

    // ----------- Beetle specific parameters -----------
    bool enable_thrust_ = false;
    bool lock_thrust_ = false;
    bool enable_gimbal_ = false;
    bool lock_gimbal_ = false;

    int gimbal_size_;
    int thrust_size_;

    int gimbal_target_delay_steps_ = 0;
    int gimbal_obs_delay_steps_ = 0;

    std::vector<float> gimbal_pos_;
    std::vector<float> gimbal_default_pos_;
    std::deque<std::vector<float>> gimbal_pos_list_;

    std::vector<float> target_gimbal_;
    std::vector<float> target_thrust_;
    double thrust_default_;
    std::deque<std::vector<float>> target_gimbal_list_;

    std::vector<float> last_action_; // will resize to action_size_

  };
}  // namespace aerial_robot_control

#endif  // BEETLE_RL_CONTROLLER_H