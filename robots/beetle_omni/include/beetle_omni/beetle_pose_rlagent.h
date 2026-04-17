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
#include <cstdint>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64.h>
#include <std_msgs/UInt32.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Empty.h>
#include <spinal/FourAxisCommand.h>
#include <spinal/Imu.h>
#include <sensor_msgs/JointState.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Transform.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <std_srvs/Trigger.h>
#include <beetle_omni/ReloadPolicy.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>


namespace aerial_robot_control
{

  class BeetlePoseRLAgent : public ControlBase
  {
  public:
    BeetlePoseRLAgent();
    ~BeetlePoseRLAgent() override;
    void initialize(ros::NodeHandle nh,
                            ros::NodeHandle nhp,
                            boost::shared_ptr<aerial_robot_model::RobotModel> robot_model,
                            boost::shared_ptr<aerial_robot_estimation::StateEstimator> estimator,
                            boost::shared_ptr<aerial_robot_navigation::BaseNavigator> navigator,
                            double ctrl_loop_du) override;
    bool update() override;
    void reset() override;
    // void activeate() override;
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
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void imuCallback(const spinal::Imu::ConstPtr& msg);
    void rotorFaultCallback(const std_msgs::UInt32::ConstPtr& msg);
    void brakeCallback(const std_msgs::Empty::ConstPtr& msg);
    void unbrakeCallback(const std_msgs::Empty::ConstPtr& msg);
    bool reloadPolicyCallback(beetle_omni::ReloadPolicy::Request &req,
                              beetle_omni::ReloadPolicy::Response &res);
    bool reloadParamsCallback(std_srvs::Trigger::Request &req,
                              std_srvs::Trigger::Response &res);

  private:
    ros::Subscriber goal_sub_, gimbal_sub_, odom_sub_, imu_sub_;
    ros::Subscriber rotor_fault_sub_, brake_sub_, unbrake_sub_;
    ros::Publisher thrust_pub_, thrust_debug_pub_, gimbal_pub_, gimbal_debug_pub_;
    ros::Publisher gimbal_effort_pub1_, gimbal_effort_pub2_, gimbal_effort_pub3_, gimbal_effort_pub4_;
    ros::Publisher obs_debug_pub_, rotor_halt_mask_pub_;
    ros::ServiceServer reload_policy_srv_;
    ros::ServiceServer reload_params_srv_;
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
    std::deque<tf::Vector3> ang_vel_list_;
    std::deque<tf::Vector3> lin_vel_list_;

    // ----------- general parameters -----------
    std::vector<float> observation_;
    std::vector<std::vector<float>> history_observations_;
    std::vector<float> action_;
    std::vector<float> last_action_; // will resize to action_size_
    double control_hz_;
    size_t obs_size_;
    size_t action_size_;
    int single_obs_size_;
    int history_length_;
    bool history_obs_ = false;
    bool ideal_obs_ = false;
    bool fault_obs_ = false;
    bool fault_goal_ = true;
    int step_count_ = 0;
    int decimation_ = 4;
    bool verbose_ = false;
    bool debug_ = false;
    bool forward_info_ = false;
    int ideal_delay_ = 4;

    std::map<std::string, double> scales;

    tf::Vector3 pos_error;
    tf::Vector3 ang_error;
    tf::Vector3 pos_body,ang_body,pos_goal,ang_goal;

    bool fc2root_transform_ = false;
    tf::Vector3 root2fc_pos_;
    tf::Quaternion root2fc_quat_;
    geometry_msgs::Transform fc2root_T;

    // ----------- ONNX Runtime -----------
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> sess_opts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    const char* input_name_ = nullptr;
    const char* output_name_ = nullptr;
    bool catch_obs_ = false;

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
    bool fault_injection_ = false;

    int gimbal_size_;
    int thrust_size_;

    int gimbal_target_delay_steps_ = 0;
    int gimbal_obs_delay_steps_ = 0;

    std::vector<float> gimbal_pos_;
    std::vector<float> gimbal_vel_;
    std::vector<float> gimbal_default_pos_;
    std::vector<std::string> gimbal_names_;
    std::deque<std::vector<float>> gimbal_pos_list_;
    double thrust_default_;
    double thrust_limit_;
    double thrust_tau_;
    std::deque<std::vector<float>> target_gimbal_list_;
    std::vector<float> target_gimbal_;
    std::vector<float> target_gimbal_effort_;
    std::deque<std::vector<float>> target_thrust_list_;
    std::vector<float> target_thrust_;
    std::vector<float> thrust_scale_;
    int thrust_target_delay_steps_ = 0;
    double thrust_scale_default_ = 0.0;
    uint32_t rotor_halt_mask_ = 0;
    double gimbal_kp_;
    double gimbal_kd_;
    bool gimbal_effort_ctrl_ = false;
  };
}  // namespace aerial_robot_control

#endif  // BEETLE_RL_CONTROLLER_H
