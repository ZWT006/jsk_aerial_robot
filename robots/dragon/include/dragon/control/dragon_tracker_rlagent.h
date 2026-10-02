//
// Created by wentao-zhang on 2026-07-11.
//
// Whole-body trajectory-tracking RL controller for DRAGON.
// Deployment counterpart of the Isaac Lab training env
// robots/dragon/policy/dragon_track_env.py (DragonTrackEnv); structure follows
// beetle_omni/beetle_pose_rlagent.
//

#ifndef DRAGON_TRACKER_RLAGENT_H
#define DRAGON_TRACKER_RLAGENT_H

#include "aerial_robot_control/control/base/base.h"
#include <onnxruntime_cxx_api.h>
#include <mutex>
#include <chrono>
#include <cmath>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include <std_msgs/Empty.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_srvs/Trigger.h>
#include <spinal/FourAxisCommand.h>
#include <sensor_msgs/JointState.h>
#include <nav_msgs/Odometry.h>
#include <aerial_robot_msgs/FullStateTarget.h>
#include <aerial_robot_model/ReloadPolicy.h>
#include <tf/transform_datatypes.h>

namespace aerial_robot_control
{

  class DragonTrackerRLAgent : public ControlBase
  {
  public:
    DragonTrackerRLAgent();
    ~DragonTrackerRLAgent() override;
    void initialize(ros::NodeHandle nh,
                    ros::NodeHandle nhp,
                    boost::shared_ptr<aerial_robot_model::RobotModel> robot_model,
                    boost::shared_ptr<aerial_robot_estimation::StateEstimator> estimator,
                    boost::shared_ptr<aerial_robot_navigation::BaseNavigator> navigator,
                    double ctrl_loop_du) override;
    bool update() override;
    void reset() override;

  protected:
    // ----------- Reference trajectory -----------
    // One reference frame: fc pose in (trajectory) world + the 6 link-joint positions
    // in sim order. Loaded from the CSV written by convert_trajectory2csv.py.
    struct RefFrame
    {
      tf::Vector3 pos;
      tf::Quaternion quat;
      std::vector<double> joint_pos;
    };

    bool loadTrajectory(const std::string& file_path);
    // triangle-wave reflection (training) or clamp-at-end, by loop_mode_
    double wrapPhase(double virtual_phase) const;
    // raw clip interpolation (lerp/slerp between the two neighbouring frames)
    void interpolateFrame(double frame_index, RefFrame& out) const;
    // interpolation + the rigid alignment transform (yaw rotation + translation)
    void interpolateAlignedFrame(double frame_index, RefFrame& out) const;
    // compute the align transform so that the frame at the current phase coincides
    // with the current fc pose (yaw + configured translation axes)
    bool alignTrajectory();
    // advance playback phase (TRACK mode) and refresh the K world-frame look-ahead
    // targets; mirrors _aars_set_motion_targets
    void updateReference();

    // ----------- RL agent -----------
    void initRLAgent(std::string model_path);
    bool usesIdealObservationSource() const;
    void configureIdealObservationSubscriber();
    void buildObservation();
    void policyForward();
    void sendCmd();
    void appendObsTerm(const std::string& term, std::vector<float>& out);
    size_t obsTermDim(const std::string& term) const;
    size_t actionGroupDim(const std::string& group) const;
    bool buildGimbalPositionLimits(double roll_limit, double pitch_limit,
                                   std::vector<double>& limits) const;
    bool loadEffortControlParams(const std::string& group_name,
                                 const std::vector<std::string>& names,
                                 std::vector<double>& kp,
                                 std::vector<double>& kd,
                                 std::vector<double>& effort_limit,
                                 std::vector<ros::Publisher>& effort_pubs);
    void applyJointAction(const float* action_slice);
    void applyGimbalAction(const float* action_slice);
    void applyThrustAction(const float* action_slice);
    // RSI-style action seed: action = (measured state - default) / scale, zero thrust.
    // Keeps the first commands holding the current configuration instead of snapping
    // the joints to zero. Caller must hold data_mutex_.
    void seedActionFromState();

    // ----------- ROS callbacks -----------
    void jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg);
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void targetStateCallback(const aerial_robot_msgs::FullStateTarget::ConstPtr& msg);
    void brakeCallback(const std_msgs::Empty::ConstPtr& msg);
    void unbrakeCallback(const std_msgs::Empty::ConstPtr& msg);
    void enableJointAndGimbalCallback(const std_msgs::Empty::ConstPtr& msg);
    void forceLandingCallback(const std_msgs::Empty::ConstPtr& msg);
    void haltCallback(const std_msgs::Empty::ConstPtr& msg);
    bool reloadPolicyCallback(aerial_robot_model::ReloadPolicy::Request& req,
                              aerial_robot_model::ReloadPolicy::Response& res);
    bool reloadParamsCallback(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);

    void publishDebug();

  private:
    ros::Subscriber joint_state_sub_, odom_sub_, target_state_sub_, brake_sub_, unbrake_sub_;
    ros::Subscriber enable_all_arm_sub_, enable_all_tracking_sub_;
    ros::Subscriber force_landing_sub_, halt_sub_;
    ros::Publisher thrust_pub_, joint_pub_, gimbal_pub_;
    std::vector<ros::Publisher> joint_effort_pubs_, gimbal_effort_pubs_;
    ros::Publisher thrust_debug_pub_, joint_debug_pub_, gimbal_debug_pub_;
    ros::Publisher obs_debug_pub_;
    ros::ServiceServer start_track_srv_, stop_track_srv_, reset_track_srv_;
    ros::ServiceServer reload_policy_srv_, reload_params_srv_;

    spinal::FourAxisCommand thrust_cmd_;
    sensor_msgs::JointState joint_cmd_;
    sensor_msgs::JointState gimbal_cmd_;

    std::mutex data_mutex_;

    // ----------- layout / mapping (from YAML, must match the training printout) -----------
    // sim-order actuator names; obs/action vectors follow these orders
    std::vector<std::string> joint_names_;        // 6, e.g. joint1_yaw, joint2_pitch, ...
    std::vector<std::string> gimbal_names_;       // 8, e.g. gimbal2_roll, gimbal2_pitch, ...
    std::vector<std::string> thrust_pair_names_;  // 4, e.g. edf2, edf1, edf3, edf4
    std::vector<int> pair_module_index_;          // pair k -> base_thrust index (edfN -> N-1)
    int joint_size_ = 6;
    int gimbal_size_ = 8;
    int pair_size_ = 4;
    int target_obs_horizon_ = 5;
    std::vector<double> joint_default_pos_;   // sim order; training default = 0
    std::vector<double> gimbal_default_pos_;

    // ----------- measured states (sim order) -----------
    std::vector<float> joint_pos_, joint_vel_;
    std::vector<float> gimbal_pos_, gimbal_vel_;
    // Previous policy-step positions used in the joint_vel/gimbal_vel observation
    // slots when velocity feedback is replaced by position history.
    std::vector<float> last_joint_pos_, last_gimbal_pos_;
    bool last_dof_pos_initialized_ = false;
    ros::Time joint_state_stamp_;
    bool joint_catch_ = false;
    bool gimbal_catch_ = false;
    bool joint_vel_from_msg_ = true;   // fall back to finite difference when msg has no velocity
    nav_msgs::Odometry odom_msg_;
    bool odom_catch_ = false;

    // ----------- target state FIFO -----------
    std::string target_state_topic_;
    std::deque<RefFrame> target_state_fifo_;
    bool target_state_catch_ = false;
    ros::Time target_state_stamp_;

    // ----------- legacy CSV reference trajectory helpers -----------
    // Kept only for source compatibility; the RL tracking path uses target_state_fifo_.
    std::vector<RefFrame> traj_;
    double traj_fps_ = 0.0;
    bool traj_loaded_ = false;
    std::string traj_file_;
    double phase_ = 0.0;         // playback phase [frames]
    double phase_rate_ = 0.0;    // frames advanced per policy step
    double start_frame_ = 0.0;
    bool tracking_ = false;      // false: HOLD (phase frozen, stride 0), true: TRACK
    std::string loop_mode_ = "hold";  // hold | pingpong
    // rigid alignment: p' = Rz(align_yaw_) p + align_trans_; q' = Rz(align_yaw_) q
    double align_yaw_ = 0.0;
    tf::Vector3 align_trans_{0.0, 0.0, 0.0};
    bool align_yaw_enable_ = true;
    bool align_xy_enable_ = true;
    bool align_z_enable_ = false;
    // world-frame look-ahead targets for the current policy step (size track_horizon_)
    std::vector<RefFrame> target_frames_;

    // ----------- observation / action -----------
    std::vector<float> observation_;
    std::vector<std::vector<float>> history_observations_;
    int history_length_ = 1;
    int single_obs_size_ = 0;
    std::vector<std::string> target_obs_terms_;
    std::vector<std::string> obs_terms_;
    std::vector<std::string> history_terms_;
    std::vector<std::string> action_groups_;
    std::vector<float> action_;
    std::vector<float> last_action_;
    size_t obs_size_ = 0;
    size_t action_size_ = 0;
    bool catch_obs_ = false;
    std::map<std::string, double> scales_;

    int step_count_ = 0;
    int decimation_ = 1;   // main loop 40 Hz == policy rate -> 1
    bool action_seeded_ = false;   // action_ holds a valid (measured-state) seed
    bool align_done_ = false;      // one-time startup alignment happened
    bool ideal_obs_ = false;
    bool ideal_pos_obs_ = true;
    bool ideal_quat_obs_ = true;
    bool ideal_lin_vel_obs_ = true;
    bool ideal_ang_vel_obs_ = true;
    bool replace_dof_vel_with_last_pos_ = false;
    std::string ideal_obs_topic_;
    std::string active_ideal_obs_topic_;
    bool ideal_odom_subscribed_ = false;
    int ideal_delay_ = 0;
    std::deque<tf::Vector3> lin_vel_list_, ang_vel_list_;
    bool verbose_ = false;
    bool debug_ = false;
    bool forward_info_ = false;

    // observation feature cache (filled by buildObservation)
    tf::Vector3 fc_pos_;
    tf::Quaternion fc_quat_;
    tf::Vector3 lin_vel_body_, ang_vel_body_;
    tf::Vector3 pos_error_;
    tf::Vector3 rpy_error_;
    double yaw_error_ = 0.0;
    double joint_error_ = 0.0;

    // ----------- command shaping -----------
    bool enable_thrust_ = false;
    bool enable_joint_ = false;
    bool enable_gimbal_ = false;
    bool joint_effort_ctrl_ = false;
    bool gimbal_effort_ctrl_ = false;
    bool action_clamp_ = true;    // clamp scaled joint/gimbal position targets
    double thrust_hover_ = 8.5;   // N per EDF, ctrlCfg.thrust_hover
    double thrust_limit_ = 16.0;  // N, rotor max
    double thrust_scale_default_ = 1.0;
    double joint_limit_ = 1.51;   // rad, command clamp
    double gimbal_roll_limit_ = M_PI;   // rad, symmetric command clamp
    double gimbal_pitch_limit_ = M_PI;  // rad, symmetric command clamp
    std::vector<double> gimbal_position_limits_;  // sim order, selected by name
    std::vector<double> joint_kp_, joint_kd_, joint_effort_limit_;
    std::vector<double> gimbal_kp_, gimbal_kd_, gimbal_effort_limit_;
    std::vector<float> target_joint_, target_gimbal_, target_thrust_;
    // optional command delay emulation (policy-step granularity), as in the beetle agent
    int joint_target_delay_steps_ = 0;
    int gimbal_target_delay_steps_ = 0;
    int thrust_target_delay_steps_ = 0;
    std::deque<std::vector<float>> target_joint_list_, target_gimbal_list_, target_thrust_list_;

    // ----------- ONNX Runtime -----------
    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::SessionOptions> sess_opts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    const char* input_name_ = nullptr;
    const char* output_name_ = nullptr;

    // inference timing stats
    size_t infer_count_ = 0;
    uint64_t infer_total_ns_ = 0;
    uint64_t infer_max_ns_ = 0;
    uint64_t infer_min_ns_ = std::numeric_limits<uint64_t>::max();
    ros::Time last_infer_report_time_;
  };

}  // namespace aerial_robot_control

#endif  // DRAGON_TRACKER_RLAGENT_H
