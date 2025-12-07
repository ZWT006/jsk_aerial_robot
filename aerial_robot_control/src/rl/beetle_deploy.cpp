#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Transform.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/UInt8.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Empty.h>
#include <deque>
#include <tf/transform_datatypes.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <mutex>
#include <vector>
#include <chrono>
#include <iomanip>
#include <iostream>

// replace with actual package message header if different
#include <spinal/FourAxisCommand.h>
#include <spinal/Imu.h>

// ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>
#include <cmath> // for std::isfinite
#include <math.h> // for M_PI


using spinal::FourAxisCommand;
using spinal::Imu;

tf2_ros::Buffer tf2_buffer;


geometry_msgs::Transform inverseTransform(const geometry_msgs::Transform &Transform)
{
    geometry_msgs::Transform inverse_transform;
    tf2::Quaternion quaternion(Transform.rotation.x, Transform.rotation.y, Transform.rotation.z, Transform.rotation.w);
    tf2::Vector3 translation_vector(Transform.translation.x, Transform.translation.y, Transform.translation.z);
    tf2::Quaternion inverse_quaternion = quaternion.inverse();

    tf2::Matrix3x3 quaternion_matrix(inverse_quaternion);
    tf2::Vector3 inverse_tanslation_vector = quaternion_matrix * translation_vector;
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
    tf2::Quaternion q_A2B(A2B.rotation.x, A2B.rotation.y, A2B.rotation.z, A2B.rotation.w);
    tf2::Quaternion q_B2C(B2C.rotation.x, B2C.rotation.y, B2C.rotation.z, B2C.rotation.w);
    tf2::Quaternion q_A2C = q_A2B * q_B2C;
    tf2::Vector3 t_A2B(A2B.translation.x, A2B.translation.y, A2B.translation.z);
    tf2::Vector3 t_B2C(B2C.translation.x, B2C.translation.y, B2C.translation.z);
    tf2::Matrix3x3 R_A2B(q_A2B);
    tf2::Vector3 t_A2C = t_A2B + R_A2B * t_B2C;

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

class PolicyDeployer {
public:
    PolicyDeployer(const ros::NodeHandle& nh, const std::string& onnx_path, int control_hz = 50, int decimation = 4)
        : nh_(nh),
          control_hz_(control_hz),
          obs_size_(0),
          action_size_(0),
          decimation_(decimation)
    {
        // Subscribers
        imu_sub_ = nh_.subscribe("imu", 1, &PolicyDeployer::_imu_callback, this);
        odom_sub_ = nh_.subscribe("odometry", 1, &PolicyDeployer::_odom_callback, this);
        gimbal_sub_ = nh_.subscribe("joint_states", 1, &PolicyDeployer::_gimbal_callback, this);
        goal_sub_ = nh_.subscribe("goal_pose", 1, &PolicyDeployer::_goal_pose_callback, this);

        // Subscribers from keyboard or other manual control cmds
        std::string bake_topic = "/gimbalrotor/teleop_command";
        halt_sub_           = nh_.subscribe(bake_topic + "/halt",           1, &PolicyDeployer::_control_disable_callback, this);
        land_sub_           = nh_.subscribe(bake_topic + "/land",           1, &PolicyDeployer::_control_disable_callback, this);
        start_sub_          = nh_.subscribe(bake_topic + "/start",          1, &PolicyDeployer::_control_disable_callback, this);
        takeoff_sub_        = nh_.subscribe(bake_topic + "/takeoff",        1, &PolicyDeployer::_control_enable_callback, this);
        force_landing_sub_  = nh_.subscribe(bake_topic + "/force_landing",  1, &PolicyDeployer::_control_disable_callback, this);

        // Publishers
        thrust_pub_ = nh_.advertise<FourAxisCommand>("/gimbalrotor/four_axes/command", 1);
        thrust_debug_pub_ = nh_.advertise<FourAxisCommand>("/gimbalrotor/four_axes/command_debug", 1);
        gimbal_pub_ = nh_.advertise<sensor_msgs::JointState>("/gimbalrotor/gimbals_ctrl", 1);
        gimbal_debug_pub_ = nh_.advertise<sensor_msgs::JointState>("/gimbalrotor/gimbals_ctrl_debug", 1);
        obs_debug_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("/gimbalrotor/observation_debug", 1);

        // defaults
        gimbal_default_pos_ = {0.0f, 0.0f, 0.0f, 0.0f};
        thrust_default_ = 7.0f;
        step_count_ = 0;
        decimation_ = 4;
        desired_pose_.header.frame_id = "world";
        desired_pose_.pose.position.x = 0.0;
        desired_pose_.pose.position.y = 0.0;
        desired_pose_.pose.position.z = 0.45;
        tf::Quaternion quat = tf::createQuaternionFromRPY(0.0, 0.0, 0.0);
        desired_pose_.pose.orientation.x = quat.x();
        desired_pose_.pose.orientation.y = quat.y();
        desired_pose_.pose.orientation.z = quat.z();
        desired_pose_.pose.orientation.w = quat.w();

        // ONNX Runtime init
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "beetle");
        sess_opts_ = std::make_unique<Ort::SessionOptions>();
        sess_opts_->SetIntraOpNumThreads(2);
        sess_opts_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(*env_, onnx_path.c_str(), *sess_opts_);

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

        ROS_INFO("ONNX model input size: %zu, output size: %zu", obs_size_, action_size_);
        // ensure containers sized accordingly
        last_action_.assign(action_size_, 0.0f);
        target_gimbal_.assign(std::min<size_t>(4, action_size_), 0.0f);
        target_thrust_.assign(std::max<size_t>(4, action_size_ > 4 ? action_size_ - 4 : 4), 0.0f);
        // control loop timer will be started after initial data received
        ROS_INFO("Target Gimbal and Thrust vector sizes: %zu, %zu", target_gimbal_.size(), target_thrust_.size()); 
        ROS_INFO("PolicyDeployer constructed, waiting sensors before starting loop.");
        // init inference timing
        infer_count_ = 0;
        infer_total_ns_ = 0;
        infer_max_ns_ = 0;
        infer_min_ns_ = std::numeric_limits<uint64_t>::max();
        last_infer_report_time_ = ros::Time::now();
    }

    ~PolicyDeployer() {
        // free allocated C strings from GetInputNameAllocated / GetOutputNameAllocated
        if (input_name_) allocator_.Free(const_cast<char*>(input_name_));
        if (output_name_) allocator_.Free(const_cast<char*>(output_name_));
    }

    void setTranform(const geometry_msgs::Transform &cog2root_T_in) {
        cog2root_T = cog2root_T_in;
    }

    void setDesiredPose(const geometry_msgs::PoseStamped &desired_pose_in) {
        desired_pose_ = desired_pose_in;
    }

    void setControlEnable(bool thrust_enable, bool gimbal_enable) {
        enable_thrust_ = thrust_enable;
        enable_gimbal_ = gimbal_enable;
    }

    void lockControl(bool thrust_lock, bool gimbal_lock) {
        lock_thrust_ = thrust_lock;
        lock_gimbal_ = gimbal_lock;
    }

    void setGimbalDelaySteps(int target_delay_steps, int obs_delay_steps) {
        gimbal_target_delay_steps_ = target_delay_steps;
        gimbal_obs_delay_steps_ = obs_delay_steps;
        // clear any previous buffer and optionally pre-fill with current target (to avoid empty front)
        std::lock_guard<std::mutex> lk(data_mutex_);
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
        ROS_INFO("Set gimbal target delay steps: %d, obs delay steps: %d", gimbal_target_delay_steps_, gimbal_obs_delay_steps_);
    }

    void spin() {
        // wait for initial messages (timeout)
        ros::Time start = ros::Time::now();
        ros::Duration timeout(500.0);
        ros::Rate r(50);
        setControlEnable(false, false); // start with gimbal only control
        while (ros::ok()) {
            {
                std::lock_guard<std::mutex> lk(data_mutex_);
                if (imu_catch_ && odom_catch_ && gimbal_catch_) break;
            }
            // if (ros::Time::now() - start > timeout) {
            //     ROS_WARN("Timeout waiting for initial sensor messages; continuing anyway");
            //     break;
            // }
            ros::spinOnce();
            r.sleep();
        }

        // start control loop (Rate-based)
        ROS_INFO("Starting control loop at %.1f Hz", control_hz_);
        ros::Rate control_rate(control_hz_);
        while (ros::ok()) {
            ros::Time t0 = ros::Time::now();
            step_count_++;
            if (step_count_ % decimation_ == 0) {
                _control_step();
            }
            if (target_gimbal_.size() != 4) {
                ROS_WARN_THROTTLE(5.0, "target_gimbal_ size (%zu) != 4, something wrong!!!", target_gimbal_.size());
            }
            publish_thrust(target_thrust_);
            publish_gimbal(target_gimbal_);
            publish_observation_debug();
            ros::Time now = ros::Time::now();
            double elapsed = (now - last_infer_report_time_).toSec();
            if (elapsed >= 2.0 && infer_count_ > 0) {
                double avg_ns = static_cast<double>(infer_total_ns_) / static_cast<double>(infer_count_);
                double avg_ms = avg_ns / 1e6;
                double min_ms = static_cast<double>(infer_min_ns_) / 1e6;
                double max_ms = static_cast<double>(infer_max_ns_) / 1e6;
                double avg_period_ms = (elapsed / static_cast<double>(infer_count_)) * 1000.0;
                ROS_INFO("\n"
                    "ORT Inference (last %.2fs): calls=%lu\n"
                    "  avg_time=%.3f ms, avg_period=%.3f ms\n"
                    "  min=%.3f ms, max=%.3f ms\n"
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
                    "Pos Error: [%.3f, %.3f, %.3f]\n"
                    "Ang Error: [%.3f, %.3f, %.3f]\n"
                    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
                    elapsed,
                    static_cast<unsigned long>(infer_count_),
                    avg_ms,
                    avg_period_ms,
                    min_ms,
                    max_ms,
                    pos_error.x(), pos_error.y(), pos_error.z(),
                    ang_error.x(), ang_error.y(), ang_error.z()
                );
                // reset counters
                infer_count_ = 0;
                infer_total_ns_ = 0;
                infer_max_ns_ = 0;
                infer_min_ns_ = std::numeric_limits<uint64_t>::max();
                last_infer_report_time_ = now;
            }
            ros::spinOnce();  // ensure callbacks processed
            control_rate.sleep();
            // optional: detect large loop overrun
        }
    }

private:
    // -------- ROS
    ros::NodeHandle nh_;
    ros::Subscriber imu_sub_, odom_sub_, gimbal_sub_, goal_sub_;
    ros::Subscriber control_enable_sub_, halt_sub_, land_sub_, start_sub_, takeoff_sub_, force_landing_sub_;
    ros::Publisher thrust_pub_, thrust_debug_pub_, gimbal_pub_, gimbal_debug_pub_;
    ros::Publisher obs_debug_pub_;

    // -------- data
    std::mutex data_mutex_;
    Imu imu_msg_;
    bool imu_catch_ = false;
    nav_msgs::Odometry odom_msg_;
    bool odom_catch_ = false;
    sensor_msgs::JointState gimbal_msg_;
    bool gimbal_catch_ = false;
    geometry_msgs::PoseStamped desired_pose_;
    geometry_msgs::Transform cog2root_T;
    geometry_msgs::Transform world2cog_T;
    geometry_msgs::Transform world2root_T;

    tf::Vector3 pos_error;
    tf::Vector3 ang_error;

    bool enable_thrust_ = false;
    bool lock_thrust_ = false;
    bool enable_gimbal_ = false;
    bool lock_gimbal_ = false;

    int gimbal_target_delay_steps_ = 0;
    int gimbal_obs_delay_steps_ = 0;

    std::vector<float> gimbal_pos_ = std::vector<float>(4, 0.0f);
    std::deque<std::vector<float>> gimbal_pos_list_;
    std::vector<float> gimbal_default_pos_;
    std::vector<float> last_action_; // will resize to action_size_

    // -------- control
    double control_hz_;
    size_t obs_size_;
    size_t action_size_;
    int step_count_ = 0;
    int decimation_ = 4;

    // ONNX inference timing stats
    size_t infer_count_ = 0;
    uint64_t infer_total_ns_ = 0;
    uint64_t infer_max_ns_ = 0;
    uint64_t infer_min_ns_ = std::numeric_limits<uint64_t>::max();
    ros::Time last_infer_report_time_;

    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> sess_opts_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    const char* input_name_ = nullptr;
    const char* output_name_ = nullptr;

    // publishers data
    std::vector<float> target_thrust_ = std::vector<float>(4, 0.0f);
    std::vector<float> target_gimbal_ = std::vector<float>(4, 0.0f);
    std::deque<std::vector<float>> target_gimbal_list_;
    std::vector<float> observation_ = std::vector<float>(36, 0.0f);
    float thrust_default_;

    // ---------------- callbacks ----------------
    void _imu_callback(const Imu::ConstPtr& msg) {
        std::lock_guard<std::mutex> lk(data_mutex_);
        imu_msg_ = *msg;
        imu_catch_ = true;
    }

    void _odom_callback(const nav_msgs::Odometry::ConstPtr& msg) {
        std::lock_guard<std::mutex> lk(data_mutex_);
        odom_msg_ = *msg;
        world2cog_T.translation.x = odom_msg_.pose.pose.position.x;
        world2cog_T.translation.y = odom_msg_.pose.pose.position.y;
        world2cog_T.translation.z = odom_msg_.pose.pose.position.z;
        world2cog_T.rotation = odom_msg_.pose.pose.orientation;
        world2root_T = forwardTransform(world2cog_T, cog2root_T);
        // world2root_T = world2cog_T;
        odom_catch_ = true;
    }

    void _gimbal_callback(const sensor_msgs::JointState::ConstPtr& msg) {
        std::lock_guard<std::mutex> lk(data_mutex_);
        gimbal_msg_ = *msg;
        // copy positions safely
        for (size_t i = 0; i < std::min<size_t>(msg->position.size(), gimbal_pos_.size()); ++i) {
            gimbal_pos_[i] = msg->position[i];
        }
        gimbal_catch_ = true;
        if (gimbal_obs_delay_steps_ > 0) {
            // maintain history buffer
            gimbal_pos_list_.push_back(gimbal_pos_);
            if (gimbal_pos_list_.size() > static_cast<size_t>(gimbal_obs_delay_steps_)) {
                gimbal_pos_ = gimbal_pos_list_.front();
                gimbal_pos_list_.pop_front();
            }
        }
    }

    void _goal_pose_callback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        {
            std::lock_guard<std::mutex> lk(data_mutex_);
            desired_pose_ = *msg;
        }
        const auto& p = msg->pose.position;
        std::cout << "============ Received new goal pose ============\n";
        std::cout << std::fixed << std::setprecision(4)
                  << "Position: x=" << p.x << ", y=" << p.y << ", z=" << p.z << "\n";
        tf::Quaternion q(msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z, msg->pose.orientation.w);
        double roll, pitch, yaw;
        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
        std::cout << "Angle: [" << std::fixed << std::setprecision(4)
                  << roll << ", " << pitch << ", " << yaw << "]\n";
    }
    // -------- Control Enable/Disable Callbacks --------
    void _control_enable_callback(const std_msgs::Empty::ConstPtr& msg) {
        setControlEnable(true, true);
        ROS_INFO("Control enabled via control_enable topic.");
    }

    void _control_disable_callback(const std_msgs::Empty::ConstPtr& msg) {
        setControlEnable(false, false);
        ROS_INFO("Control disabled via control_enable topic.");
    }

    // -------- helpers --------
    tf::Vector3 rotate_by_quat_inv(const tf::Quaternion& q, const tf::Vector3& v) {
        tf::Quaternion qinv = q.inverse();
        tf::Matrix3x3 R(qinv);
        return R * v;
    }

    // Build observation similar to Python: ensure final length obs_size_ (39) by inserting zeros if needed
    bool build_observation(std::vector<float>& out_obs) {
        std::lock_guard<std::mutex> lk(data_mutex_);
        if (!odom_catch_ || !imu_catch_ || !gimbal_catch_) return false;

        // body quaternion and pos
        tf::Quaternion body_quat(world2root_T.rotation.x,
                                 world2root_T.rotation.y,
                                 world2root_T.rotation.z,
                                 world2root_T.rotation.w);
        // tf::Quaternion body_quat(odom_msg_.pose.pose.orientation.x,
        //                          odom_msg_.pose.pose.orientation.y,
        //                          odom_msg_.pose.pose.orientation.z,
        //                          odom_msg_.pose.pose.orientation.w);
        body_quat.normalize();

        tf::Vector3 body_pos(world2root_T.translation.x,
                             world2root_T.translation.y,
                             world2root_T.translation.z);
        // tf::Vector3 body_pos(odom_msg_.pose.pose.position.x,
        //                      odom_msg_.pose.pose.position.y,
        //                      odom_msg_.pose.pose.position.z);
        // linear velocity of COM in body frame
        tf::Vector3 lin_vel_world(odom_msg_.twist.twist.linear.x,
                                  odom_msg_.twist.twist.linear.y,
                                  odom_msg_.twist.twist.linear.z);
        tf::Vector3 lin_vel_body = rotate_by_quat_inv(body_quat, lin_vel_world);
        // tf::Vector3 lin_vel_body = lin_vel_world;

        tf::Vector3 ang_vel_world(odom_msg_.twist.twist.angular.x,
                                  odom_msg_.twist.twist.angular.y,
                                  odom_msg_.twist.twist.angular.z);
        // tf::Vector3 ang_vel_body = rotate_by_quat_inv(body_quat, ang_vel_world);
        tf::Vector3 ang_vel_body = ang_vel_world;

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
        tf::Vector3 t02_minus_t01 = goal_world - body_pos;
        tf::Vector3 goal_pos = rotate_by_quat_inv(body_quat, t02_minus_t01);
        goal_quat = body_quat.inverse() * goal_quat;

        pos_error = goal_pos;
        tf::Matrix3x3 goal_rot_mat(goal_quat);
        double goal_roll, goal_pitch, goal_yaw;
        goal_rot_mat.getRPY(goal_roll, goal_pitch, goal_yaw);
        ang_error = tf::Vector3(goal_roll, goal_pitch, goal_yaw);

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

        // last_action placeholder (use zeros if none)
        std::vector<float> last_action(8, 0.0f);
        if (!last_action_.empty()) {
            for (size_t i = 0; i < std::min<size_t>(8, last_action_.size()); ++i) last_action[i] = last_action_[i];
        }

        out_obs.reserve(obs_size_);
        // lin vel (3) 3
        out_obs.push_back(lin_vel_body.x() * 1.0f); out_obs.push_back(lin_vel_body.y() * 1.0f); out_obs.push_back(lin_vel_body.z() * 1.0f);
        // ang vel (3) 6
        out_obs.push_back(ang_vel_body.x() * 0.2f); out_obs.push_back(ang_vel_body.y() * 0.2f); out_obs.push_back(ang_vel_body.z() * 0.2f);
        // gravity (3) 9
        out_obs.push_back(gravity_b.x()); out_obs.push_back(gravity_b.y()); out_obs.push_back(gravity_b.z());
        // goal pos (3) 12
        out_obs.push_back(goal_pos.x()); out_obs.push_back(goal_pos.y()); out_obs.push_back(goal_pos.z());
        // gimbal dof (4) 16
        for (size_t i = 0; i < 4; ++i) out_obs.push_back(gimbal_pos_[i]);
        // root_rot_vec (6) 22
        out_obs.insert(out_obs.end(), root_rot_vec.begin(), root_rot_vec.end());
        // goal_rot_vec (6) 28
        out_obs.insert(out_obs.end(), goal_rot_vec.begin(), goal_rot_vec.end());
        // last_action (8) 36
        out_obs.insert(out_obs.end(), last_action.begin(), last_action.end());

        // verify obs36 length (should be 36)
        if (out_obs.size() != 36) return false;
        return out_obs.size() == obs_size_;
    }

    // -------- control step and inference --------
    void _control_step() {
        std::vector<float> obs;
        if (!build_observation(obs)) {
            ROS_WARN_THROTTLE(5.0, "Waiting for sensor data to build observation");
            return;
        }

        // If model expects different input length, attempt to adapt (pad/truncate)
        if (obs.size() != obs_size_) {
            if (obs.size() < obs_size_) {
                // pad with zeros
                obs.resize(obs_size_, 0.0f);
                ROS_WARN_THROTTLE(5.0, "Observation length (%zu) < model input (%zu). Padding with zeros.", obs.size(), obs_size_);
            } else {
                // truncate
                obs.resize(obs_size_);
                ROS_WARN_THROTTLE(5.0, "Observation length (%zu) > model input (%zu). Truncating.", obs.size(), obs_size_);
            }
        }

        // Sanitize inputs: replace NaN/Inf
        size_t bad_count = 0;
        std::vector<int> bad_indices;
        for (size_t i = 0; i < obs.size(); ++i) {
            if (!std::isfinite(obs[i])) {
                obs[i] = 0.0f;
                bad_indices.push_back(static_cast<int>(i));
                ++bad_count;
            }
        }
        if (bad_count > 0) {
            ROS_WARN("Found %zu non-finite values in observation, replaced with 0.0", bad_count);
            std::ostringstream ss;
            ss << "Indices: ";
            for (size_t i = 0; i < bad_indices.size(); ++i) {
                ss << bad_indices[i] << (i + 1 < bad_indices.size() ? ", " : "");
            }
            ROS_WARN_STREAM(ss.str());
        }

        // Optional debug: print first elements
        if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info)) {
            std::ostringstream ss;
            ss << "Obs[0..min(8,N)]:";
            for (size_t i = 0; i < std::min<size_t>(8, obs.size()); ++i) ss << " " << std::fixed << std::setprecision(6) << obs[i];
            ROS_DEBUG_STREAM(ss.str());
        }

        // ONNX inference
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
            ROS_ERROR("ONNX runtime returned zero-length output");
            return;
        }

        // Check for NaN/Inf in output immediately
        bool out_bad = false;
        for (size_t i = 0; i < out_len; ++i) {
            if (!std::isfinite(out_ptr[i])) { out_bad = true; break; }
        }
        if (out_len != action_size_) {
            ROS_WARN("ONNX runtime output length (%zu) differs from expected action_size_ (%zu)", out_len, action_size_);
            out_bad = true;
        }
        if (out_bad) {
            std::ostringstream ss;
            ss << "Model output contains non-finite values: [";
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
            ROS_WARN("Model output length (%zu) differs from stored action_size_ (%zu). Updating action_size_.", out_len, action_size_);
            action_size_ = out_len;
            last_action_.assign(action_size_, 0.0f);
        }
        // save last_action_
        {
            std::lock_guard<std::mutex> lk(data_mutex_);
            last_action_ = action;
        }

        // parse target gimbal / thrust safely
        size_t num_gimbal = std::min<size_t>(4, action.size());
        for (size_t i = 0; i < num_gimbal; ++i) target_gimbal_[i] = action[i];
        for (size_t i = 0; i < std::min<size_t>(4, action.size() - 4); ++i) target_thrust_[i] = action[4 + i];
        for (size_t i = 0; i < obs.size(); ++i) observation_[i] = obs[i];
    }

    void publish_observation_debug() {
        std_msgs::Float32MultiArray msg;
        msg.data = observation_;
        msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        msg.layout.dim[0].label = "observation";
        msg.layout.dim[0].size = observation_.size();
        msg.layout.dim[0].stride = observation_.size();
        obs_debug_pub_.publish(msg);
    }

    void publish_thrust(const std::vector<float>& thrust) {
        spinal::FourAxisCommand msg;
        std::vector<float> scaled = thrust;
        msg.base_thrust.resize(scaled.size());
        for (size_t i = 0; i < scaled.size(); ++i){ 
		msg.base_thrust[i] = scaled[i] * 1.25f + thrust_default_;
		// msg.base_thrust[i] = msg.base_thrust[i] * 0.2;
	    }
        if (msg.base_thrust.size() != 4)
            ROS_WARN("=========================================\n"
                     "====  Thrust Size isn't same as 4! ======\n"
                     "=========================================");
        msg.angles = {0.0f, 0.0f, 0.0f};
        thrust_debug_pub_.publish(msg);
        if (!enable_thrust_ || lock_thrust_) return;
        thrust_pub_.publish(msg);
    }

    void publish_gimbal(std::vector<float>& target_pos) {
        sensor_msgs::JointState msg;
        msg.header.stamp = ros::Time::now();
        size_t num = target_pos.size();

        // prepare names safely
        std::vector<std::string> names;
        {
            std::lock_guard<std::mutex> lk(data_mutex_);
            if (gimbal_msg_.name.size() != num) {
                // fallback names
                ROS_WARN_THROTTLE(5.0, "gimbal_msg_.name size (%zu) != target_pos size (%zu), something wrong!!!", gimbal_msg_.name.size(), num);
                return;
            } else {
                names = gimbal_msg_.name;
            }
        }

        if (gimbal_target_delay_steps_ > 0) {
            std::lock_guard<std::mutex> lk(data_mutex_);
            // push latest (current) desired into buffer
            target_gimbal_list_.push_back(target_pos);
            // when buffer grows beyond the requested delay, pop the oldest and use it as the delayed command
            if (target_gimbal_list_.size() > static_cast<size_t>(gimbal_target_delay_steps_)) {
                target_pos = target_gimbal_list_.front();
                target_gimbal_list_.pop_front();
            } else {
                // buffer not yet filled to required depth: keep using initial/default target_gimbal_ (or choose other policy)
            }
        }

        msg.name = names;
        msg.position.resize(num);
        for (size_t i = 0; i < num; ++i) {
            float default_pos = (i < gimbal_default_pos_.size()) ? gimbal_default_pos_[i] : 0.0f;
            msg.position[i] = target_pos[i] * 0.25f + default_pos;
        }
        gimbal_debug_pub_.publish(msg);
        if (!enable_gimbal_ || lock_gimbal_) return;
        gimbal_pub_.publish(msg);

        // warn once in a while if names missing
        {
            std::lock_guard<std::mutex> lk(data_mutex_);
            if (gimbal_msg_.name.size() < num) {
                static ros::Time last_warn = ros::Time(0);
                if ((ros::Time::now() - last_warn).toSec() > 5.0) {
                    ROS_WARN("gimbal_msg.name length (%zu) < target_pos length (%zu), using fallback names", gimbal_msg_.name.size(), num);
                    last_warn = ros::Time::now();
                }
            }
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "beetle_deploy");
    ros::NodeHandle nh("~");

    std::string onnx_path;
    int freq;
    int decimation;
    int gimbal_target_delay_steps;
    int gimbal_obs_delay_steps;
    double goal_pos_x, goal_pos_y, goal_pos_z;
    double goal_ang_R, goal_ang_P, goal_ang_Y;
    bool enable_thrust_, enable_gimbal_;
    geometry_msgs::PoseStamped desired_pose;
    nh.param<std::string>("onnx_path", onnx_path, std::string(""));
    nh.param<int>("control_freq", freq, 200);
    nh.param<int>("decimation", decimation, 4);
    nh.param<int>("gimbal_target_delay_steps", gimbal_target_delay_steps, 0);
    nh.param<int>("gimbal_obs_delay_steps", gimbal_obs_delay_steps, 0);
    nh.param<double>("goal_pos_x", goal_pos_x, 0.0);
    nh.param<double>("goal_pos_y", goal_pos_y, 0.0);
    nh.param<double>("goal_pos_z", goal_pos_z, 0.6);
    nh.param<double>("goal_ang_R", goal_ang_R, 0.0);
    nh.param<double>("goal_ang_P", goal_ang_P, 0.0);
    nh.param<double>("goal_ang_Y", goal_ang_Y, M_PI_2);
    nh.param<bool>("enable_thrust", enable_thrust_, false);
    nh.param<bool>("enable_gimbal", enable_gimbal_, false);
    desired_pose.header.frame_id = "world";
    desired_pose.pose.position.x = goal_pos_x;
    desired_pose.pose.position.y = goal_pos_y;
    desired_pose.pose.position.z = goal_pos_z;
    tf::Quaternion quat = tf::createQuaternionFromRPY(goal_ang_R, goal_ang_P, goal_ang_Y);
    desired_pose.pose.orientation.x = quat.x();
    desired_pose.pose.orientation.y = quat.y();
    desired_pose.pose.orientation.z = quat.z();
    desired_pose.pose.orientation.w = quat.w();

    ROS_INFO("*************** Deploying ONNX Policy ****************");
    ROS_INFO("Control frequency: %d Hz", freq);
    ROS_INFO("Control decimation: %d", decimation);
    ROS_INFO("Initial desired pose: pos(%.2f, %.2f, %.2f), ang(R=%.2f, P=%.2f, Y=%.2f)",
             goal_pos_x, goal_pos_y, goal_pos_z,
             goal_ang_R, goal_ang_P, goal_ang_Y);
    if (onnx_path.empty()) {
        ROS_ERROR("Parameter ~onnx_path is unset or empty. Please provide a valid ONNX model path.");
        return 1;
    }else {
        ROS_INFO("Using ONNX model path: %s", onnx_path.c_str());
    }
    
    // Wait and get cog2root transform
    ROS_INFO("Waiting for transform from gimbalrotor/root to gimbalrotor/cog...");
    tf2_ros::TransformListener tf2_listener(tf2_buffer);
    geometry_msgs::TransformStamped lookupTransform;
    bool get_initial_transform = false;
    while (ros::ok() && !get_initial_transform) {
        try {
            lookupTransform = tf2_buffer.lookupTransform("gimbalrotor/cog","gimbalrotor/root",ros::Time(0));
            ROS_INFO("Successfully loaded cog2root transform:");
            ROS_INFO("  Translation: [%.4f, %.4f, %.4f]", 
                        lookupTransform.transform.translation.x, lookupTransform.transform.translation.y, lookupTransform.transform.translation.z);
            ROS_INFO("  Rotation (quat): [%.4f, %.4f, %.4f, %.4f]",
                        lookupTransform.transform.rotation.x, lookupTransform.transform.rotation.y, 
                        lookupTransform.transform.rotation.z, lookupTransform.transform.rotation.w);
            get_initial_transform = true;
        } catch (tf2::TransformException &ex) {
            ROS_WARN("Failed to get cog2root transform: %s", ex.what());
            ROS_WARN("Waiting and retrying...");
            lookupTransform.transform.translation.x = 0.0;
            lookupTransform.transform.translation.y = 0.0;
            lookupTransform.transform.translation.z = 0.0;
            lookupTransform.transform.rotation.x = 0.0;
            lookupTransform.transform.rotation.y = 0.0;
            lookupTransform.transform.rotation.z = 0.0;
            lookupTransform.transform.rotation.w = 1.0;
            ros::Duration(1.0).sleep(); // wait before retrying
        }
    }
    try {
        PolicyDeployer deployer(nh, onnx_path, freq);
        deployer.setTranform(lookupTransform.transform);
        deployer.setDesiredPose(desired_pose);
        deployer.setControlEnable(enable_thrust_, enable_gimbal_);
        deployer.lockControl(!enable_thrust_, !enable_gimbal_);
        deployer.setGimbalDelaySteps(gimbal_target_delay_steps, gimbal_obs_delay_steps);
        deployer.spin();
    } catch (const std::exception& e) {
        ROS_ERROR("Exception: %s", e.what());
        return 1;
    }

    return 0;
}
