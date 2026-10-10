# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import math
import os
from collections.abc import Sequence

from aerial_lab.utility.aerialutility import print_training_config
import gymnasium as gym
import isaaclab.envs.mdp as mdp
import aerial_lab.tasks.manager_based.velocity.mdp.rewards as LocomotionVelRewards
import isaaclab.sim as sim_utils
import torch
from aerial_lab.utility.delaymodel import DelayModel, DelayModelCfg
from aerial_lab.utility.estimatormodel import DragonStateEstimatorModel, DragonStateEstimatorModelCfg
from aerial_lab.utility.utilitymath import (
    degradeSymmetryQuatwithoutTilt,
    sampleCenterQuatwithTilt,
    sampleSymmetryQuatwithTilt,
    sampleSymmetryQuatwithTiltforEnv,
    sampleUniformQuatwithTilt,
    sampleHalfsphereQuatwithTilt,
    sampleUniformQuatwithYaw,
)
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, VecEnvObs, VecEnvStepReturn
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, Imu, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.terrains import TerrainImporterCfg, MeshRandomGridTerrainCfg, TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_euler,
    matrix_from_quat,
    normalize,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
    sample_uniform,
    euler_xyz_from_quat,
    subtract_frame_transforms,
)

from aerial_lab.assets import ISAACLAB_ASSETS_DATA_DIR  # isort: skip
from aerial_lab.assets.aerialrobot import DRAGON_CFG  # isort: skip
from aerial_lab.tasks.direct.dragon.motion_loader import DragonMotion, DragonMotionLibrary  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG  # isort: skip
from aerial_lab.actuators.rotorgroup import RotorGroup  # isort: skip
from aerial_lab.utility.noisemodel import NoiseModel  # isort: skip

from aerial_lab.tasks.direct.spidar.mdp.terminations import (
    base_orientation_out_of_range,
    contact_force_exceeded,
    base_height_out_of_range,
)

# Direct RL env work flow:
# reset(): reset the environment, return initial observation
#   ├── reset_idx():
#   ├── scene.write_data_to_sim():
#   ├── sim.forward()
#   ├── sim.render()
#   └── return get_observations()
# step(action): apply the action, step the simulation, return next observation, reward, done, info
#   ├── pre_physics_step(action): apply action delay, compute target thrust/torque, set actuator command
#   ├── _ in range(self.cfg.decimation):
#   │   ├── _sim_step_counter += 1
#   │   ├── _apply_action()
#   │   ├── scene.write_data_to_sim()
#   │   └── self.scene.update(physics_dt)
#   ├── episode_length_buf += 1
#   ├── common_step_counter += 1
#   ├── _get_dones() -> reset_terminated, reset_time_outs
#   ├── _get_rewards() -> reward_buf
#   ├── reset_idx(reset_env_ids) for envs that are done
#   │   ├── scene.write_data_to_sim()
#   │   ├── sim.forward()
#   │   └── sim.render()
#   ├── event_manager.apply(mode="interval", dt=self.step_dt)
#   ├── _get_observations() for reset envs
#   └── return obs_buf, reward_buf, reset_terminated, reset_time_outs, extras


class _DebugVisAdapter:
    """Small adapter so BaseEnvWindow checkboxes can toggle a single visualization group."""

    def __init__(self, getter, setter):
        self._getter = getter
        self._setter = setter
        self.has_debug_vis_implementation = True
        self.cfg = self
        try:
            self.debug_vis = bool(self._getter())
        except AttributeError:
            self.debug_vis = False

    def set_debug_vis(self, debug_vis: bool):
        self.debug_vis = bool(debug_vis)
        self._setter(self.debug_vis)


class PoseTrackingEnvWindow(BaseEnvWindow):
    """Window manager for the Spidar environment."""

    def __init__(self, env: DragonTrackEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # keep strong references: BaseEnvWindow only stores a weakref.proxy to the checkbox
        # target, so inline adapters would be garbage-collected and raise a ReferenceError
        # ("weakly-referenced object no longer exists") when the checkbox is toggled.
        self._debug_vis_adapters = {
            "targets": _DebugVisAdapter(lambda: self.env._vis_targets, self.env.set_targets_vis),
            "thrusts": _DebugVisAdapter(lambda: self.env._vis_thrusts, self.env.set_thrusts_vis),
            "track_ends": _DebugVisAdapter(lambda: self.env._vis_track_ends, self.env.set_track_ends_vis),
            "actual_ends": _DebugVisAdapter(lambda: self.env._vis_actual_ends, self.env.set_actual_ends_vis),
        }
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization -- each checkbox toggles a single group
                    self._create_debug_vis_ui_element("targets", self._debug_vis_adapters["targets"])
                    self._create_debug_vis_ui_element("thrusts", self._debug_vis_adapters["thrusts"])
                    self._create_debug_vis_ui_element("track_ends", self._debug_vis_adapters["track_ends"])
                    self._create_debug_vis_ui_element("actual_ends", self._debug_vis_adapters["actual_ends"])


PUSH_LIN_VEL = 0.15
PUSH_ANG_VEL = 0.3

# Debug-vis knob: stride N for sparse-sampling the `track_horizon` future reference frames when
# drawing the target end-effector (front_end / back_end) markers. N=1 -> draw every future frame;
# N=2 -> every other frame; etc. Frame 0 (the current desired frame) is always included. Purely a
# visualization parameter -- has no effect on training.
TRACK_ENDS_FRAME_STRIDE = 4


# TODO Lists:
# [x] 0. Random joint reset with analytical gimbal
# [x] 1. Ground Effects
# [x] 2. Aerodynamics Interference
# 3. ~DR on rotor Pose~
# 4. ~Try history observation~
# [x] 5. External disturbance on links

@configclass
class EventCfg:
    """Configuration for randomization."""

    # Always got error: TypeError: randomize_rigid_body_mass.__init__() got an unexpected keyword argument 'asset_cfg'
    scale_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="link.*"),
            "mass_distribution_params": (0.90, 1.10),
            "operation": "scale",
        },
    )

    noise_base_com_pos = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="link.*"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )



@configclass
class DragonTrackEnvCfg(DirectRLEnvCfg):
    # # #### env parameters ########################################
    sim_dt = 1 / 200.0
    decimation = 5
    num_steps_per_env = 24
    episode_length_s = 15.0

    # # #### Training parameters ####################################
    play_mode = False
    evaluate_mode = False
    resume_mode = False
    resume_step_count = 4000 * num_steps_per_env
    add_noise = True
    add_delay = True
    add_randomization = True
    use_state_estimator = True  # EKF delay and scale
    decrease_rotor_thrust = True  # scale real rotor output by U(rotor_thrust_scale), sampled per pair on reset
    rotor_thrust_scale = [0.9, 0.95]
    training_view = True
    if play_mode or evaluate_mode:
        training_view = True
    replace_dof_vel_with_last_pos = True

    # # #### Robot parameters ########################################
    robot_mass = 7.0  # kg, w/o battery
    bodyName = "fc"
    rotor_num = 8
    paired_rotor_num = rotor_num // 2
    gimbal_num = 8
    joint_num = 6
    # policy controls 6 joints, 8 gimbals, and 4 paired rotor commands
    action_space = joint_num + gimbal_num + paired_rotor_num

    # # #### Reference-motion tracking ##############################
    # Primary target trajectory clip: drives Stage 1 tracking.
    # Only needs body pose/twist + the joint_num link-joint positions (no gimbal / rotor thrust);
    # see motion_loader.DragonMotion for the format.
    demo_file = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/dragon/trajectory/amp-dragon-move-target-192s.npz"
    # Evaluation uses one explicitly configured clip. Its timeline holds the first frame for
    # evaluation_hold_time_s, plays the clip forward once, then holds the last frame for the same
    # duration. Change this path directly when evaluating another trajectory.
    evaluation_demo_file = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/dragon/trajectory/evl-amp-Yaw-L.npz"
    evaluation_hold_time_s = 10.0
    # Stage 2 target expansion. Before the switch step every env uses `demo_file` exactly as in
    # Stage 1. Afterwards, each resetting env samples one trajectory for its whole episode with
    # probability num_frames_i / total_num_frames. This list is the complete Stage 2 target pool;
    # `demo_file` is listed explicitly rather than being injected by the environment.
    multi_trajectory_switch_stage = True
    multi_trajectory_switch_step_count = 20000 * num_steps_per_env
    multi_trajectory_demo_files = [
        f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/dragon/trajectory/amp-dragon-move-target-192s.npz",
        f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/dragon/trajectory/dragon-demo.npz",
        f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/dragon/trajectory/amp-Instability.npz",
        # f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/dragon/trajectory/amp-Pitch-L.npz",
        f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/dragon/trajectory/amp-Yaw-L.npz",
    ]
    # AMP expert clip (single .npz): the dynamically-feasible "real" state trajectory (optimal-control
    # solution of `demo_file`), storing body pose/twist + all 14 joints (6 link + 8 gimbal) + rotor
    # thrust. Only the AMP env uses it (as the discriminator's expert motion), decoupled from the
    # tracking target -- AMP is unpaired, so the two clips need not be frame-aligned or equal length.
    amp_demo_file = f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/dragon/trajectory/amp-dragon-move-real-filter1Hz-Average1s.npz"
    # number of future reference frames fed to the policy (the K-step look-ahead horizon)
    track_horizon = 5
    # spacing between consecutive look-ahead frames, expressed in control steps
    track_frame_stride = 1
    # RSI base/joint source: hybrid_initialization True -> nearest dynamically-feasible frame of the
    # real clip (_real_motion, searched around the target phase); False -> the target clip (_motion)
    # at the sampled phase.
    hybrid_initialization = True
    hybrid_init_look_offset = 5  # (tracking delay ~= 30 steps) global shift of the real search center
    hybrid_init_look_window = 5   # +/- frames searched around the (shifted) mapped phase
    hybrid_init_state_weight = [1.0, 1.0, 1.0]  # search-distance weights: base_pos, base_quat, joint_pos
    # RSI gimbal source: gimbal_init_from_demo True -> take gimbals from the demo source, False ->
    # analytical (rotors pointing up). The demo source is _real_motion when hybrid_initialization is on,
    # else _motion. Fail-loud: if True and the source clip has no gimbal columns, the DragonMotion load
    # raises (so a config/behaviour mismatch is never silently masked) -- there is no hover/analytical
    # fallback in that case.
    gimbal_init_from_demo = True
    # RSI thrust source: thrust_init_from_demo True -> take rotor thrust from the demo source (same
    # source rule as gimbal above), False -> hover thrust. Same fail-loud behaviour when the source
    # clip lacks rotor_thrust.
    thrust_init_from_demo = True
    # Past current-observation frames ordered [t-1, ..., t-H]. At reset, all slots use the
    # clean reset frame constructed in _reset_idx.
    history_mode = False
    obs_history_len = 3

    # # #### Observation / state layout ############################
    # target obs : per future frame = relative position (3) + relative rotation 6D (6)
    #              + target joint position (joint_num); repeated `track_horizon` times.
    per_target_frame_len = 3 + 6 + joint_num
    target_obs_len = track_horizon * per_target_frame_len
    # current obs : proprioception at the current step.
    #   joint pos (joint_num) + joint vel (joint_num) + gimbal pos (gimbal_num)
    #   + base rotation 6D (6) + linear velocity (3) + angular velocity (3) + last action (action_space)
    current_obs_len = joint_num + joint_num + gimbal_num + gimbal_num + 6 + 3 + 3 + action_space
    # history obs : `obs_history_len` past `current_obs` frames, [t-1, ..., t-H].
    single_obs_len = current_obs_len  # one history unit
    history_obs_len = obs_history_len * single_obs_len if history_mode else 0
    # policy observation = target | current | history; current already contains the last action.
    observation_space = target_obs_len + current_obs_len + history_obs_len
    # critic state = current | target | last action + privileged rotor thrust/torque (rotor_num * 2)
    state_space = current_obs_len + target_obs_len + rotor_num * 2
    amp_obs_history_len = 5
    # AMP observation = body lin vel (3) + body ang vel (3) + body rot6d (6)
    #                   + joint pos rel (joint_num) + gimbal pos rel (gimbal_num)
    #                   + paired rotor thrust (paired_rotor_num)
    # thrust uses the paired representation (one value per rotor pair): the policy commands and the
    # demo split each apply an identical value to a pair's two rotors, so the per-rotor form would be
    # exactly duplicated -- the paired form is lossless here and matches the 4-dim action space.
    single_amp_obs_len = 3 + 3 + 6 + joint_num + gimbal_num
    amp_observation_space = amp_obs_history_len * single_amp_obs_len

    # # #### Model parameters
    # # Actuator parameters ########################################
    thrust_to_torque_ratio = -0.0165  # -0.0165
    thrust_limit_max = 14.5  # N
    rotor_direction_cfg = {"edf.*_left": 1, "edf.*_right": -1}
    contact_force_threshold = 100.0  # N wouldn't trigger terminaton in lide-down state, but can detect hard contact
    contact_velocity_threshold = 1.0  # m/s, can detect hard contact with low false positive in normal flight
    # DeepMimic-style reference-tracking early termination: end the episode when the joint
    # configuration deviates from the demo target by more than this (L2 norm over the 6 link
    # joints, rad). RSI starts each episode on the reference, so this only fires once the policy
    # lets the joints diverge -- removing the "hover and ignore the joints" local optimum.
    # Tuning knob: tighten (~1.5) to force closer joint tracking; loosen to give early training
    # slack. Reference: running joint-err norm ~1.0, final ~2.0 in the stuck run. Large (1e3) disables.
    joint_track_termination_threshold = 1.0
    position_track_termination_threshold = 0.5  # m, L2 norm over xyz position error
    orientation_track_termination_threshold = math.pi * 0.2  # rad, L2 norm over 6D rotation error
    # Stable-training diagnostics: once the mean episode length for the current done batch is above
    # this threshold, write one CSV row for each early-terminated env from _get_dones().
    early_termination_csv_enabled = False
    early_termination_csv_filename = "early_termination_causes.csv"
    early_termination_csv_min_mean_episode_length = 400.0

    rotorCfg = {
        "rotor_num": gimbal_num,
        "dt": sim_dt,
        "mode": "foc",
        "thrust_coeff": 1.0,
        "torque_coeff": thrust_to_torque_ratio,
        "randomize_ratio": 0.0,  # 3% thrust/torque randomization, scale
        "thrust_scale": rotor_thrust_scale if decrease_rotor_thrust else [1.0, 1.0],
        "max_vel": 1000.0,
        "max_foc": thrust_limit_max,
        "vel_wn": 1.0,
        "vel_zeta": 0.8,
        "foc_wn": 1.0,
        "foc_zeta": 0.8,
        "foc_tau": 0.0942,
    }

    ui_window_class_type = PoseTrackingEnvWindow

    class randomization:
        body_ang = math.pi  # body tilt angle for init orientation sampling
        z_ubound = 4.0
        z_lbound = 2.0
        lin_vel = 1.0
        ang_vel = 1.5
        # RSI: deviation of the initial state from the sampled demo frame ----
        rsi_pos_dev = 0.05      # [m] uniform xyz position deviation, scaled by _pos_sample_stage
        rsi_ang_dev = math.pi * 0.02  # [rad] orientation deviation, scaled by _quat_sample_stage
        rsi_lin_vel_dev = 0.2   # [m/s] base linear-velocity deviation, scaled by _vel_random_stage
        rsi_ang_vel_dev = 0.3   # [rad/s] base angular-velocity deviation, scaled by _vel_random_stage
        rsi_joint_dev = math.pi * 0.05  # [rad] joint-position deviation, scaled by _joint_random_stage
        rsi_gimbal_dev = math.pi * 0.15  # [rad] gimbal-position deviation, scaled by _gimbal_random_stage
        # RSI: only start an episode from demo frames whose reference height (base_pos_w z) is
        # at least this value [m]. Filters out the takeoff/landing (near-ground) frames so the
        # robot never spawns with its long links/end-effectors penetrating the ground (which would
        # trigger an immediate contact-crash on reset). Set to None to sample the whole clip.
        rsi_min_height = 0.5

    class normalization:
        class obs_scales:
            ang_vel = 1.0
            lin_vel = 1.0
            dof_vel = 0.5
            dof_delta = 10.0
            torque = 5.0
            thrust = 0.1

    class control:
        body_ang = math.pi * 0.0   # body tilt angle for desired orientation sampling
        max_tilt_angle = torch.cos(torch.tensor(math.pi * 1.5))  # max tilt angle for termination condition
        xy_range = 2.0
        z_ubound = 4.0
        z_lbound = 2.0
        clip_observations = 100.0
        clip_actions = 100.0
        # class control:
        # action scale = 0.25 * effort_limit / stiffness (BeyondMimic impedance heuristic):
        # one unit of policy output commands ~25% of each actuator's max torque -> uniform
        # per-actuator authority. MUST be kept consistent with the kp set in DRAGON_CFG; changing
        # the actuator stiffness without updating these scales causes torque saturation.
        joint_action_scale = 0.15         # XM430-W350: 0.25*4.1/51.5 => 2:1 gear ratio
        gimbal_pitch_action_scale = 0.25   # XC330-T181: 0.25*0.76/0.593
        gimbal_roll_action_scale = 0.125   # XM430-W210: 0.25*3.0/19.6
        thrust_action_scale = 0.5  # - 7 ~ 7  => 14

        thrust_limit = 14.5  # N
        thrust_hover = 8.5  # N

        # Joint  00 - ID: 00, Name: joint1_yaw, Default: 1.5600 rad, Limit: [-1.5708, 1.5708] rad
        # Joint  01 - ID: 02, Name: joint2_pitch, Default: 0.0000 rad, Limit: [-1.5708, 1.5708] rad
        # Joint  02 - ID: 03, Name: joint1_pitch, Default: 0.0000 rad, Limit: [-1.5708, 1.5708] rad
        # Joint  03 - ID: 05, Name: joint2_yaw, Default: 1.5600 rad, Limit: [-1.5708, 1.5708] rad
        # Joint  04 - ID: 08, Name: joint3_pitch, Default: 0.0000 rad, Limit: [-1.5708, 1.5708] rad
        # Joint  05 - ID: 11, Name: joint3_yaw, Default: 1.5600 rad, Limit: [-1.5708, 1.5708] rad
        # Gimbal 00 - ID: 01, Name: gimbal2_roll, Default: 0.0000 rad, Limit: [-3.1416, 3.1416] rad
        # Gimbal 01 - ID: 04, Name: gimbal2_pitch, Default: 0.0000 rad, Limit: [-3.1416, 3.1416] rad
        # Gimbal 02 - ID: 06, Name: gimbal1_roll, Default: 0.0000 rad, Limit: [-3.1416, 3.1416] rad
        # Gimbal 03 - ID: 07, Name: gimbal3_roll, Default: 0.0000 rad, Limit: [-3.1416, 3.1416] rad
        # Gimbal 04 - ID: 09, Name: gimbal1_pitch, Default: 0.0000 rad, Limit: [-3.1416, 3.1416] rad
        # Gimbal 05 - ID: 10, Name: gimbal3_pitch, Default: 0.0000 rad, Limit: [-3.1416, 3.1416] rad
        # Gimbal 06 - ID: 12, Name: gimbal4_roll, Default: 0.0000 rad, Limit: [-3.1416, 3.1416] rad
        # Gimbal 07 - ID: 13, Name: gimbal4_pitch, Default: 0.0000 rad, Limit: [-3.1416, 3.1416] rad
        # Thrust 00 - ID: 15, Name: edf2_left, Direction: Left
        # Thrust 01 - ID: 16, Name: edf2_right, Direction: Right
        # Thrust 02 - ID: 24, Name: edf1_left, Direction: Left
        # Thrust 03 - ID: 25, Name: edf1_right, Direction: Right
        # Thrust 04 - ID: 27, Name: edf3_left, Direction: Left
        # Thrust 05 - ID: 28, Name: edf3_right, Direction: Right
        # Thrust 06 - ID: 34, Name: edf4_left, Direction: Left
        # Thrust 07 - ID: 35, Name: edf4_right, Direction: Right
        gimbal_pitch_ids = [1, 4, 5, 7]
        gimbal_roll_ids = [0, 2, 3, 6]
        pos_hovering = [0.0, 0.0, 2.0]
        dof_pos_default = {
            "joint1_yaw": 0.0,
            "joint2_yaw": 0.0,
            "joint3_yaw": 0.0,
            "joint1_pitch": 0.0,
            "joint2_pitch": 0.0,
            "joint3_pitch": 0.0,
            "gimbal1_roll": 0.0,
            "gimbal2_roll": 0.0,
            "gimbal3_roll": 0.0,
            "gimbal4_roll": 0.0,
            "gimbal1_pitch": 0.0,
            "gimbal2_pitch": 0.0,
            "gimbal3_pitch": 0.0,
            "gimbal4_pitch": 0.0,
        }
        dof_pos_limit = {
            "gimbal_lower": -math.pi,
            "gimbal_upper": math.pi,
            "joint_lower": -math.pi * 0.5,
            "joint_upper": math.pi * 0.5,
        }
    control.thrust_limit = thrust_limit_max

    # Staged/Curriculum parameters (in number of steps) #############################
    # (history_mode / obs_history_len are defined above with the observation layout)
    curriculum_mode = False

    vel_random_mode = True  # add randomness to initial (RSI) base velocity during reset
    # vel randomization parameters ===========================================
    vel_random_step_start = 10 * num_steps_per_env
    vel_random_step_end = 3000 * num_steps_per_env

    gimbal_random_mode = True  # add randomness to gimbal positions during reset
    # dof randomization parameters ===========================================
    gimbal_random_step_start = 10 * num_steps_per_env
    gimbal_random_step_end = 3000 * num_steps_per_env

    joint_random_mode = True  # add randomness to initial (RSI) joint positions during reset
    # joint randomization parameters =========================================
    joint_random_step_start = 10 * num_steps_per_env
    joint_random_step_end = 3000 * num_steps_per_env

    pos_mode = True  # add randomness to initial (RSI) base position during reset
    # position sampling parameters ===========================================
    pos_sample_step_start = 10 * num_steps_per_env
    pos_sample_step_end = 3000 * num_steps_per_env

    quat_mode = True  # add randomness to initial (RSI) base orientation during reset
    # orientation sampling parameters ========================================
    quat_sample_step_start = 10 * num_steps_per_env
    quat_sample_step_end = 3000 * num_steps_per_env

    push_mode = False  # apply random push disturbance during episode (per-link external impulse force)
    # push disturbance parameters ============================================
    push_episode_min = 1.0 / (sim_dt * decimation)
    push_episode_max = 14.0 / (sim_dt * decimation)
    push_lin_vel_range = 0.5  # desired body-frame delta linear velocity magnitude per link [m/s]
    push_link_prob = 0.5  # per-link probability of being pushed when a disturbance is triggered
    push_impulse_duration_min_steps = 2  # physics steps the impulse force lasts
    push_impulse_duration_max_steps = 6
    push_step_start = 1000 * num_steps_per_env
    push_step_end = 3000 * num_steps_per_env  # after this step, always applied during episode

    # static-hover task mixing #################################
    # A fraction of envs get their reference frozen at the RSI initial demo frame (a zero-velocity
    # setpoint) instead of the moving trajectory, so they learn to *hold* a thrust-vectored joint
    # configuration -- the prerequisite skill the full-tracking envs cannot bootstrap on their own
    # (they drift into the saturated/dead-gradient region of the joint reward). The fraction decays
    # linearly from rate_start to rate_end over [step_start, step_end]. static_hover_mode=False
    # disables it (the env then behaves exactly as before).
    static_hover_mode = True
    static_hover_step_start = 0
    static_hover_step_end = 5000 * num_steps_per_env
    static_hover_rate_start = 0.5   # fraction of static-hover envs early in training
    static_hover_rate_end = 0.0    # fraction after step_end

    # # reward scales #############################################
    lin_vel_reward_scale = -0.0       # Scale = 100
    ang_vel_reward_scale = -0.0       # -0.01       # Scale = 100
    lin_acc_reward_scale = -0.0      # -0.001
    ang_acc_reward_scale = -0.0      # -0.001
    thrust_power_reward_scale = -5.0e-4  # -1.0e-4  # Scale = ave(10)^2*4 = 400
    contact_reward_scale = -0.01  # Scale = 0.5
    # joint mechanical-stop force (j*_center_plate); grows with over-commanded preload, so it
    # penalizes pressing the stop too hard (joint_limit only sees the saturated position)
    joint_contact_reward_scale = -1.0e-3
    died_reward_scale = -0.0  # -1.5

    # # # Tracking rewards (DeepMimic / whole_body_tracking style) ----------------
    # Each tracking term is exp(-err^2 / std^2) in (0, 1], with a positive weight; the
    # terms are *added* (not multiplied). "Track everything, don't just hover" is enforced
    # by the joint-drift early termination (joint_track_termination_threshold), not by
    # multiplicative coupling. All-positive tracking reward also makes a separate died/alive
    # term unnecessary: staying alive keeps accumulating positive reward, so early termination
    # is implicitly penalized (losing future reward) -- hence no died_reward / alive_reward.
    position_tracking_reward_scale = 0.8     # base (imu) position tracking
    orientation_tracking_reward_scale = 0.8  # base (imu) orientation tracking
    joint_tracking_reward_scale = 1.5        # 6-link joint-position tracking (single L2 term)

    position_deviation_reward_scale = -0.2      # base position deviation from demo (single L2 term)
    orientation_deviation_reward_scale = -0.2   # base orientation deviation from demo (
    joint_deviation_reward_scale = -0.3      # 6-link joint-position deviation from demo (single L2 term)
    # std controls the tolerance band of each exp term (smaller -> sharper, demands tighter tracking)
    position_tracking_std = 0.3              # [m]
    orientation_tracking_std = math.pi / 5.0 # [rad]
    joint_tracking_std = math.pi / 6.0       # [rad], L2 norm over the 6 joints

    body_lin_vel_tracking_reward_scale = -0.01  # squared L2 (-0.01) -, exp form +;
    body_lin_vel_tracking_std = 0.15  # average lin vel in demo ~0.05 m/s
    body_ang_vel_tracking_reward_scale = -0.01  # squared L2 (-0.01) -, exp form +;
    body_ang_vel_tracking_std = 0.25  # average ang vel in demo ~0.1 rad/s

    # smoothing reward scales
    # Action-rate penalties are computed after applying action scales, so the squared deltas
    # are in physical command units: joint/gimbal rad and paired-thrust N.
    joint_action_rate_reward_scale = -0.5
    gimbal_action_rate_reward_scale = -0.8
    thrust_action_rate_reward_scale = -0.05

    joint_vel_reward_scale = 0.0  # -0.01
    joint_acc_reward_scale = -5.0e-4  # -1.5e-7  5e-7
    joint_limit_reward_scale = -1.0  # -0.01; limits = URDF limits * robot soft_joint_pos_limit_factor

    gimbal_vel_reward_scale = 0.0  # -1.0e-5
    gimbal_acc_reward_scale = -5.0e-4  # -1.5e-7  5e-7
    gimbal_limit_reward_scale = -0.01  # -0.01
    gimbal_limit_range = math.pi * 0.9
    gimbal_smooth_reward_scale = -0.0  # -1.0e-4
    gimbal_vibration_reward_scale = 0.0  # -1.0e-3  => reduce gimbal vibration -1.0e-6

    thrust_limit_reward_scale = -0.05  # -0.01
    thrust_limit = control.thrust_limit * 0.9

    # # Joint / gimbal torque rewards (torque read from the implicit-actuator PD estimate) #######
    # joint and gimbal are penalized with SEPARATE scales because their stiffness (kp) differs by
    # ~2 orders of magnitude -> raw torque magnitudes are not comparable across the two groups.
    # torque_power: mechanical power sum|applied_torque * joint_vel| (energy/effort minimization).
    # torque_limit: sum of soft-limit violation of the *computed* (pre-clip) torque, so the penalty
    #               still has gradient once the actuator saturates (cf. humanoid_hsi _reward_torque_limits).
    joint_torque_power_reward_scale = -2.5e-5
    joint_torque_limit_reward_scale = -0.05
    gimbal_torque_power_reward_scale = 0.0
    gimbal_torque_limit_reward_scale = 0.0
    soft_joint_torque_limit = 0.4  # 6.2 * 0.4 = 2.48 Nm, prevent pulley tooth skipping
    soft_gimbal_torque_limit = 0.9  # fraction of effort_limit before the torque-limit penalty kicks in

    # # Delay model parameters ##################################
    obsPosDelay = DelayModelCfg()
    obsPosDelay.name = "pos"
    obsPosDelay.dim = 3
    obsPosDelay.default = [0.0, 0.0, 0.0]
    obsPosDelay.min_delay_steps = 0
    obsPosDelay.max_delay_steps = 0
    obsPosDelay.jitter_delay_steps = 0
    obsQuatDelay = DelayModelCfg()
    obsQuatDelay.name = "quat"
    obsQuatDelay.dim = 4
    obsQuatDelay.default = [1.0, 0.0, 0.0, 0.0]
    obsQuatDelay.min_delay_steps = 0
    obsQuatDelay.max_delay_steps = 0
    obsQuatDelay.jitter_delay_steps = 0
    obsLinVelDelay = DelayModelCfg()
    obsLinVelDelay.name = "linvel"
    obsLinVelDelay.dim = 3
    obsLinVelDelay.default = [0.0, 0.0, 0.0]
    obsLinVelDelay.min_delay_steps = 0
    obsLinVelDelay.max_delay_steps = 1
    obsLinVelDelay.jitter_delay_steps = 1
    obsAngVelDelay = DelayModelCfg()
    obsAngVelDelay.name = "angvel"
    obsAngVelDelay.dim = 3
    obsAngVelDelay.default = [0.0, 0.0, 0.0]
    obsAngVelDelay.min_delay_steps = 0
    obsAngVelDelay.max_delay_steps = 1
    obsAngVelDelay.jitter_delay_steps = 1
    obsJointPosDelay = DelayModelCfg()
    obsJointPosDelay.name = "joint_pos"
    obsJointPosDelay.dim = 6
    obsJointPosDelay.default = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    obsJointPosDelay.min_delay_steps = 0
    obsJointPosDelay.max_delay_steps = 1
    obsJointPosDelay.jitter_delay_steps = 1
    obsJointVelDelay = DelayModelCfg()
    obsJointVelDelay.name = "joint_vel"
    obsJointVelDelay.dim = 6
    obsJointVelDelay.default = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    obsJointVelDelay.min_delay_steps = 0
    obsJointVelDelay.max_delay_steps = 1
    obsJointVelDelay.jitter_delay_steps = 1
    obsGimbalPosDelay = DelayModelCfg()
    obsGimbalPosDelay.name = "gimbal_pos"
    obsGimbalPosDelay.dim = 8
    obsGimbalPosDelay.default = [0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0]
    obsGimbalPosDelay.min_delay_steps = 0
    obsGimbalPosDelay.max_delay_steps = 1
    obsGimbalPosDelay.jitter_delay_steps = 1
    obsGimbalVelDelay = DelayModelCfg()
    obsGimbalVelDelay.name = "gimbal_vel"
    obsGimbalVelDelay.dim = 8
    obsGimbalVelDelay.default = [0.0, 0.0, 0.0, 0.0,
                                 0.0, 0.0, 0.0, 0.0]
    obsGimbalVelDelay.min_delay_steps = 0
    obsGimbalVelDelay.max_delay_steps = 1
    obsGimbalVelDelay.jitter_delay_steps = 1
    actJointDelay = DelayModelCfg()
    actJointDelay.name = "joint"
    actJointDelay.dim = 6
    actJointDelay.default = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # joint
    actJointDelay.min_delay_steps = 0 * 5
    actJointDelay.max_delay_steps = 1 * 5
    actJointDelay.jitter_delay_steps = 1
    actGimbalDelay = DelayModelCfg()
    actGimbalDelay.name = "gimbal"
    actGimbalDelay.dim = 8
    actGimbalDelay.default = [0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0]  # gimbal
    actGimbalDelay.min_delay_steps = 0 * 5
    actGimbalDelay.max_delay_steps = 1 * 5
    actGimbalDelay.jitter_delay_steps = 1
    actThrustDelay = DelayModelCfg()
    actThrustDelay.name = "thrust"
    actThrustDelay.dim = 8
    actThrustDelay.default = [control.thrust_hover] * rotor_num  # hover thrust
    actThrustDelay.min_delay_steps = 0 * 5
    actThrustDelay.max_delay_steps = 2 * 5
    actThrustDelay.jitter_delay_steps = 2

    # # State estimator model ####################################
    # actor lin_vel / ang_vel as the aerial_robot estimator (EXPERIMENT_ESTIMATE) outputs them;
    # obsLinVelDelay / obsAngVelDelay are applied after it. Critic keeps the IMU truth.

    stateEstimator = DragonStateEstimatorModelCfg()
    stateEstimator.dt = sim_dt
    stateEstimator.gravity = 9.81
    stateEstimator.yaw_iir_cutoff_hz = 10.0
    stateEstimator.yaw_iir_jsk_formula = True
    stateEstimator.estimate_lin_vel = True
    stateEstimator.sigma_acc = 0.05
    stateEstimator.sigma_acc_bias = 0.001
    stateEstimator.sigma_pos = 0.0002
    stateEstimator.mocap_period_steps = 2  # 100 Hz
    stateEstimator.accel_delay_steps_range = (8, 16)  # 40~80 ms
    stateEstimator.accel_noise_std_range = (0.0, 0.5)  # m/s^2
    stateEstimator.mocap_delay_steps_range = (2, 4)
    stateEstimator.mocap_noise_std_range = (0.0, 0.0005)  # m

    # # # # Noise Configuration
    noiseCfg = {
        "root_pos": {
            "type": "uniform",
            "dim": 3,
            "scale_lb": 1.0,
            "scale_ub": 1.0,
            "mean": 0.005,
            "std": 0.01,
            "clip": 0.3,
        },
        "root_quat": {
            "type": "uniform",
            "dim": 3,
            "scale_lb": 1.0,
            "scale_ub": 1.0,
            "mean": math.pi * (1.0 / 180.0),
            "std": math.pi * (2.0 / 180.0),
            "clip": 0.3,
        },
        "lin_vel": {
            "type": "uniform",
            "dim": 3,
            "scale_lb": 0.90,
            "scale_ub": 1.10,
            "mean": 0.01,
            "std": 0.05,
            "clip": 3.0,
        },
        "ang_vel": {
            "type": "uniform",
            "dim": 3,
            "scale_lb": 0.90,
            "scale_ub": 1.10,
            "mean": 0.01,
            "std": 0.05,
            "clip": 3.0,
        },
        "joint_pos": {
            "type": "uniform",
            "dim": joint_num,
            "scale_lb": 1.0,
            "scale_ub": 1.0,
            "mean": math.pi * (3.0 / 180.0),
            "std": math.pi * (2.0 / 180.0),
            "clip": 0.1,
        },
        "joint_vel": {
            "type": "uniform",
            "dim": joint_num,
            "scale_lb": 1.0,
            "scale_ub": 1.0,
            "mean": math.pi * (3.0 / 180.0),
            "std": math.pi * (2.0 / 180.0),
            "clip": 0.1,
        },
        "gimbal_pos": {
            "type": "uniform",
            "dim": gimbal_num,
            "scale_lb": 1.0,
            "scale_ub": 1.0,
            "mean": math.pi * (3.0 / 180.0),
            "std": math.pi * (2.0 / 180.0),
            "clip": 0.1,
        },
        "gimbal_vel": {
            "type": "uniform",
            "dim": gimbal_num,
            "scale_lb": 1.0,
            "scale_ub": 1.0,
            "mean": math.pi * (3.0 / 180.0),
            "std": math.pi * (2.0 / 180.0),
            "clip": 0.1,
        },
        "joint_pos_delta": {
            "type": "uniform",
            "dim": joint_num,
            "scale_lb": 0.995,
            "scale_ub": 1.005,
            "mean": 0.0,
            "std": math.pi * (0.2 / 180.0),  # sim_dt
            "clip": 5.0,
        },
        "gimbal_pos_delta": {
            "type": "uniform",
            "dim": gimbal_num,
            "scale_lb": 0.995,
            "scale_ub": 1.005,
            "mean": 0.0,
            "std": math.pi * (0.2 / 180.0),  # sim_dt
            "clip": 5.0,
        },
    }

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     collision_group=-1,
    #     terrain_generator=TerrainGeneratorCfg(
    #         seed=0,
    #         size=(10.0, 10.0),
    #         border_width=2.0,
    #         num_rows=8,
    #         num_cols=8,
    #         horizontal_scale=0.1,
    #         vertical_scale=0.01,
    #         slope_threshold=0.75,
    #         curriculum=False,
    #         sub_terrains={
    #             "flat_rough": MeshRandomGridTerrainCfg(
    #                 proportion=1.0,
    #                 grid_width=0.15,
    #                 grid_height_range=(0.0, 0.02),
    #                 platform_width=1.0,
    #             ),
    #         },
    #     ),
    #     max_init_terrain_level=None,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=False,
    # )
    # # # # Too many warnings from the terrain generator
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     collision_group=-1,
    #     terrain_generator=TerrainGeneratorCfg(
    #         seed=0,
    #         size=(10.0, 10.0),
    #         border_width=0.0,
    #         num_rows=8,
    #         num_cols=8,
    #         horizontal_scale=0.1,
    #         vertical_scale=0.01,
    #         slope_threshold=0.75,
    #         curriculum=False,
    #         sub_terrains={
    #             "flat_rough": HfRandomUniformTerrainCfg(
    #                 proportion=1.0,
    #                 noise_range=(-0.02, 0.02),  # height noise range in meters
    #                 noise_step=0.02,
    #                 border_width=0.0,
    #             ),
    #         },
    #     ),
    #     max_init_terrain_level=None,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=False,
    # )

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            enable_external_forces_every_iteration=True,
            solver_type=1,            # TGS solver is more stable for stacking and complex contact, 
            # but slower than PGS. PGS is faster but can be unstable with high mass ratio or complex contact.
            max_velocity_iteration_count=1,  # reduce velocity iterations for faster simulation, 
            # can be increased for more accuracy and stability at the cost of speed
        ),
    )

    # robot_cfg: ArticulationCfg = MINI_QUADROTOR_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg: ArticulationCfg = DRAGON_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=4.0, replicate_physics=True)

    body_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/link.*",  # Bind to the robot root link
        # prim_path="/World/envs/env_.*/Robot/base_link",  # Bind to the robot root link
        history_length=decimation,  # cover every physics substep of one env step
        update_period=0,  # Update every physics step
        track_air_time=True,
        # track_contact_points=True,
        debug_vis=training_view,
        # filter_prim_paths_expr=["/World/ground"],  # Only track contacts with the ground
        # filter_prim_paths_expr=[terrain.prim_path],  # Only track contacts with the ground
    )

    thrust_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/edf.*",  # Bind to all rotor parent links
        history_length=decimation,  # cover every physics substep of one env step
        update_period=0,
        track_air_time=True,
        track_pose=True,
        debug_vis=training_view,
    )

    foot_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/inter_joint.*",  # Bind to the robot foot link
        history_length=1,
        update_period=0,  # Update every physics step
        track_air_time=True,
        track_pose=True,
        debug_vis=training_view,
        # filter_prim_paths_expr=["/World/ground"],  # Only track contacts with the ground
        # filter_prim_paths_expr=[terrain.prim_path],  # Only track contacts with the ground
    )

    # The center plate sits between the pitch/yaw axes and only touches the j*_pitch_frame /
    # j*_yaw_frame colliders at the joint mechanical stop, so its net force is the stop force.
    joint_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/j.*_center_plate",
        history_length=decimation,  # cover every physics substep of one env step
        update_period=0,  # Update every physics step
        debug_vis=training_view,
    )

    end_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*_end",  # Bind to the robot foot link
        history_length=decimation,  # cover every physics substep of one env step
        update_period=0,  # Update every physics step
        track_air_time=True,
        track_pose=True,
        debug_vis=training_view,
        # filter_prim_paths_expr=["/World/ground"],  # Only track contacts with the ground
        # filter_prim_paths_expr=[terrain.prim_path],  # Only track contacts with the ground
    )

    # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#inertia-measurement-unit
    imu_link_name = "fc"
    imu_sensor: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/fc",
        update_period=0,
        history_length=1,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),  # w, x, y, z
        ),
        debug_vis=training_view,
    )

    if add_randomization:
        events: EventCfg = EventCfg()


class DragonTrackEnv(DirectRLEnv):
    cfg: DragonTrackEnvCfg

    def get_evaluation_spec(self):
        """Return the task-owned evaluation spec used by scripts/rsl_rl/evaluate.py."""
        from .dragonEvaluation import DragonEvaluation

        return DragonEvaluation(self)

    def __init__(self, cfg: DragonTrackEnvCfg, render_mode: str | None = None, **kwargs):
        play_mode = bool(getattr(cfg, "play_mode", False))
        evaluate_mode = bool(getattr(cfg, "evaluate_mode", False))
        track_whole = play_mode or evaluate_mode
        if evaluate_mode:
            cfg.demo_file = cfg.evaluation_demo_file
            cfg.multi_trajectory_switch_stage = False
        elif play_mode:
            cfg.episode_length_s = 200.0

        # PoseTrackingEnvWindow is created inside DirectRLEnv.__init__, so these flags must exist
        # before the parent init builds the debug-visualization checkboxes.
        training_vis = bool(getattr(cfg, "training_view", False))
        self._vis_targets = training_vis or track_whole
        self._vis_thrusts = training_vis
        self._vis_track_ends = track_whole
        self._end_fk_ready = False
        self._vis_actual_ends = track_whole
        self._actual_ends_ready = False

        super().__init__(cfg, render_mode, **kwargs)
        try:
            print(f"DragonTrackEnv init: cfg class = {cfg.__class__.__name__}")
            print(f"cfg.robot_cfg.asset_path = {getattr(cfg.robot_cfg.spawn, 'asset_path', None)}")
        except Exception as e:
            print("DragonTrackEnv init: failed to print cfg.robot_cfg:", e)
        print_training_config()
        # Basic cfgs ===================================================
        # # Noise models
        self.noiseModel = NoiseModel(cfg.noiseCfg, device=self.device, num_envs=self.num_envs)
        # # Delay models
        # self._obsPosDelayModel = DelayModel(cfg.obsPosDelay, device=self.device, num_envs=self.num_envs)
        # self._obsQuatDelayModel = DelayModel(cfg.obsQuatDelay, device=self.device, num_envs=self.num_envs)
        self._obsLinVelDelayModel = DelayModel(cfg.obsLinVelDelay, device=self.device, num_envs=self.num_envs)
        self._obsAngVelDelayModel = DelayModel(cfg.obsAngVelDelay, device=self.device, num_envs=self.num_envs)
        self._obsJointPosDelayModel = DelayModel(cfg.obsJointPosDelay, device=self.device, num_envs=self.num_envs)
        self._obsJointVelDelayModel = DelayModel(cfg.obsJointVelDelay, device=self.device, num_envs=self.num_envs)
        self._obsGimbalPosDelayModel = DelayModel(cfg.obsGimbalPosDelay, device=self.device, num_envs=self.num_envs)
        self._obsGimbalVelDelayModel = DelayModel(cfg.obsGimbalVelDelay, device=self.device, num_envs=self.num_envs)
        self._actJointDelayModel = DelayModel(cfg.actJointDelay, device=self.device, num_envs=self.num_envs)
        self._actGimbalDelayModel = DelayModel(cfg.actGimbalDelay, device=self.device, num_envs=self.num_envs)
        self._actThrustDelayModel = DelayModel(cfg.actThrustDelay, device=self.device, num_envs=self.num_envs)
        # # State estimator model
        self._stateEstimator = DragonStateEstimatorModel(cfg.stateEstimator, self.device, self.num_envs)
        self._est_lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._est_ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        # # Randomization events
        self.randomCfg = cfg.randomization
        # # Normalization
        self.obsScales = cfg.normalization.obs_scales
        # # Control Configuration
        self.ctrlCfg = self.cfg.control
        # # Training Configuration
        # print_training_config()

        # # Robot Physical Properties ================================================
        self._body_id = self._robot.find_bodies(self.cfg.bodyName)[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._imu_id = self._robot.find_bodies(self.cfg.imu_link_name)[0]

        _body_pos_w = self._robot.data.body_pos_w[0, self._body_id]  # shape: (3,)
        _body_quat_w = self._robot.data.body_quat_w[0, self._body_id]  # shape: (4,)
        _imu_pos_w = self._robot.data.body_pos_w[0, self._imu_id]  # shape: (3,)
        _imu_quat_w = self._robot.data.body_quat_w[0, self._imu_id]  # shape: (4,)
        _imu_pos_body, _imu_quat_body = subtract_frame_transforms(
            _body_pos_w.unsqueeze(0),  # shape: (1, 3)
            _body_quat_w.unsqueeze(0),  # shape: (1, 4)
            _imu_pos_w.unsqueeze(0),   # shape: (1, 3)
            _imu_quat_w.unsqueeze(0),  # shape: (1, 4)
        )

        Roll, Pitch, Yaw = euler_xyz_from_quat(_imu_quat_body[0])
        _imu_rpy_body = torch.stack([Roll[0], Pitch[0], Yaw[0]], dim=-1)  # shape: (3,)
        self._imu_pos_body = _imu_pos_body[0]  # shape: (3,)
        self._imu_quat_body = _imu_quat_body[0]  # shape: (4,)
        self._imu_rpy_body = _imu_rpy_body  # shape: (3,)

        self._thrust_ids = self._robot.find_bodies("edf.*")
        self._link_ids = self._robot.find_bodies("link.*")
        # These IDs are consumed on every physics substep. Cache their device tensors instead of
        # allocating new CUDA tensors in the action hot path.
        self._thrust_body_ids_tensor = torch.tensor(
            self._thrust_ids[0], dtype=torch.int32, device=self.device
        )
        self._link_body_ids_tensor = torch.tensor(
            self._link_ids[0], dtype=torch.int32, device=self.device
        )
        self._rotor_directions = [1 if "left" in name else -1 for name in self._thrust_ids[1]]
        self._rotor_action_indices = self._aars_build_paired_rotor_action_indices(self._thrust_ids[1])
        self._gimbal_ids = self._robot.find_joints("gimbal.*")
        self._joint_ids = self._robot.find_joints("joint.*")
        # self._rotor_ids = self._robot.find_joints("rotor.*")

        # Default positions: look up each URDF-ordered joint name in dof_pos_default, fallback 0.0
        self._joint_default_pos = torch.tensor(
            [self.ctrlCfg.dof_pos_default.get(name, 0.0) for name in self._joint_ids[1]],
            device=self.device, dtype=torch.float32,
        ).unsqueeze(0).expand(self.num_envs, -1).clone()  # [num_envs, n_joints]

        self._gimbal_default_pos = torch.tensor(
            [self.ctrlCfg.dof_pos_default.get(name, 0.0) for name in self._gimbal_ids[1]],
            device=self.device, dtype=torch.float32,
        ).unsqueeze(0).expand(self.num_envs, -1).clone()  # [num_envs, n_gimbals]

        self._joint_effort_limit = self._robot.data.joint_effort_limits[:, self._joint_ids[0]].clone()
        self._gimbal_effort_limit = self._robot.data.joint_effort_limits[:, self._gimbal_ids[0]].clone()

        # Training-side limits from cfg (independent of URDF physics limits).
        # Applied uniformly to all joints/gimbals of each type.
        _n_joints = len(self._joint_ids[0])
        _n_gimbals = len(self._gimbal_ids[0])
        self._joint_lower_pos = torch.full(
            (self.num_envs, _n_joints), self.ctrlCfg.dof_pos_limit["joint_lower"],
            device=self.device, dtype=torch.float32,
        )
        self._joint_upper_pos = torch.full(
            (self.num_envs, _n_joints), self.ctrlCfg.dof_pos_limit["joint_upper"],
            device=self.device, dtype=torch.float32,
        )
        self._gimbal_lower_pos = torch.full(
            (self.num_envs, _n_gimbals), self.ctrlCfg.dof_pos_limit["gimbal_lower"],
            device=self.device, dtype=torch.float32,
        )
        self._gimbal_upper_pos = torch.full(
            (self.num_envs, _n_gimbals), self.ctrlCfg.dof_pos_limit["gimbal_upper"],
            device=self.device, dtype=torch.float32,
        )
        print("=" * 60)
        # print("Beetle robot link list: ")
        # for link in self._robot.body_names:
        #     print(f" - {link}")
        # print("Beetle robot joint list: ")
        # for joint in self._robot.joint_names:
        #     print(f" - {joint}")
        print("Root Body ID: ", self._body_id, " Name: ", self.cfg.bodyName)
        print("Robot Mass (DR): ", self._robot_mass)
        print("Robot Mass (Ideal): ", self.cfg.robot_mass)
        print("IMU Offset Position (body frame): ", self._imu_pos_body.cpu().numpy())
        print("IMU Offset EulerAng (body frame): ", self._imu_rpy_body.cpu().numpy())
        for i in range(len(self._link_ids[0])):
            link_id = self._link_ids[0][i]
            link_name = self._link_ids[1][i]
            print(f"Link  {i:02d} - ID: {link_id:02d}, Name: {link_name}")
        for i in range(len(self._joint_ids[0])):
            default_pos = self._joint_default_pos[0, i].item()
            lower_pos = self._joint_lower_pos[0, i].item()
            upper_pos = self._joint_upper_pos[0, i].item()
            print(
                f"Joint  {i:02d} - ID: {self._joint_ids[0][i]:02d}, Name: {self._joint_ids[1][i]}, "
                f"Default: {default_pos:.4f} rad, "
                f"Limit: [{lower_pos:.4f}, {upper_pos:.4f}] rad"
            )
        for i in range(len(self._gimbal_ids[0])):
            default_pos = self._gimbal_default_pos[0, i].item()
            lower_pos = self._gimbal_lower_pos[0, i].item()
            upper_pos = self._gimbal_upper_pos[0, i].item()
            print(
                f"Gimbal {i:02d} - ID: {self._gimbal_ids[0][i]:02d}, Name: {self._gimbal_ids[1][i]}, "
                f"Default: {default_pos:.4f} rad, "
                f"Limit: [{lower_pos:.4f}, {upper_pos:.4f}] rad"
            )
        for i in range(len(self._thrust_ids[0])):
            direction = "Left" if self._rotor_directions[i] == 1 else "Right"
            print(
                f"Thrust {i:02d} - ID: {self._thrust_ids[0][i]:02d}, Name: {self._thrust_ids[1][i]}, "
                f"Direction: {direction}"
            )
        # for i in range(len(self._rotor_ids[0])):
        #     print(
        #         f"Rotor  {i:02d} - ID: {self._rotor_ids[0][i]:02d}, Name: {self._rotor_ids[1][i]}"
        #     )
        # masses = self._robot.root_physx_view.get_masses()[0]
        # body_ids = torch.arange(self._robot.num_bodies, dtype=torch.int)
        # print("Robot link masses: ")
        # for i, body_id in enumerate(body_ids):
        #     body_name = self._robot.body_names[body_id]
        #     body_mass = masses[i].item()
        #     print(f" - Body ID: {body_id}, Name: {body_name}, Mass: {body_mass:.4f} kg")
        print("=" * 60)

        # # Actuator Models =========================================================
        self._rotors = RotorGroup(
            cfg=cfg.rotorCfg,
            devices=self.device,
            num_envs=self.num_envs,
            rotor_ids=self._thrust_ids[0],
            rotor_names=self._thrust_ids[1],
            rotor_directions=self._rotor_directions,
        )

        # # Environment State =======================================================
        # # # Action ================================================================
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._last_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._last_last_actions = self._last_actions.clone()
        self._actions_scaled = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._action_joint_pos = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)
        self._last_action_joint_pos = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)
        self._action_gimbal_pos = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._last_action_gimbal_pos = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._action_thrust_force = torch.zeros(self.num_envs, self.cfg.rotor_num, device=self.device)
        self._last_action_thrust_force = torch.zeros(self.num_envs, self.cfg.rotor_num, device=self.device)

        # # # Observation ===========================================================
        self._observation_buffers = torch.zeros(
            self.num_envs,
            self.cfg.obs_history_len,
            self.cfg.single_obs_len,
            device=self.device,
        )

        # # # State ================================================================
        # # # # Disired Pose %%%%%%%%%%%%%%%%%
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_zyx_euler_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._desired_lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_ang_vel_w = torch.zeros(self.num_envs, 3, device=self.device)

        self._init_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._init_zyx_euler_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._init_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._position_error = torch.zeros(self.num_envs, 3, device=self.device)
        self._angle_error = torch.zeros(self.num_envs, 3, device=self.device)

        self._desired_joint_pos_w = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)
        self._joint_pos_error = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)

        self._links_pos_w = torch.zeros(self.num_envs, len(self._link_ids[0]), 3, device=self.device)
        self._links_quat_w = torch.zeros(self.num_envs, len(self._link_ids[0]), 4, device=self.device)
        self._links_lin_vel_w = torch.zeros(self.num_envs, len(self._link_ids[0]), 3, device=self.device)
        self._links_ang_vel_w = torch.zeros(self.num_envs, len(self._link_ids[0]), 3, device=self.device)
        self._links_lin_acc_w = torch.zeros(self.num_envs, len(self._link_ids[0]), 3, device=self.device)
        self._links_ang_acc_w = torch.zeros(self.num_envs, len(self._link_ids[0]), 3, device=self.device)

        self._thruster_pos_w = torch.zeros(self.num_envs, self.cfg.rotor_num, 3, device=self.device)
        self._thruster_quat_w = torch.zeros(self.num_envs, self.cfg.rotor_num, 4, device=self.device)
        self._thruster_lin_vel_w = torch.zeros(self.num_envs, self.cfg.rotor_num, 3, device=self.device)
        self._thruster_ang_vel_w = torch.zeros(self.num_envs, self.cfg.rotor_num, 3, device=self.device)

        # # # # Reset Event %%%%%%%%%%%%%%%%%%
        # # # # Reset Quat Event %%%%%%%%%%%%%
        # # # # Reset Pos  Event %%%%%%%%%%%%%
        # # # # Push Event %%%%%%%%%%%%%%%%%%%
        self._push_episode_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._push_done_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device).fill_(False)
        # short-duration impulse force applied independently to each link (in its body frame)
        self._num_links = len(self._link_ids[0])
        self._link_masses = self._robot.root_physx_view.get_masses()[0][self._link_ids[0]].to(self.device)  # (num_links,)
        self._push_impulse_steps_left = torch.zeros(self.num_envs, self._num_links, device=self.device, dtype=torch.int32)
        self._push_impulse_force_b = torch.zeros(self.num_envs, self._num_links, 3, device=self.device)
        self._external_force_b = torch.zeros(self.num_envs, self._num_links, 3, device=self.device)
        # Host-side upper bound for composer writes after an impulse is scheduled. The extra step
        # writes zeros once after the longest possible impulse, clearing the composer's persistent
        # force entries without polling a CUDA tensor from the CPU on every physics substep.
        self._push_wrench_write_steps_remaining = 0

        # # # # Stages %%%%%%%%%%%%%%%%%%%%%%%
        # If stage = 0 (mode = False), then no randomization/fault/disturbance is applied (defult to reset stage),
        # If stage = 1 (mode = False), then the randomization/fault/disturbance is fully applied
        # If mode = True, then the stage will gradually increase from 0 to 1 during the specified step range
        # , and the randomization/fault/disturbance will be scaled by the stage value
        self._vel_random_stage = 0.0
        self._quat_sample_stage = 0.0
        self._pos_sample_stage = 0.0
        self._gimbal_random_stage = 0.0
        self._joint_random_stage = 0.0
        self._push_stage = 0.0
        # static-hover task mixing: per-env mask (True = reference frozen at the RSI frame) and the
        # linear curriculum stage (0 -> 1) that drives the active fraction from rate_start to rate_end.
        self._static_hover_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._static_hover_stage = 0.0

        # per-cause death masks (set in _get_dones, logged as the Episode_Termination/ breakdown in
        # _reset_idx). Initialized here so the first (startup) reset, which logs before any _get_dones
        # call, reads valid buffers.
        self._died_position = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._died_orientation = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._died_joint = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._died_crash = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._died_drift = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Early-termination diagnostics are written as one CSV row per terminated env.
        self._early_termination_csv_path: str | None = None

        # # # # Reference Motion (target trajectory) %%%%%%%%%%%%%%%%%%
        # target clip, re-ordered to this environment's joint order. It drives the tracking task and the
        # RSI base/joint state. gimbal / rotor columns are requested only when they are the reset SOURCE,
        # i.e. the *_init_from_demo flag is on AND we are not in hybrid mode (in hybrid mode gimbal/thrust
        # come from _real_motion instead). Requesting a column the clip lacks raises in DragonMotion
        # (fail-loud) -- that is intended when the config expects it but the clip cannot provide it.
        self._motion = DragonMotion(
            self.cfg.demo_file,
            joint_names_env=self._joint_ids[1],
            gimbal_names_env=self._gimbal_ids[1]
            if (self.cfg.gimbal_init_from_demo and not self.cfg.hybrid_initialization)
            else None,
            rotor_names_env=self._thrust_ids[1]
            if (self.cfg.thrust_init_from_demo and not self.cfg.hybrid_initialization)
            else None,
            device=self.device,
        )
        if self.cfg.multi_trajectory_switch_stage:
            target_motion_files = list(self.cfg.multi_trajectory_demo_files)
            if len(target_motion_files) == 0:
                raise ValueError(
                    "multi_trajectory_switch_stage is enabled, but multi_trajectory_demo_files is empty."
                )
            normalized_motion_files = [os.path.realpath(path) for path in target_motion_files]
            if len(set(normalized_motion_files)) != len(normalized_motion_files):
                raise ValueError("The Stage 2 target trajectory pool contains duplicate files.")
            stage1_motion_path = os.path.realpath(self.cfg.demo_file)
            if stage1_motion_path not in normalized_motion_files:
                raise ValueError(
                    "multi_trajectory_demo_files must explicitly include demo_file when "
                    "multi_trajectory_switch_stage is enabled."
                )
            self._stage1_motion_id = normalized_motion_files.index(stage1_motion_path)
            target_motions = [
                self._motion
                if normalized_path == stage1_motion_path
                else DragonMotion(
                    motion_file,
                    joint_names_env=self._joint_ids[1],
                    device=self.device,
                )
                for motion_file, normalized_path in zip(target_motion_files, normalized_motion_files)
            ]
            target_motion_names = [os.path.basename(path) for path in target_motion_files]
        else:
            self._stage1_motion_id = 0
            target_motions = [self._motion]
            target_motion_names = [os.path.basename(self.cfg.demo_file)]
        self._motion_library = DragonMotionLibrary(
            target_motions,
            names=target_motion_names,
            step_dt=self.step_dt,
            reset_min_height=self.randomCfg.rsi_min_height,
        )
        # Hybrid RSI (training only): the AMP `real` clip, loaded here so _reset_idx can search it
        # for the nearest dynamically-feasible frame to the target start. Evaluation resets exactly
        # from its target clip; AMP subclasses load their expert clip separately when this is None.
        self._real_motion = None
        self._hybrid_init_logged = False
        if self.cfg.hybrid_initialization and not self.cfg.evaluate_mode:
            self._real_motion = DragonMotion(
                self.cfg.amp_demo_file,
                joint_names_env=self._joint_ids[1],
                gimbal_names_env=self._gimbal_ids[1],
                rotor_names_env=self._thrust_ids[1],
                device=self.device,
            )
        # virtual (monotonically increasing) playback phase in demo frames; the actual
        # frame index ping-pongs within [0, max_index] via triangle-wave reflection.
        self._motion_virtual = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # The selected target trajectory remains fixed from reset until the episode ends.
        self._motion_ids = torch.full(
            (self.num_envs,), self._stage1_motion_id, device=self.device, dtype=torch.long
        )
        # Integer demo frames advanced per control step. The library validates that every target
        # clip is aligned with the environment control rate during initialization.
        self._motion_phase_step = self._motion_library.phase_steps(self._motion_ids)
        self._evaluation_hold_steps = 0
        self._evaluation_clip_steps = 0
        self._evaluation_last_frame = 0
        if self.cfg.evaluate_mode:
            if self.cfg.evaluation_hold_time_s < 0.0:
                raise ValueError("evaluation_hold_time_s must be non-negative.")
            self._evaluation_hold_steps = round(self.cfg.evaluation_hold_time_s / self.step_dt)
            if not math.isclose(
                self._evaluation_hold_steps * self.step_dt,
                self.cfg.evaluation_hold_time_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("evaluation_hold_time_s must be an integer multiple of the control step.")
            phase_step = int(self._motion_phase_step[0].item())
            self._evaluation_last_frame = int(self._motion.max_index)
            self._evaluation_clip_steps = math.ceil(self._evaluation_last_frame / phase_step)
            evaluation_steps = 2 * self._evaluation_hold_steps + self._evaluation_clip_steps
            self.cfg.episode_length_s = evaluation_steps * self.step_dt
            print(
                f"[evaluation_timeline] trajectory={os.path.basename(self.cfg.demo_file)} | "
                f"hold_start={self._evaluation_hold_steps} steps | "
                f"clip={self._evaluation_clip_steps} steps | "
                f"hold_end={self._evaluation_hold_steps} steps | "
                f"episode={self.cfg.episode_length_s:.3f}s"
            )
        self._motion_future_offsets = torch.arange(
            1,
            self.cfg.track_horizon + 1,
            device=self.device,
            dtype=torch.long,
        )
        self._multi_trajectory_stage_logged = False
        # cached world-frame look-ahead targets (K future frames) for the observation
        self._target_pos_w = torch.zeros(self.num_envs, self.cfg.track_horizon, 3, device=self.device)
        self._target_quat_w = torch.zeros(self.num_envs, self.cfg.track_horizon, 4, device=self.device)
        self._target_quat_w[..., 0] = 1.0
        self._target_joint_pos_w = torch.zeros(
            self.num_envs, self.cfg.track_horizon, self.cfg.joint_num, device=self.device
        )
        # cached world-frame look-ahead reference base twist (K future frames); data path only.
        self._target_lin_vel_w = torch.zeros(self.num_envs, self.cfg.track_horizon, 3, device=self.device)
        self._target_ang_vel_w = torch.zeros(self.num_envs, self.cfg.track_horizon, 3, device=self.device)

        # # # # Rotor %%%%%%%%%%%%%%%%%%%%%%%%
        self._action_thrust_rate = torch.zeros(self.num_envs, self.cfg.paired_rotor_num, device=self.device)
        self._target_thrust_force = torch.zeros(self.num_envs, self.cfg.rotor_num, 3, device=self.device)
        self._target_rotor_torque = torch.zeros(self.num_envs, self.cfg.rotor_num, 3, device=self.device)
        self._thrust_force = torch.zeros(self.num_envs, self.cfg.rotor_num, device=self.device)
        self._thrust_limit_violation = torch.zeros(self.num_envs, self.cfg.rotor_num, device=self.device)

        # # # # Joint %%%%%%%%%%%%%%%%%%%%%%%%
        self._action_joint_rate = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)
        self._joint_fixed = torch.zeros(self.num_envs, self.cfg.joint_num, dtype=torch.bool, device=self.device).fill_(False)
        self._joint_pos = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)
        self._joint_pos_last = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)
        self._joint_vel = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)
        self._joint_vel_last = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)
        self._joint_acc = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)
        self._joint_limit_violation = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)
        self._joint_torque = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)
        self._joint_torque_power = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)
        self._joint_torque_limit_violation = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)

        # # # # Gimbal %%%%%%%%%%%%%%%%%%%%%%%
        self._action_gimbal_rate = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._gimbal_pos = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._gimbal_pos_last = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._gimbal_vel = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._gimbal_vel_last = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._gimbal_acc = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._gimbal_limit_violation = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._gimbal_torque = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._gimbal_torque_limit_violation = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)

        # # # # Metrics %%%%%%%%%%%%%%%%%%%%%
        self._contact = torch.zeros(self.num_envs, device=self.device)
        self._joint_contact = torch.zeros(self.num_envs, device=self.device)
        self._lin_vel = torch.zeros(self.num_envs, device=self.device)
        self._ang_vel = torch.zeros(self.num_envs, device=self.device)
        self._lin_acc = torch.zeros(self.num_envs, device=self.device)
        self._ang_acc = torch.zeros(self.num_envs, device=self.device)
        self._lin_vel_tracking = torch.zeros(self.num_envs, device=self.device)
        self._ang_vel_tracking = torch.zeros(self.num_envs, device=self.device)
        self._distance_to_goal = torch.zeros(self.num_envs, device=self.device)
        self._angular_to_goal = torch.zeros(self.num_envs, device=self.device)
        self._joint_pos_to_goal = torch.zeros(self.num_envs, device=self.device)
        self._thrust_power = torch.zeros(self.num_envs, self.cfg.rotor_num, device=self.device)
        self._joint_error = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)
        self._joint_smooth = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)
        self._joint_vibration = torch.zeros(self.num_envs, self.cfg.joint_num, device=self.device)
        self._gimbal_error = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._gimbal_smooth = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)
        self._gimbal_vibration = torch.zeros(self.num_envs, self.cfg.gimbal_num, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "lin_acc",
                "ang_acc",
                "lin_vel_tracking",
                "ang_vel_tracking",
                "thrust_power",
                "contact",
                "joint_contact",
                "died",
                "position_tracking",
                "position_deviation",
                "orientation_tracking",
                "orientation_deviation",
                "joint_tracking",
                "joint_deviation",
                "joint_action_rate",
                "gimbal_action_rate",
                "thrust_action_rate",
                "joint_vel",
                "joint_acc",
                "joint_limit",
                "gimbal_acc",
                "gimbal_limit",
                "thrust_limit",
                "joint_torque_power",
                "gimbal_torque_power",
                "joint_torque_limit",
                "gimbal_torque_limit",
            ]
        }

        # # # Setup Scene ============================================================
        self._foot_sensor_cfg = SceneEntityCfg("foot_contact_sensor", body_names=["inter_joint.*"])
        self._foot_sensor_cfg.resolve(self.scene)

        self._body_sensor_cfg = SceneEntityCfg("body_contact_sensor", body_names=["link.*"])
        self._body_sensor_cfg.resolve(self.scene)
        self._body_asset_cfg = SceneEntityCfg("robot", body_names=["link.*"])
        self._body_asset_cfg.resolve(self.scene)

        self._end_sensor_cfg = SceneEntityCfg("end_contact_sensor", body_names=[".*_end"])
        self._end_sensor_cfg.resolve(self.scene)
        self._end_asset_cfg = SceneEntityCfg("robot", body_names=[".*_end"])
        self._end_asset_cfg.resolve(self.scene)

        self._rotor_sensor_cfg = SceneEntityCfg("thrust_sensor", body_names=["edf.*"])
        self._rotor_sensor_cfg.resolve(self.scene)

        self._joint_asset_cfg = SceneEntityCfg("robot", joint_names=["joint.*"])
        self._joint_asset_cfg.resolve(self.scene)
        self._gimbal_asset_cfg = SceneEntityCfg("robot", joint_names=["gimbal.*"])
        self._gimbal_asset_cfg.resolve(self.scene)
        self._thrust_asset_cfg = SceneEntityCfg("robot", body_names=["edf.*"])
        self._thrust_asset_cfg.resolve(self.scene)

        # Create/update debug markers after the actuator, sensor, and target buffers exist, using
        # the final per-group flags initialized before the UI window was built.
        self._refresh_debug_vis()

    def _aars_compute_analytical_gimbal_pos(self, mount_quat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        mount_quat_w: [num_envs, num_mounts, 4] (w,x,y,z)
        returns roll, pitch: each [num_envs, num_mounts]

        Condition: R_mount @ Rx(roll) @ Ry(pitch) @ [0,0,1] = [0,0,1]
        """
        from isaaclab.utils.math import quat_apply_inverse
        z_world = torch.zeros(*mount_quat_w.shape[:-1], 3, device=mount_quat_w.device)
        z_world[..., 2] = 1.0

        # d = R_mount^T @ [0,0,1], which is the direction of the world z-axis in the mount frame
        d = quat_apply_inverse(mount_quat_w, z_world)  # [num_envs, num_mounts, 3]

        pitch = torch.arcsin(d[..., 0].clamp(-1.0, 1.0))                 # [num_envs, num_mounts]
        cp = torch.cos(pitch)
        roll = torch.where(cp.abs() < 1e-6,
                        torch.zeros_like(pitch),
                        torch.arctan2(-d[..., 1], d[..., 2]))         # [num_envs, num_mounts]
        return roll, pitch

    def _aars_build_paired_rotor_action_indices(self, rotor_names: Sequence[str]) -> torch.Tensor:
        """Map each physical rotor to a paired action index using the rotor name prefix."""
        pair_name_to_action_idx = {}
        pair_counts = {}
        rotor_action_indices = []

        for rotor_name in rotor_names:
            pair_name, separator, side = rotor_name.rpartition("_")
            if separator == "" or side not in ("left", "right"):
                raise ValueError(
                    f"Rotor '{rotor_name}' must end with '_left' or '_right' to build paired actions."
                )

            if pair_name not in pair_name_to_action_idx:
                pair_name_to_action_idx[pair_name] = len(pair_name_to_action_idx)

            rotor_action_indices.append(pair_name_to_action_idx[pair_name])
            pair_counts[pair_name] = pair_counts.get(pair_name, 0) + 1

        if len(pair_name_to_action_idx) != self.cfg.paired_rotor_num:
            raise ValueError(
                f"Expected {self.cfg.paired_rotor_num} rotor pairs, but found {len(pair_name_to_action_idx)}."
            )

        invalid_pairs = [pair_name for pair_name, count in pair_counts.items() if count != 2]
        if invalid_pairs:
            raise ValueError(f"Each rotor pair must contain exactly two rotors, invalid pairs: {invalid_pairs}")

        return torch.tensor(rotor_action_indices, device=self.device, dtype=torch.long)

    def _aars_expand_paired_rotor_actions(self, paired_rotor_actions: torch.Tensor) -> torch.Tensor:
        """Expand policy rotor-pair actions to per-rotor commands in simulator body order."""
        return paired_rotor_actions.index_select(1, self._rotor_action_indices)

    def _aars_compress_paired_rotor_actions(self, rotor_values: torch.Tensor) -> torch.Tensor:
        """Average left/right rotor values back into the paired action representation."""
        paired_rotor_values = torch.zeros(
            rotor_values.shape[0],
            self.cfg.paired_rotor_num,
            device=rotor_values.device,
            dtype=rotor_values.dtype,
        )
        pair_indices = self._rotor_action_indices.unsqueeze(0).expand(rotor_values.shape[0], -1)
        paired_rotor_values.scatter_add_(1, pair_indices, rotor_values)
        return paired_rotor_values * 0.5

    def _setup_scene(self):
        # # # add articulation to scene
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        # # # setup IMU sensor
        self._imu_sensor = Imu(self.cfg.imu_sensor)
        self.scene.sensors["imu_sensor"] = self._imu_sensor
        # # # setup contact sensor
        self._body_contact_sensor = ContactSensor(self.cfg.body_contact_sensor)
        self.scene.sensors["body_contact_sensor"] = self._body_contact_sensor
        self._foot_contact_sensor = ContactSensor(self.cfg.foot_contact_sensor)
        self.scene.sensors["foot_contact_sensor"] = self._foot_contact_sensor
        self._end_contact_sensor = ContactSensor(self.cfg.end_contact_sensor)
        self.scene.sensors["end_contact_sensor"] = self._end_contact_sensor
        self._thrust_sensor = ContactSensor(self.cfg.thrust_sensor)
        self.scene.sensors["thrust_sensor"] = self._thrust_sensor
        self._joint_contact_sensor = ContactSensor(self.cfg.joint_contact_sensor)
        self.scene.sensors["joint_contact_sensor"] = self._joint_contact_sensor
        # # # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
            if self.cfg.use_state_estimator:
                self._est_lin_vel_b, self._est_ang_vel_b = self._stateEstimator.update(
                    self._imu_sensor.data.pos_w,
                    self._imu_sensor.data.quat_w,
                    self._imu_sensor.data.lin_vel_b,
                    self._imu_sensor.data.ang_vel_b,
                )

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # advance the reference-motion phase and refresh look-ahead targets AFTER reward/dones
        # (so they grade against the same frame the policy observed) and before reset / obs,
        # so the next observation carries the next reference frame.
        # static-hover envs keep their phase frozen (zero advance) so their reference stays put.
        if self.cfg.evaluate_mode:
            self._aars_set_evaluation_motion_phase(self._robot._ALL_INDICES)
        else:
            self._motion_virtual += self._motion_phase_step * (~self._static_hover_active).to(torch.long)
        self._aars_set_motion_targets(self._robot._ALL_INDICES)

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _aars_scale_actions_for_observation(self, actions: torch.Tensor) -> torch.Tensor:
        """Scale policy actions for observations."""
        actions_scaled = actions.clone()
        joint_end = self.cfg.joint_num
        gimbal_end = joint_end + self.cfg.gimbal_num

        actions_scaled[:, :joint_end] *= self.ctrlCfg.joint_action_scale
        gimbal_actions_scaled = actions_scaled[:, joint_end:gimbal_end]
        gimbal_actions_scaled[:, self.ctrlCfg.gimbal_pitch_ids] *= self.ctrlCfg.gimbal_pitch_action_scale
        gimbal_actions_scaled[:, self.ctrlCfg.gimbal_roll_ids] *= self.ctrlCfg.gimbal_roll_action_scale
        actions_scaled[:, gimbal_end:] *= self.ctrlCfg.thrust_action_scale
        actions_scaled[:, gimbal_end:] *= self.obsScales.thrust
        return actions_scaled

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._last_last_actions = self._last_actions.clone()
        self._last_actions = self._actions.clone()
        self._last_action_joint_pos = self._action_joint_pos.clone()
        self._last_action_gimbal_pos = self._action_gimbal_pos.clone()
        self._last_action_thrust_force = self._action_thrust_force.clone()
        self._actions = torch.clip(actions, -self.ctrlCfg.clip_actions, self.ctrlCfg.clip_actions).to(self.device)  # TODO: check action limits
        self._action_joint_pos = (
            self._actions[:, : self.cfg.joint_num] * self.ctrlCfg.joint_action_scale + self._joint_default_pos
        )
        self._action_gimbal_pos = (
            self._actions[:, self.cfg.joint_num: self.cfg.joint_num + self.cfg.gimbal_num].clone()
        )
        self._action_gimbal_pos[:, self.ctrlCfg.gimbal_pitch_ids] = (
            self._action_gimbal_pos[:, self.ctrlCfg.gimbal_pitch_ids]
            * self.ctrlCfg.gimbal_pitch_action_scale
            + self._gimbal_default_pos[:, self.ctrlCfg.gimbal_pitch_ids]
        )
        self._action_gimbal_pos[:, self.ctrlCfg.gimbal_roll_ids] = (
            self._action_gimbal_pos[:, self.ctrlCfg.gimbal_roll_ids]
            * self.ctrlCfg.gimbal_roll_action_scale
            + self._gimbal_default_pos[:, self.ctrlCfg.gimbal_roll_ids]
        )
        paired_rotor_actions = self._actions[
            :, self.cfg.joint_num + self.cfg.gimbal_num
            : self.cfg.joint_num + self.cfg.gimbal_num + self.cfg.paired_rotor_num
        ]
        self._actions_scaled[:] = self._aars_scale_actions_for_observation(self._actions)
        self._action_thrust_force = self._aars_expand_paired_rotor_actions(
            paired_rotor_actions
        ) * self.ctrlCfg.thrust_action_scale + self.ctrlCfg.thrust_hover  # shape: (N, 8)
        # schedule any new push disturbance for this env step
        self._aars_update_push_impulse_wrench()
        # #################################################
        # self._target_rotor_torque = (
        #     self.cfg.thrust_to_torque_ratio
        #     * self._target_thrust_force
        #     * torch.tensor(self.cfg.rotor_direction, device=self.device)
        # )
        # self._target_thrust_force[:, :] = 0
        # print("Rotor velocities: ", self._rotor_vel[0])
        # print("Rotor velocities: ", self._robot.data.joint_vel[:, self._rotor_ids[0]][0])
        # print("Rotor positions: ", self._robot.data.joint_pos[:, self._rotor_ids[0]][0])
        #################################################
        # print("Action debug info:")
        # print(" Gimbal pos targets: ", self._action_gimbal_pos[0])
        # print(" Rotor thrust targets: ", self._target_thrust_force[0])
        # print(" Rotor torque targets: ", self._target_rotor_torque[0])
        #################################################

    def _apply_action(self) -> None:
        """Apply the action to the robot. Every dt step"""
        # Apply action delay
        self._action_joint_pos_delayed = self._actJointDelayModel.update(self._action_joint_pos)
        self._action_gimbal_pos_delayed = self._actGimbalDelayModel.update(self._action_gimbal_pos)
        self._action_thrust_force_delayed = self._actThrustDelayModel.update(self._action_thrust_force)
        self._target_thrust_force, self._target_rotor_torque = self._rotors.forward(self._action_thrust_force_delayed)
        # Apply action
        self._robot.set_joint_position_target(self._action_joint_pos_delayed, self._joint_ids[0])
        self._robot.set_joint_position_target(self._action_gimbal_pos_delayed, self._gimbal_ids[0])
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self._target_thrust_force,
            torques=self._target_rotor_torque,
            body_ids=self._thrust_body_ids_tensor,
            env_ids=self._robot._ALL_INDICES_WP,
        )
        # apply the push disturbance as an external wrench on the base body
        self._aars_apply_push_wrench()

    def _aars_update_push_impulse_wrench(self) -> None:
        """Schedule short-duration impulse forces independently on each link for envs whose push time has arrived."""
        if not self.cfg.push_mode or self._push_stage <= 0.0:
            return

        push_mask = (self.episode_length_buf >= self._push_episode_step) & (~self._push_done_ids)
        push_env_ids = torch.nonzero(push_mask, as_tuple=False).squeeze(-1)
        if push_env_ids.numel() == 0:
            return

        n = push_env_ids.numel()
        # each link is independently pushed with probability push_link_prob (links may be pushed simultaneously)
        link_active = (
            torch.rand(n, self._num_links, device=self.device) < self.cfg.push_link_prob
        ).unsqueeze(-1)  # (n, num_links, 1)

        impulse_duration_steps = torch.randint(
            int(self.cfg.push_impulse_duration_min_steps),
            int(self.cfg.push_impulse_duration_max_steps) + 1,
            (n, self._num_links),
            device=self.device,
            dtype=self._push_impulse_steps_left.dtype,
        )
        impulse_duration_s = impulse_duration_steps.to(torch.float32).unsqueeze(-1) * self.physics_dt

        # independent body-frame velocity jump per link -> mass-scaled constant impulse force over its duration
        delta_lin_vel_b = (
            torch.empty(n, self._num_links, 3, device=self.device).uniform_(-1.0, 1.0)
            * self.cfg.push_lin_vel_range
            * self._push_stage
        )
        impulse_force_b = self._link_masses.view(1, -1, 1) * delta_lin_vel_b / impulse_duration_s

        self._push_impulse_force_b[push_env_ids] = impulse_force_b * link_active
        self._push_impulse_steps_left[push_env_ids] = (
            impulse_duration_steps * link_active.squeeze(-1).to(impulse_duration_steps.dtype)
        )
        self._push_done_ids[push_env_ids] = True
        self._push_wrench_write_steps_remaining = max(
            self._push_wrench_write_steps_remaining,
            int(self.cfg.push_impulse_duration_max_steps) + 1,
        )

    def _aars_apply_push_wrench(self) -> None:
        """Apply the active per-link push impulse as external forces on the links via the wrench composer."""
        # _aars_update_push_impulse_wrench never schedules an impulse at stage zero, so there is
        # nothing to apply or clear before the push curriculum starts. This avoids a full-device
        # buffer update and wrench-composer write on every physics substep during the warm-up.
        if (
            not self.cfg.push_mode
            or self._push_stage <= 0.0
            or self._push_wrench_write_steps_remaining <= 0
        ):
            return

        # Keep the physics hot path asynchronous. In particular, do not branch on
        # torch.any(...).item(): that synchronizes the CPU with the CUDA stream once per physics
        # substep. Inactive entries become zero and active countdowns are decremented in place.
        impulse_active_mask = (self._push_impulse_steps_left > 0).unsqueeze(-1)
        torch.mul(self._push_impulse_force_b, impulse_active_mask, out=self._external_force_b)
        self._push_impulse_steps_left.sub_(1).clamp_min_(0)

        # body-frame forces on the links; the thrust bodies (edf) keep their own composer entries
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=self._external_force_b,
            torques=None,
            body_ids=self._link_body_ids_tensor,
            env_ids=self._robot._ALL_INDICES_WP,
        )
        self._push_wrench_write_steps_remaining -= 1

    def _aars_get_static_hover_rate(self) -> float:
        """Fraction of envs that should be reset as static-hover, linearly interpolated by stage."""
        if not self.cfg.static_hover_mode:
            return 0.0
        rate = self.cfg.static_hover_rate_start + self._static_hover_stage * (
            self.cfg.static_hover_rate_end - self.cfg.static_hover_rate_start
        )
        lower = min(self.cfg.static_hover_rate_start, self.cfg.static_hover_rate_end)
        upper = max(self.cfg.static_hover_rate_start, self.cfg.static_hover_rate_end)
        return float(min(max(rate, lower), upper))        

    def _aars_training_progress(self) -> int:
        """Return control-step progress, including the configured offset for resumed runs."""
        progress = int(self.common_step_counter)
        if self.cfg.resume_mode:
            progress += int(self.cfg.resume_step_count)
        return progress

    def _aars_multi_trajectory_stage_active(self, progress: int | None = None) -> bool:
        """Whether newly resetting environments should enter multi-trajectory Stage 2."""
        if not self.cfg.multi_trajectory_switch_stage:
            return False
        if progress is None:
            progress = self._aars_training_progress()
        return progress >= int(self.cfg.multi_trajectory_switch_step_count)

    def _aars_set_evaluation_motion_phase(self, env_ids: Sequence[int]) -> None:
        """Map evaluation episode time onto one forward, endpoint-clamped trajectory pass."""
        clip_step = (self.episode_length_buf[env_ids] - self._evaluation_hold_steps).clamp(
            min=0,
            max=self._evaluation_clip_steps,
        )
        phase = clip_step * self._motion_phase_step[env_ids]
        self._motion_virtual[env_ids] = phase.clamp(max=self._evaluation_last_frame)

    def _aars_set_motion_targets(self, env_ids: Sequence[int]) -> None:
        """Fill the world-frame look-ahead target buffers from the current playback phase.

        Builds ``track_horizon`` future reference frames starting one step ahead of the current
        phase and stores both the full horizon (for the observation) and the first frame
        (``_desired_*``, used by reward/termination). Training uses ping-pong reflection at the
        clip ends; evaluation follows its endpoint-clamped, single-pass timeline.
        """
        v = self._motion_virtual[env_ids]  # (M,)
        m = len(env_ids)
        if self.cfg.evaluate_mode:
            # Build look-ahead from episode time so it follows the same start-hold -> one forward
            # clip -> end-hold timeline as the current phase. Clamping prevents ping-pong playback.
            future_episode_step = self.episode_length_buf[env_ids].unsqueeze(1) + (
                self.cfg.track_frame_stride * self._motion_future_offsets.unsqueeze(0)
            )
            future_clip_step = (future_episode_step - self._evaluation_hold_steps).clamp(
                min=0,
                max=self._evaluation_clip_steps,
            )
            future_virtual = future_clip_step * self._motion_phase_step[env_ids].unsqueeze(1)
            future_virtual.clamp_(max=self._evaluation_last_frame)
        else:
            # per-env look-ahead stride: 0 for static-hover envs so the whole K-step horizon
            # collapses onto the frozen current frame -> a constant zero-velocity setpoint.
            stride = self._motion_phase_step[env_ids] * self.cfg.track_frame_stride * (
                (~self._static_hover_active[env_ids]).to(torch.long)
            )  # (M,)
            # virtual phases of the K look-ahead frames (frame 0 = next reference frame)
            future_virtual = (
                v.unsqueeze(1) + stride.unsqueeze(1) * self._motion_future_offsets.unsqueeze(0)
            )  # (M, K)
        k = self.cfg.track_horizon
        future_motion_ids = self._motion_ids[env_ids].unsqueeze(1).expand(-1, k)
        ref, _, play_sign = self._motion_library.query(future_motion_ids, future_virtual)
        if self.cfg.evaluate_mode:
            # At the exact final frame the generic ping-pong query reports a reverse-play sign.
            # Evaluation clamps there instead, so retain the trajectory's forward velocity sign.
            play_sign = torch.ones_like(play_sign)

        pos = ref["base_pos_w"].reshape(m, k, 3) + self._terrain.env_origins[env_ids][:, None, :]
        quat = ref["base_quat_w"].reshape(m, k, 4)
        # joint targets are stored relative to the default pose (matches the joint-pos observation)
        joint = ref["joint_pos"].reshape(m, k, self.cfg.joint_num) - self._joint_default_pos[env_ids][:, None, :]
        # reference base twist (world frame); reversed when the clip is played backward
        play_sign = play_sign.reshape(m, k, 1)
        lin_vel = ref["base_lin_vel_w"].reshape(m, k, 3) * play_sign
        ang_vel = ref["base_ang_vel_w"].reshape(m, k, 3) * play_sign

        self._target_pos_w[env_ids] = pos
        self._target_quat_w[env_ids] = quat
        self._target_joint_pos_w[env_ids] = joint
        self._target_lin_vel_w[env_ids] = lin_vel
        self._target_ang_vel_w[env_ids] = ang_vel

        self._desired_pos_w[env_ids] = pos[:, 0]
        self._desired_quat_w[env_ids] = quat[:, 0]
        self._desired_joint_pos_w[env_ids] = joint[:, 0]
        self._desired_lin_vel_w[env_ids] = lin_vel[:, 0]
        self._desired_ang_vel_w[env_ids] = ang_vel[:, 0]

    def _aars_compute_target_obs(self, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
        """Express the cached look-ahead targets in the current robot frame.

        Returns a ``(num_envs, track_horizon * per_target_frame_len)`` tensor, each frame being
        ``[relative position (3), relative rotation 6D (6), target joint position (joint_num)]``.
        The relative rotation encodes the rotation from the current orientation to the target.
        """
        n = self.num_envs
        k = self.cfg.track_horizon
        mm = n * k
        rp = root_pos_w[:, None, :].expand(n, k, 3).reshape(mm, 3)
        rq = root_quat_w[:, None, :].expand(n, k, 4).reshape(mm, 4)
        rel_pos, rel_quat = subtract_frame_transforms(
            rp, rq, self._target_pos_w.reshape(mm, 3), self._target_quat_w.reshape(mm, 4)
        )
        rel_rot6 = matrix_from_quat(rel_quat)[:, :2, :].reshape(n, k, 6)
        rel_pos = rel_pos.reshape(n, k, 3)
        target_obs = torch.cat((rel_pos, rel_rot6, self._target_joint_pos_w), dim=-1)
        return target_obs.reshape(n, -1)

    def _get_observations(self) -> dict:
        # root_pos_w = self._obsPosDelayModel.update(self._robot.data.root_pos_w)
        # root_quat_w = self._obsQuatDelayModel.update(self._imu_sensor.data.quat_w)
        # root_pos_w = self._robot.data.root_pos_w
        # root_quat_w = self._robot.data.root_quat_w
        root_quat_w = self._imu_sensor.data.quat_w
        root_pos_w = self._imu_sensor.data.pos_w
        # root_lin_vel_w = self._obsLinVelDelayModel.update(self._robot.data.root_lin_vel_w)
        # root_ang_vel_b = self._obsAngVelDelayModel.update(self._robot.data.root_ang_vel_b)
        # root_lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)
        if self.cfg.use_state_estimator:
            root_lin_vel_b = self._obsLinVelDelayModel.update(self._est_lin_vel_b)
            root_ang_vel_b = self._obsAngVelDelayModel.update(self._est_ang_vel_b)
        else:
            root_lin_vel_b = self._obsLinVelDelayModel.update(self._imu_sensor.data.lin_vel_b)
            root_ang_vel_b = self._obsAngVelDelayModel.update(self._imu_sensor.data.ang_vel_b)
        # IMU Debug Info ========================================================
        # imu_ang_vel_b = quat_apply(self._imu_quat_body, self._robot.data.body_ang_vel_w[:, self._imu_id])
        # print("IMU Link   Ang Vel (body frame): ", imu_ang_vel_b[0])
        # print("Root       Ang Vel (body frame): ", self._robot.data.root_ang_vel_b[0])
        # print("IMU Sensor Ang Vel (body frame): ", self._imu_sensor.data.ang_vel_b[0])
        joint_pos = self._obsJointPosDelayModel.update(
            self._robot.data.joint_pos[:, self._joint_ids[0]] - self._joint_default_pos
        )
        joint_vel = self._obsJointVelDelayModel.update(
            self._robot.data.joint_vel[:, self._joint_ids[0]]
        )
        gimbal_pos = self._obsGimbalPosDelayModel.update(
            self._robot.data.joint_pos[:, self._gimbal_ids[0]] - self._gimbal_default_pos
        )
        gimbal_vel = self._obsGimbalVelDelayModel.update(
            self._robot.data.joint_vel[:, self._gimbal_ids[0]]
        )

        joint_pos_delta = joint_pos - self._joint_pos_last
        gimbal_pos_delta = gimbal_pos - self._gimbal_pos_last

        self._joint_pos_last = joint_pos.clone()
        self._gimbal_pos_last = gimbal_pos.clone()

        if self.cfg.add_noise:
            if "lin_vel" in self.noiseModel.params:
                root_lin_vel_b = self.noiseModel.apply(root_lin_vel_b, "lin_vel")
            if "ang_vel" in self.noiseModel.params:
                root_ang_vel_b = self.noiseModel.apply(root_ang_vel_b, "ang_vel")
            if "root_pos" in self.noiseModel.params:
                root_pos_w = self.noiseModel.apply(root_pos_w, "root_pos")
            if "root_quat" in self.noiseModel.params:
                axis = torch.randn(self.num_envs, 3, device=self.device)
                axis = axis / axis.norm(dim=-1, keepdim=True)
                angle = torch.empty(self.num_envs, 1, device=self.device).uniform_(
                    -self.cfg.noiseCfg["root_quat"]["std"], self.cfg.noiseCfg["root_quat"]["std"]
                )
                box_quat = quat_from_angle_axis(angle.squeeze(1), axis)
                root_quat_w = quat_mul(box_quat, root_quat_w)
            if "joint_pos" in self.noiseModel.params:
                joint_pos = self.noiseModel.apply(joint_pos, "joint_pos")
            if "joint_vel" in self.noiseModel.params:
                joint_vel = self.noiseModel.apply(joint_vel, "joint_vel")
            if "joint_pos_delta" in self.noiseModel.params:
                joint_pos_delta = self.noiseModel.apply(joint_pos_delta, "joint_pos_delta")
            if "gimbal_pos" in self.noiseModel.params:
                gimbal_pos = self.noiseModel.apply(gimbal_pos, "gimbal_pos")
            if "gimbal_vel" in self.noiseModel.params:
                gimbal_vel = self.noiseModel.apply(gimbal_vel, "gimbal_vel")
            if "gimbal_pos_delta" in self.noiseModel.params:
                gimbal_pos_delta = self.noiseModel.apply(gimbal_pos_delta, "gimbal_pos_delta")

        clip_obs = self.ctrlCfg.clip_observations

        if self.cfg.replace_dof_vel_with_last_pos:
            joint_dof_vel_obs = joint_pos_delta * self.obsScales.dof_delta
            gimbal_dof_vel_obs = gimbal_pos_delta * self.obsScales.dof_delta
        else:
            joint_dof_vel_obs = joint_vel * self.obsScales.dof_vel
            gimbal_dof_vel_obs = gimbal_vel * self.obsScales.dof_vel

        # === target obs : K look-ahead reference frames in the current robot frame ====
        target_obs = self._aars_compute_target_obs(root_pos_w, root_quat_w)
        target_obs = torch.clamp(target_obs, -clip_obs, clip_obs)

        # === current obs : proprioception at the current step =========================
        root_rot_vec = matrix_from_quat(root_quat_w)[:, :2, :].reshape(self.num_envs, 6)
        current_obs = torch.cat(
            (
                joint_pos,
                joint_dof_vel_obs,
                gimbal_pos,
                gimbal_dof_vel_obs,
                root_rot_vec,
                root_lin_vel_b * self.obsScales.lin_vel,
                root_ang_vel_b * self.obsScales.ang_vel,
                self._actions_scaled,
            ),
            dim=-1,
        )
        current_obs = torch.clamp(current_obs, -clip_obs, clip_obs)

        # === history obs : [t-1, t-2, ..., t-H], newest completed frame first ==========
        if self.cfg.history_mode:
            # Clone before mutating the backing buffer. Without the clone, history_obs is a view and
            # the update below silently changes slot 0 to the current t frame, duplicating current_obs.
            history_obs = self._observation_buffers.reshape(
                self.num_envs, -1
            ).clone()
            self._observation_buffers[:, 1:] = self._observation_buffers[:, :-1].clone()
            self._observation_buffers[:, 0] = current_obs
        else:
            history_obs = current_obs[:, :0]  # empty (num_envs, 0)

        # === policy observation = target | current(t) | history(t-1 ... t-H) ==========
        actor_obs = torch.cat((target_obs, current_obs, history_obs), dim=-1)

        critic_obs = self._get_states()
        critic_obs = torch.clamp(critic_obs, -clip_obs, clip_obs)
        observations = {"policy": actor_obs, "critic": critic_obs}
        return observations

    def _get_states(self) -> torch.Tensor:
        # privileged proprioception (no observation noise / delay)
        root_pos_w = self._imu_sensor.data.pos_w
        root_quat_w = self._imu_sensor.data.quat_w
        root_rot_vec = matrix_from_quat(root_quat_w)[:, :2, :].reshape(self.num_envs, 6)

        current_state = torch.cat(
            (
                self._robot.data.joint_pos[:, self._joint_ids[0]] - self._joint_default_pos,
                self._robot.data.joint_vel[:, self._joint_ids[0]] * self.obsScales.dof_vel,
                self._robot.data.joint_pos[:, self._gimbal_ids[0]] - self._gimbal_default_pos,
                self._robot.data.joint_vel[:, self._gimbal_ids[0]] * self.obsScales.dof_vel,
                root_rot_vec,
                self._imu_sensor.data.lin_vel_b * self.obsScales.lin_vel,
                self._imu_sensor.data.ang_vel_b * self.obsScales.ang_vel,
                self._actions_scaled,
            ),
            dim=-1,
        )

        # same look-ahead target block, computed from the true (privileged) base pose
        target_state = self._aars_compute_target_obs(root_pos_w, root_quat_w)

        target_thrust_force = self._target_thrust_force[:, :, 2] * self.obsScales.thrust
        target_rotor_torque = self._target_rotor_torque[:, :, 2] * self.obsScales.torque

        # critic state = current | target | last action | privileged rotor thrust/torque
        states = torch.cat(
            (current_state, target_state, target_thrust_force, target_rotor_torque),
            dim=-1,
        )
        return states

    def _get_rewards(self) -> torch.Tensor:
        pos_err, rot_err = compute_pose_error(
            # self._robot.data.root_pos_w,
            # self._robot.data.root_quat_w,
            self._imu_sensor.data.pos_w,
            self._imu_sensor.data.quat_w,
            self._desired_pos_w,
            self._desired_quat_w,
        )
        self._position_error = pos_err
        self._angle_error = rot_err

        self._joint_pos_error = self._robot.data.joint_pos[:, self._joint_ids[0]] - self._joint_default_pos - self._desired_joint_pos_w

        self._links_pos_w = self._robot.data.body_com_pose_w[:, self._link_ids[0]]
        self._links_quat_w = self._robot.data.body_com_quat_w[:, self._link_ids[0]]
        self._links_lin_vel_w = self._robot.data.body_com_lin_vel_w[:, self._link_ids[0]]
        self._links_ang_vel_w = self._robot.data.body_com_ang_vel_w[:, self._link_ids[0]]
        self._links_lin_acc_w = self._robot.data.body_com_lin_acc_w[:, self._link_ids[0]]
        self._links_ang_acc_w = self._robot.data.body_com_ang_acc_w[:, self._link_ids[0]]

        self._body_lin_vel_b = self._imu_sensor.data.lin_vel_b
        self._body_ang_vel_b = self._imu_sensor.data.ang_vel_b

        # # Velocity Rewards ---------------------------------------------
        lin_vel_norm = torch.linalg.norm(self._links_lin_vel_w, dim=-1)
        self._lin_vel = lin_vel_norm
        lin_vel_reward = torch.sum(torch.square(lin_vel_norm), dim=1)

        ang_vel_norm = torch.linalg.norm(self._links_ang_vel_w, dim=-1)
        self._ang_vel = ang_vel_norm
        ang_vel_reward = torch.sum(torch.square(ang_vel_norm), dim=1)

        # # Acceleration Rewards -----------------------------------------
        lin_acc_norm = torch.linalg.norm(self._links_lin_acc_w, dim=-1)
        self._lin_acc = lin_acc_norm
        lin_acc_reward = torch.sum(torch.square(lin_acc_norm), dim=1)

        ang_acc_norm = torch.linalg.norm(self._links_ang_acc_w, dim=-1)
        self._ang_acc = ang_acc_norm
        ang_acc_reward = torch.sum(torch.square(ang_acc_norm), dim=1)

        # # Tracking Rewards (additive exp form) ------------------------
        # exp(-||err||^2 / std^2) in (0, 1]; one term per tracked quantity, summed below.
        # The per-joint exp's are folded into a single L2 term: exp(-sum_j dq_j^2 / std^2)
        # == prod_j exp(-dq_j^2 / std^2), so summing the joint errors inside the exp is
        # exactly the "multiply the 6 joints together" form -- one std, one knob to tune.
        position_sq_err = torch.sum(torch.square(self._position_error), dim=1)
        distance_to_goal = torch.linalg.norm(self._position_error, dim=1)
        self._distance_to_goal = distance_to_goal
        # position_tracking_reward = torch.exp(-position_sq_err / self.cfg.position_tracking_std**2)
        position_tracking_reward = 1 - torch.tanh(distance_to_goal / self.cfg.position_tracking_std)
        position_deviation_reward = position_sq_err

        angular_sq_err = torch.sum(torch.square(self._angle_error), dim=1)
        angular_to_goal = torch.linalg.norm(self._angle_error, dim=1)
        self._angular_to_goal = angular_to_goal
        # orientation_tracking_reward = torch.exp(-angular_sq_err / self.cfg.orientation_tracking_std**2)
        orientation_tracking_reward = 1 - torch.tanh(angular_to_goal / self.cfg.orientation_tracking_std)
        orientation_deviation_reward = angular_sq_err

        joint_sq_err = torch.sum(torch.square(self._joint_pos_error), dim=1)
        joint_pos_to_goal = torch.linalg.norm(self._joint_pos_error, dim=1)
        self._joint_pos_to_goal = joint_pos_to_goal
        # joint_tracking_reward = torch.exp(-joint_sq_err / self.cfg.joint_tracking_std**2)
        joint_tracking_reward = 1 - torch.tanh(joint_pos_to_goal / self.cfg.joint_tracking_std)
        joint_deviation_reward = joint_sq_err

        body_lin_vel_w = self._robot.data.root_lin_vel_w
        body_ang_vel_w = self._robot.data.root_ang_vel_w

        # # Tracking Rewards (additive L2 form) ------------------------
        lin_vel_tracking = torch.sum(torch.square(body_lin_vel_w - self._desired_lin_vel_w), dim=1)
        self._lin_vel_tracking = lin_vel_tracking
        ang_vel_tracking = torch.sum(torch.square(body_ang_vel_w - self._desired_ang_vel_w), dim=1)
        self._ang_vel_tracking = ang_vel_tracking
        # squared L2 error form
        lin_vel_tracking_reward = lin_vel_tracking
        ang_vel_tracking_reward = ang_vel_tracking
        # exp(-||err||^2 / std^2) in (0, 1]; form
        # lin_vel_sq_err = torch.sum(torch.square(body_lin_vel_w - self._desired_lin_vel_w), dim=1)
        # lin_vel_tracking_reward = torch.exp(-lin_vel_sq_err / self.cfg.body_lin_vel_tracking_std**2)
        # ang_vel_sq_err = torch.sum(torch.square(body_ang_vel_w - self._desired_ang_vel_w), dim=1)
        # ang_vel_tracking_reward = torch.exp(-ang_vel_sq_err / self.cfg.body_ang_vel_tracking_std**2)

        # net_forces_w shape: [num_envs, num_bodies, 3], center_link has num_bodies=1
        body_contact_reward = torch.linalg.norm(
            self._body_contact_sensor.data.net_forces_w, dim=-1  # [num_envs, num_bodies]
        ).sum(dim=1)  # [num_envs]
        rotor_contact_reward = torch.linalg.norm(
            self._thrust_sensor.data.net_forces_w, dim=-1  # [num_envs, num_bodies]
        ).sum(dim=1)  # [num_envs]
        foot_contact_reward = torch.linalg.norm(
            self._foot_contact_sensor.data.net_forces_w, dim=-1  # [num_envs, num_bodies]
        ).sum(dim=1)  # [num_envs]
        contact_reward = body_contact_reward + rotor_contact_reward + foot_contact_reward
        self._contact = contact_reward
        # mean over the substep history (history_length = decimation) ~ step-averaged stop force,
        # so impacts inside earlier substeps are not missed
        joint_contact_reward = torch.linalg.norm(
            self._joint_contact_sensor.data.net_forces_w_history, dim=-1  # [num_envs, decimation, num_joint_modules]
        ).mean(dim=1).sum(dim=1)  # [num_envs]
        self._joint_contact = joint_contact_reward

        # # Goal reaching reward ----------------------------------------
        body_ang_vel_norm = torch.linalg.norm(self._body_ang_vel_b, dim=-1)
        body_lin_vel_norm = torch.linalg.norm(self._body_lin_vel_b, dim=-1)

        # Action-rate penalties use scaled command deltas so the values carry physical units.
        self._action_joint_rate = (
            self._actions[:, : self.cfg.joint_num] - self._last_actions[:, : self.cfg.joint_num]
        ) * self.ctrlCfg.joint_action_scale
        joint_action_rate_reward = torch.sum(torch.square(self._action_joint_rate), dim=1)

        self._joint_pos = (self._robot.data.joint_pos[:, self._joint_ids[0]] - self._joint_default_pos).clone()
        self._joint_vel = (self._robot.data.joint_vel[:, self._joint_ids[0]]).clone()
        self._joint_acc = (self._joint_vel - self._joint_vel_last).clone()
        self._joint_error = self._joint_pos - self._action_joint_pos
        self._joint_smooth = self._last_actions[:, : self.cfg.joint_num] - 2 * self._actions[:, : self.cfg.joint_num] + self._last_last_actions[:, : self.cfg.joint_num]
        joint_vibration_ids = self._joint_vel * self._joint_vel_last < 0.0
        self._joint_vibration = self._joint_vel * self._joint_vel_last * joint_vibration_ids.to(torch.float32) * -1.0
        self._joint_vel_last = self._joint_vel.clone()

        # same as isaaclab mdp.joint_pos_limits (whole_body_tracking): absolute joint pos vs the
        # articulation soft limits, i.e. URDF limits scaled by soft_joint_pos_limit_factor
        joint_pos_abs = self._robot.data.joint_pos[:, self._joint_ids[0]]
        joint_soft_limits = self._robot.data.soft_joint_pos_limits[:, self._joint_ids[0]]
        joint_limit = -(joint_pos_abs - joint_soft_limits[..., 0]).clip(max=0.0)
        joint_limit += (joint_pos_abs - joint_soft_limits[..., 1]).clip(min=0.0)
        self._joint_limit_violation = joint_limit.clone()

        joint_vel_reward = torch.sum(torch.square(self._joint_vel), dim=1)
        joint_acc_reward = torch.sum(torch.square(self._joint_acc), dim=1)
        joint_limit_reward = torch.sum(joint_limit, dim=1)

        _gimbal_act = slice(self.cfg.joint_num, self.cfg.joint_num + self.cfg.gimbal_num)
        self._action_gimbal_rate = (self._actions[:, _gimbal_act] - self._last_actions[:, _gimbal_act]).clone()
        self._action_gimbal_rate[:, self.ctrlCfg.gimbal_pitch_ids] *= self.ctrlCfg.gimbal_pitch_action_scale
        self._action_gimbal_rate[:, self.ctrlCfg.gimbal_roll_ids] *= self.ctrlCfg.gimbal_roll_action_scale
        gimbal_action_rate_reward = torch.sum(torch.square(self._action_gimbal_rate), dim=1)
        self._gimbal_pos = (self._robot.data.joint_pos[:, self._gimbal_ids[0]] - self._gimbal_default_pos).clone()
        self._gimbal_vel = (self._robot.data.joint_vel[:, self._gimbal_ids[0]]).clone()
        self._gimbal_acc = (self._gimbal_vel - self._gimbal_vel_last).clone()
        self._gimbal_error = self._gimbal_pos - self._action_gimbal_pos
        self._gimbal_smooth = self._last_actions[:, _gimbal_act] - 2 * \
            self._actions[:, _gimbal_act] + self._last_last_actions[:, _gimbal_act]
        gimbal_vibration_ids = self._gimbal_vel * self._gimbal_vel_last < 0.0
        self._gimbal_vibration = self._gimbal_vel * self._gimbal_vel_last * gimbal_vibration_ids.to(torch.float32) * -1.0
        self._gimbal_vel_last = self._gimbal_vel.clone()

        gimbal_limit = -(self._gimbal_pos - (-self.cfg.gimbal_limit_range)).clip(max=0.0)
        gimbal_limit += (self._gimbal_pos - (self.cfg.gimbal_limit_range)).clip(min=0.0)
        self._gimbal_limit_violation = gimbal_limit.clone()

        gimbal_acc_reward = torch.sum(torch.square(((self._gimbal_acc))), dim=1)
        gimbal_smooth_reward = torch.sum(torch.square(self._gimbal_smooth), dim=1)
        gimbal_veribration_reward = torch.sum(self._gimbal_vibration, dim=1)
        gimbal_limit_reward = torch.sum(gimbal_limit, dim=1)

        self._action_thrust_rate = (
            self._actions[:, -self.cfg.paired_rotor_num:] - self._last_actions[:, -self.cfg.paired_rotor_num:]
        ) * self.ctrlCfg.thrust_action_scale
        thrust_action_rate_reward = torch.sum(torch.square(self._action_thrust_rate), dim=1)
        self._thrust_power = torch.sum(torch.square(self._action_thrust_force), dim=1)
        thrust_power_reward = self._thrust_power
        self._thrust_limit_violation = (self._action_thrust_force - self.cfg.thrust_limit).clip(min=0.0, max=10.0)
        thrust_limit_reward = torch.sum(torch.square(self._thrust_limit_violation), dim=1)

        # # Joint / gimbal torque -------------------------------------------
        # For implicit actuators IsaacLab still computes the PD torque estimate:
        #   applied_torque  = clip(computed_torque, +/- effort_limit)  (actual torque -> power)
        #   computed_torque = kp*(q_target - q) + kd*(qd_target - qd)  (pre-clip -> limit penalty)
        self._joint_torque = self._robot.data.applied_torque[:, self._joint_ids[0]]
        self._gimbal_torque = self._robot.data.applied_torque[:, self._gimbal_ids[0]]
        _joint_computed_torque = self._robot.data.computed_torque[:, self._joint_ids[0]]
        _gimbal_computed_torque = self._robot.data.computed_torque[:, self._gimbal_ids[0]]

        # torque power: mechanical power |tau * qd| summed over DOFs (cf. humanoid_hsi joint_powers).
        # separate joint/gimbal scales handle the very different magnitudes between the two groups.
        self._joint_torque_power = torch.abs(self._joint_torque * self._joint_vel)
        joint_torque_power_reward = torch.sum(torch.abs(self._joint_torque * self._joint_vel), dim=1)
        gimbal_torque_power_reward = torch.sum(torch.abs(self._gimbal_torque * self._gimbal_vel), dim=1)

        # torque limit: soft-limit violation of the pre-clip torque (keeps gradient past saturation).
        self._joint_torque_limit_violation = (
            torch.abs(_joint_computed_torque) - self._joint_effort_limit * self.cfg.soft_joint_torque_limit
        ).clip(min=0.0)
        self._gimbal_torque_limit_violation = (
            torch.abs(_gimbal_computed_torque) - self._gimbal_effort_limit * self.cfg.soft_gimbal_torque_limit
        ).clip(min=0.0)
        joint_torque_limit_reward = torch.sum(self._joint_torque_limit_violation, dim=1)
        gimbal_torque_limit_reward = torch.sum(self._gimbal_torque_limit_violation, dim=1)

        remaining_steps = (self.max_episode_length - self.episode_length_buf).clamp(min=0).float()
        died_reward = self.reset_terminated.float() * remaining_steps

        rewards = {
            "lin_vel": lin_vel_reward * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel_reward * self.cfg.ang_vel_reward_scale * self.step_dt,
            "lin_acc": lin_acc_reward * self.cfg.lin_acc_reward_scale * self.step_dt,
            "ang_acc": ang_acc_reward * self.cfg.ang_acc_reward_scale * self.step_dt,
            "lin_vel_tracking": lin_vel_tracking_reward * self.cfg.body_lin_vel_tracking_reward_scale * self.step_dt,
            "ang_vel_tracking": ang_vel_tracking_reward * self.cfg.body_ang_vel_tracking_reward_scale * self.step_dt,
            "thrust_power": thrust_power_reward * self.cfg.thrust_power_reward_scale * self.step_dt,
            "contact": contact_reward * self.cfg.contact_reward_scale * self.step_dt,
            "joint_contact": joint_contact_reward * self.cfg.joint_contact_reward_scale * self.step_dt,
            "died": died_reward * self.cfg.died_reward_scale * self.step_dt,
            "position_tracking": position_tracking_reward * self.cfg.position_tracking_reward_scale * self.step_dt,
            "position_deviation": position_deviation_reward * self.cfg.position_deviation_reward_scale * self.step_dt,
            "orientation_tracking": orientation_tracking_reward * self.cfg.orientation_tracking_reward_scale * self.step_dt,
            "orientation_deviation": orientation_deviation_reward * self.cfg.orientation_deviation_reward_scale * self.step_dt,
            "joint_tracking": joint_tracking_reward * self.cfg.joint_tracking_reward_scale * self.step_dt,
            "joint_deviation": joint_deviation_reward * self.cfg.joint_deviation_reward_scale * self.step_dt,
            "joint_action_rate": joint_action_rate_reward * self.cfg.joint_action_rate_reward_scale * self.step_dt,
            "gimbal_action_rate": gimbal_action_rate_reward * self.cfg.gimbal_action_rate_reward_scale * self.step_dt,
            "thrust_action_rate": thrust_action_rate_reward * self.cfg.thrust_action_rate_reward_scale * self.step_dt,
            "joint_vel": joint_vel_reward * self.cfg.joint_vel_reward_scale * self.step_dt,
            "joint_acc": joint_acc_reward * self.cfg.joint_acc_reward_scale * self.step_dt,
            "joint_limit": joint_limit_reward * self.cfg.joint_limit_reward_scale * self.step_dt,
            "gimbal_acc": gimbal_acc_reward * self.cfg.gimbal_acc_reward_scale * self.step_dt,
            "gimbal_limit": gimbal_limit_reward * self.cfg.gimbal_limit_reward_scale * self.step_dt,
            "thrust_limit": thrust_limit_reward * self.cfg.thrust_limit_reward_scale * self.step_dt,
            "joint_torque_power": joint_torque_power_reward * self.cfg.joint_torque_power_reward_scale * self.step_dt,
            "gimbal_torque_power": gimbal_torque_power_reward * self.cfg.gimbal_torque_power_reward_scale * self.step_dt,
            "joint_torque_limit": joint_torque_limit_reward * self.cfg.joint_torque_limit_reward_scale * self.step_dt,
            "gimbal_torque_limit": gimbal_torque_limit_reward * self.cfg.gimbal_torque_limit_reward_scale * self.step_dt,
        }
        total_reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        # total_reward = torch.zeros(self.num_envs, device=self.device)
        # self._angular_to_goal_reward = total_reward.mean().item()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.evaluate_mode:
            time_out = self.episode_length_buf >= self.max_episode_length
        else:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        # # Angle reset logic
        # if self.ctrlCfg.ang_reset_rate > 0.0 and self.common_step_counter > self.cfg.ang_curricular_steps:
        #     reset_mask = (self.episode_length_buf > self._reset_ang_episode_counts) & ~self._reset_ang_pose_mask
        #     reset_ang_mask = reset_mask & (torch.rand(self.num_envs, device=self.device) < self.ctrlCfg.ang_reset_rate)
        #     reset_env_ids = torch.nonzero(reset_ang_mask, as_tuple=False).squeeze(-1)
        #     ang_range = self.ctrlCfg.body_ang * self._quat_sample_rate
        #     self._desired_quat_w[reset_env_ids] = sampleCenterQuatwithTilt(
        #         torch.tensor(ang_range), len(reset_env_ids)
        #     ).to(self.device)
        #     self._reset_ang_pose_mask |= reset_mask
        # import ipdb; ipdb.set_trace()
        # died = torch.linalg.norm(self._body_contact_sensor.data.force_matrix_w.squeeze(1).squeeze(1), dim=-1) > 0.1
        crash_rotor = contact_force_exceeded(
            self,
            self._rotor_sensor_cfg,
            threshold=self.cfg.contact_force_threshold,
        )
        crash_end = contact_force_exceeded(
            self,
            self._end_sensor_cfg,
            threshold=self.cfg.contact_force_threshold,
        )
        crash_body = contact_force_exceeded(
            self,
            self._body_sensor_cfg,
            threshold=self.cfg.contact_force_threshold,
        )
        body_vel_w_z = self._robot.data.body_com_lin_vel_w[:, self._body_id[0]][:, 2].abs()
        crash_body_vel = body_vel_w_z > self.cfg.contact_velocity_threshold
        crash_body = torch.logical_and(crash_body, crash_body_vel)
        # crash_body_env_ids = torch.nonzero(crash_body_vel, as_tuple=False).squeeze(-1)
        # if crash_body_env_ids.numel() > 0:
        #     print(
        #         "Debug crash body env_ids:",
        #         crash_body_env_ids.tolist(),
        #         "vel_w_z:",
        #         body_vel_w_z[crash_body_env_ids].detach().cpu().tolist(),
        #     )
        crash = torch.logical_or(crash_rotor, torch.logical_or(crash_end, crash_body))
        # crash = (
        #     torch.linalg.norm(self._body_contact_sensor.data.net_forces_w.squeeze(1), dim=-1)
        #     > self.cfg.contact_force_threshold
        # )
        # static root height: 0.1483
        drift = base_height_out_of_range(
            self,
            low=0.0,
            high=6.0,
        )
        # unsafe = base_orientation_out_of_range(
        #     self,
        #     max_tilt=self.ctrlCfg.max_tilt_angle,
        # )
        # reference-tracking early termination: states drifted too far from the demo target
        # (_desired_joint_pos_w holds the frame from the previous step's motion targets --
        # the same target the policy observed for this action -- so grading against it here is phase-consistent).
        joint_track_err = torch.linalg.norm(
            self._robot.data.joint_pos[:, self._joint_ids[0]]
            - self._joint_default_pos - self._desired_joint_pos_w,
            dim=1,
        )
        pos_err, rot_err = compute_pose_error(
            self._imu_sensor.data.pos_w,
            self._imu_sensor.data.quat_w,
            self._desired_pos_w,
            self._desired_quat_w,
        )
        position_track_err = torch.linalg.norm(pos_err, dim=1)
        orientation_track_err = torch.linalg.norm(rot_err, dim=1)
        # per-cause death masks, stored for the Episode_Termination/ breakdown logged in _reset_idx
        # (may overlap when several causes fire on the same step).
        self._died_position = position_track_err > self.cfg.position_track_termination_threshold
        self._died_orientation = orientation_track_err > self.cfg.orientation_track_termination_threshold
        self._died_joint = joint_track_err > self.cfg.joint_track_termination_threshold
        self._died_crash = crash
        self._died_drift = drift
        track_lost = torch.logical_or(
            self._died_joint,
            torch.logical_or(self._died_position, self._died_orientation),
        )
        # died = torch.logical_or(crash, torch.logical_or(drift, unsafe))
        died = torch.logical_or(crash, torch.logical_or(drift, track_lost))
        self._aars_log_early_termination_rows(
            died=died,
            time_out=time_out,
            position_track_err=position_track_err,
            orientation_track_err=orientation_track_err,
            joint_track_err=joint_track_err,
            crash=crash,
            drift=drift,
            track_lost=track_lost,
        )
        # died = torch.zeros_like(time_out, dtype=torch.bool)
        # print("Debug Done Info:")
        # print(" Crash: ", crash)
        # print(" Drift: ", drift)
        # print(" Unsafe: ", unsafe)
        # # # # DEBUG
        # import ipdb; ipdb.set_trace()
        # if self.cfg.evaluate_mode:
        #     died = torch.zeros_like(time_out, dtype=torch.bool)
        return died, time_out

    def _aars_hybrid_real_ref(self, frame_index: torch.Tensor, ref_target: dict) -> dict:
        """Nearest-feasible-frame RSI: pick the ``real`` clip frame closest to the target start state.

        The target start phase (``frame_index`` on the target clip) is mapped onto the real clip by the
        clip length ratio, shifted by ``hybrid_init_look_offset`` (global shift) and searched
        +/- ``hybrid_init_look_window`` frames. Per env the frame minimizing the weighted
        (position, orientation, joint) distance to the target start is selected; its full state
        (base pose/twist, joint, gimbal, thrust) is returned as a ref-like dict so the downstream reset
        code -- and the gimbal_init_from_demo / thrust_init_from_demo sub-switches -- consume it unchanged.
        """
        rm = self._real_motion
        w = self.cfg.hybrid_init_state_weight
        W = int(self.cfg.hybrid_init_look_window)
        m = frame_index.shape[0]
        # map target frame -> real frame by phase ratio, then apply the global look-ahead offset
        ratio = rm.max_index / max(self._motion.max_index, 1e-6)
        center_raw = frame_index * ratio                                   # (m,) mapped phase, no offset
        center = center_raw + float(self.cfg.hybrid_init_look_offset)
        offsets = torch.arange(-W, W + 1, device=self.device, dtype=center.dtype)
        cand = (center.unsqueeze(1) + offsets.unsqueeze(0)).clamp(0.0, float(rm.max_index))  # (m, n)
        n = cand.shape[1]
        real = rm.interpolate(cand.reshape(-1))
        cand_pos = real["base_pos_w"].reshape(m, n, 3)
        cand_joint = real["joint_pos"].reshape(m, n, self.cfg.joint_num)
        d_pos = torch.linalg.norm((cand_pos - ref_target["base_pos_w"].unsqueeze(1)), dim= -1)
        # orientation distance = geodesic (axis-angle) magnitude of the relative rotation [rad], the
        # rotation-space analogue of the position/joint L2 norms (all linear distances: m / rad). Same
        # convention as the env's orientation-deviation reward (compute_pose_error's rot_err magnitude).
        quat_err_angle = quat_error_magnitude(
            real["base_quat_w"], ref_target["base_quat_w"].unsqueeze(1).expand(m, n, 4).reshape(-1, 4)
        ).reshape(m, n)                                                    # geodesic angle [rad]
        d_quat = quat_err_angle
        d_joint = torch.linalg.norm((cand_joint - ref_target["joint_pos"].unsqueeze(1)), dim= -1)
        dist = w[0] * d_pos + w[1] * d_quat + w[2] * d_joint               # (m, n)
        best = torch.argmin(dist, dim=1)                                   # (m,)

        def pick(name: str, dim: int) -> torch.Tensor:
            t = real[name].reshape(m, n, dim)
            return t.gather(1, best.view(m, 1, 1).expand(m, 1, dim)).squeeze(1)

        out = {
            "base_pos_w": pick("base_pos_w", 3),
            "base_quat_w": pick("base_quat_w", 4),
            "base_lin_vel_w": pick("base_lin_vel_w", 3),
            "base_ang_vel_w": pick("base_ang_vel_w", 3),
            "joint_pos": pick("joint_pos", self.cfg.joint_num),
        }
        if "gimbal_pos" in real:
            out["gimbal_pos"] = pick("gimbal_pos", self.cfg.gimbal_num)
        if "rotor_thrust" in real:
            out["rotor_thrust"] = pick("rotor_thrust", self.cfg.rotor_num)

        # one-time diagnostic on the first full reset: how close the matched frame is (per-component
        # distance) and where it sits (mean real-frame index relative to the mapped phase), so the
        # look_offset / look_window can be judged from a single line.
        if not self._hybrid_init_logged and m == self.num_envs:
            self._hybrid_init_logged = True
            sel_frame = cand.gather(1, best.view(m, 1)).squeeze(1)
            rel_index = (sel_frame - center_raw).mean().item()             # +: match leads mapped phase
            edge_hit = ((best == 0) | (best == n - 1)).float().mean().item() * 100.0
            base_pos_init = d_pos.gather(1, best.view(m, 1)).squeeze(1).mean().item()
            base_quat_init = quat_err_angle.gather(1, best.view(m, 1)).squeeze(1).mean().item()  # [rad]
            joint_pos_init = d_joint.gather(1, best.view(m, 1)).squeeze(1).mean().item()
            print(
                f"[hybrid_init] first reset-ALL | base_pos_init={base_pos_init:.4f} m  "
                f"base_quat_init={base_quat_init:.4f} rad  joint_pos_init={joint_pos_init:.4f} rad  "
                f"| mean rel_index(vs mapped)={rel_index:+.2f}  edge_hit={edge_hit:.1f}%  (window=±{W}, offset={self.cfg.hybrid_init_look_offset})"
            )
        return out

    def _aars_early_termination_csv_columns(self) -> list[str]:
        return [
            "iter",
            "env_id",
            "trajectory_id",
            "trajectory_name",
            "ref_frame_index",
            "_push_episode_step",
            "episode_length_buf",
            "position_track_err",
            "orientation_track_err",
            "joint_track_err",
            "crash",
            "drift",
            "track_lost",
        ]

    def _aars_early_termination_csv_file(self) -> str:
        if self._early_termination_csv_path is None:
            filename = str(getattr(self.cfg, "early_termination_csv_filename", "early_termination_causes.csv"))
            log_dir = getattr(self.cfg, "log_dir", None)
            if log_dir is None:
                log_dir = os.path.abspath(os.path.join("logs", "dragon_track"))

            diagnostics_dir = os.path.join(os.path.abspath(log_dir), "diagnostics")
            rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")
            if rank not in (None, "", "0"):
                root, ext = os.path.splitext(filename)
                filename = f"{root}_rank{rank}{ext or '.csv'}"

            os.makedirs(diagnostics_dir, exist_ok=True)
            self._early_termination_csv_path = os.path.join(diagnostics_dir, filename)
        return self._early_termination_csv_path

    def _aars_log_early_termination_rows(
        self,
        died: torch.Tensor,
        time_out: torch.Tensor,
        position_track_err: torch.Tensor,
        orientation_track_err: torch.Tensor,
        joint_track_err: torch.Tensor,
        crash: torch.Tensor,
        drift: torch.Tensor,
        track_lost: torch.Tensor,
    ) -> None:
        if not bool(getattr(self.cfg, "early_termination_csv_enabled", True)):
            return

        early_terminated = died & ~time_out
        if not bool(torch.any(early_terminated).item()):
            return

        done = died | time_out
        mean_episode_length = torch.mean(self.episode_length_buf[done].float()).item()
        min_mean_episode_length = float(getattr(self.cfg, "early_termination_csv_min_mean_episode_length", 300.0))
        if mean_episode_length <= min_mean_episode_length:
            return
        if self.common_step_counter <= 1:
            return

        env_ids = torch.nonzero(early_terminated, as_tuple=False).squeeze(-1)
        target_virtual = self._motion_virtual[env_ids] + (
            self._motion_phase_step[env_ids]
            * self.cfg.track_frame_stride
            * (~self._static_hover_active[env_ids]).to(torch.long)
        )
        motion_ids = self._motion_ids[env_ids]
        _, ref_frame_index, _ = self._motion_library.query(motion_ids, target_virtual)

        path = self._aars_early_termination_csv_file()
        file_has_header = os.path.exists(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self._aars_early_termination_csv_columns())
            if not file_has_header:
                writer.writeheader()
            for row_id, env_id in enumerate(env_ids):
                motion_id = int(motion_ids[row_id].item())
                writer.writerow(
                    {
                        "iter": self.common_step_counter,
                        "env_id": int(env_id.item()),
                        "trajectory_id": motion_id,
                        "trajectory_name": self._motion_library.motion_names[motion_id],
                        "ref_frame_index": float(ref_frame_index[row_id].item()),
                        "_push_episode_step": int(self._push_episode_step[env_id].item()),
                        "episode_length_buf": int(self.episode_length_buf[env_id].item()),
                        "position_track_err": float(position_track_err[env_id].item()),
                        "orientation_track_err": float(orientation_track_err[env_id].item()),
                        "joint_track_err": float(joint_track_err[env_id].item()),
                        "crash": int(crash[env_id].item()),
                        "drift": int(drift[env_id].item()),
                        "track_lost": int(track_lost[env_id].item()),
                    }
                )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        # # Curriculum logic #####################################################################
        progress = self._aars_training_progress()
        multi_trajectory_stage_active = self._aars_multi_trajectory_stage_active(progress)
        if multi_trajectory_stage_active and not self._multi_trajectory_stage_logged:
            self._multi_trajectory_stage_logged = True
            weights = ", ".join(
                f"{name}={weight:.4f}"
                for name, weight in zip(
                    self._motion_library.motion_names,
                    self._motion_library.sampling_weights.tolist(),
                )
            )
            print(f"[multi_trajectory_stage] active at progress={progress} | {weights}")

        # # # # Initial Velocity Curriculum =========================
        if self.cfg.vel_random_mode:
            if progress < self.cfg.vel_random_step_start:
                self._vel_random_stage = 0.0
            elif progress < self.cfg.vel_random_step_end:
                self._vel_random_stage = (progress - self.cfg.vel_random_step_start) / (
                    self.cfg.vel_random_step_end - self.cfg.vel_random_step_start
                )
            else:
                self._vel_random_stage = 1.0

        # # # # Initial Gimbal Position Curriculum =================
        if self.cfg.gimbal_random_mode:
            if progress < self.cfg.gimbal_random_step_start:
                self._gimbal_random_stage = 0.0
            elif progress < self.cfg.gimbal_random_step_end:
                self._gimbal_random_stage = (progress - self.cfg.gimbal_random_step_start) / (
                    self.cfg.gimbal_random_step_end - self.cfg.gimbal_random_step_start
                )
            else:
                self._gimbal_random_stage = 1.0

        # # # # Initial Joint Position Curriculum ================
        if self.cfg.joint_random_mode:
            if progress < self.cfg.joint_random_step_start:
                self._joint_random_stage = 0.0
            elif progress < self.cfg.joint_random_step_end:
                self._joint_random_stage = (progress - self.cfg.joint_random_step_start) / (
                    self.cfg.joint_random_step_end - self.cfg.joint_random_step_start
                )
            else:
                self._joint_random_stage = 1.0

        # # # # Initial Position Sampling Curriculum =============
        if self.cfg.pos_mode:
            if progress < self.cfg.pos_sample_step_start:
                self._pos_sample_stage = 0.0
            elif progress < self.cfg.pos_sample_step_end:
                self._pos_sample_stage = (progress - self.cfg.pos_sample_step_start) / (
                    self.cfg.pos_sample_step_end - self.cfg.pos_sample_step_start
                )
            else:
                self._pos_sample_stage = 1.0

        # # # # Initial Orientation Sampling Curriculum ==========
        if self.cfg.quat_mode:
            if progress < self.cfg.quat_sample_step_start:
                self._quat_sample_stage = 0.0
            elif progress < self.cfg.quat_sample_step_end:
                self._quat_sample_stage = (progress - self.cfg.quat_sample_step_start) / (
                    self.cfg.quat_sample_step_end - self.cfg.quat_sample_step_start
                )
            else:
                self._quat_sample_stage = 1.0

        # # # # Push Curriculum ===================================
        if self.cfg.push_mode:
            self._push_done_ids[env_ids] = False
            self._push_impulse_steps_left[env_ids] = 0
            self._push_impulse_force_b[env_ids] = 0.0
            self._push_episode_step[env_ids] = torch.randint(
                int(self.cfg.push_episode_min),
                int(self.cfg.push_episode_max) + 1,
                (len(env_ids),),
                device=self.device,
                dtype=self._push_episode_step.dtype
            )
            if progress < self.cfg.push_step_start:
                self._push_stage = 0.0
            elif progress < self.cfg.push_step_end:
                self._push_stage = (progress - self.cfg.push_step_start) / (
                    self.cfg.push_step_end - self.cfg.push_step_start
                )
            else:
                self._push_stage = 1.0

        # # # # Static-Hover Curriculum =========================
        if self.cfg.static_hover_mode:
            if progress < self.cfg.static_hover_step_start:
                self._static_hover_stage = 0.0
            elif progress < self.cfg.static_hover_step_end:
                self._static_hover_stage = (progress - self.cfg.static_hover_step_start) / (
                    self.cfg.static_hover_step_end - self.cfg.static_hover_step_start
                )
            else:
                self._static_hover_stage = 1.0

        if self.cfg.play_mode:
            self._vel_random_stage = 1.0
            self._gimbal_random_stage = 1.0
            self._joint_random_stage = 1.0
            self._pos_sample_stage = 1.0
            self._quat_sample_stage = 1.0
            self._push_stage = 1.0
            self._static_hover_stage = 1.0  # rate_end (mostly full tracking) in play mode
        elif self.cfg.evaluate_mode:
            # Evaluation always resets to the exact first-frame state: no random RSI phase or
            # position/orientation/velocity/joint/gimbal perturbation, no push, no static-hover mix.
            self._vel_random_stage = 0.0
            self._gimbal_random_stage = 0.0
            self._joint_random_stage = 0.0
            self._pos_sample_stage = 0.0
            self._quat_sample_stage = 0.0
            self._push_stage = 0.0
            self._static_hover_stage = 1.0

        # ### Logging before reset #####################################################################

        lin_vel = torch.mean(self._lin_vel)
        ang_vel = torch.mean(self._ang_vel)
        lin_acc = torch.mean(self._lin_acc)
        ang_acc = torch.mean(self._ang_acc)
        distance_to_goal = torch.mean(self._distance_to_goal)
        angular_to_goal = torch.mean(self._angular_to_goal)
        joint_pos_to_goal = torch.mean(self._joint_pos_to_goal)
        contact = torch.mean(self._contact)
        joint_contact = torch.mean(self._joint_contact)

        action_thrust_rate = torch.mean(torch.sum(torch.abs(self._action_thrust_rate), dim=1))
        thrust_average = torch.mean(self._action_thrust_force, dim=1)
        thrust_average = torch.mean(thrust_average)
        thrust_limit_violation = torch.mean(self._thrust_limit_violation)

        action_joint_rate = torch.mean(torch.sum(torch.abs(self._action_joint_rate), dim=1))
        joint_vel = torch.mean(torch.sum(torch.square(self._joint_vel), dim=1))
        joint_acc = torch.mean(torch.sum(torch.square(self._joint_acc), dim=1))
        joint_limit_violation = torch.mean(self._joint_limit_violation)
        joint_error = torch.mean(torch.sum(torch.abs(self._joint_error), dim=1))
        joint_smooth = torch.mean(torch.sum(torch.abs(self._joint_smooth), dim=1))
        joint_vibration = torch.mean(torch.sum(self._joint_vibration, dim=1))
        joint_torque = torch.mean(torch.sum(torch.abs(self._joint_torque), dim=1))
        joint_torque_power = torch.mean(torch.sum(torch.abs(self._joint_torque_power), dim=1))
        joint_torque_limit_violation = torch.mean(torch.sum(self._joint_torque_limit_violation, dim=1))

        action_gimbal_rate = torch.mean(torch.sum(torch.abs(self._action_gimbal_rate), dim=1))
        gimbal_acc = torch.mean(torch.sum(torch.abs(self._gimbal_acc), dim=1))
        gimbal_vel = torch.mean(torch.sum(torch.abs(self._gimbal_vel), dim=1))
        gimbal_limit_violation = torch.mean(self._gimbal_limit_violation)
        gimbal_error = torch.mean(torch.sum(torch.abs(self._gimbal_error), dim=1))
        gimbal_smooth = torch.mean(torch.sum(torch.abs(self._gimbal_smooth), dim=1))
        gimbal_vibration = torch.mean(torch.sum(self._gimbal_vibration, dim=1))
        gimbal_torque = torch.mean(torch.sum(torch.abs(self._gimbal_torque), dim=1))
        gimbal_torque_limit_violation = torch.mean(torch.sum(self._gimbal_torque_limit_violation, dim=1))

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        # per-cause breakdown of the deaths (masks set in _get_dones; may overlap across causes)
        extras["Episode_Termination/died_position"] = torch.count_nonzero(self._died_position[env_ids]).item()
        extras["Episode_Termination/died_orientation"] = torch.count_nonzero(self._died_orientation[env_ids]).item()
        extras["Episode_Termination/died_joint"] = torch.count_nonzero(self._died_joint[env_ids]).item()
        extras["Episode_Termination/died_crash"] = torch.count_nonzero(self._died_crash[env_ids]).item()
        extras["Episode_Termination/died_drift"] = torch.count_nonzero(self._died_drift[env_ids]).item()

        extras["Metrics/lin_vel"] = lin_vel.item()
        extras["Metrics/ang_vel"] = ang_vel.item()
        extras["Metrics/lin_acc"] = lin_acc.item()
        extras["Metrics/ang_acc"] = ang_acc.item()
        extras["Metrics/distance_to_goal"] = distance_to_goal.item()
        extras["Metrics/angular_to_goal"] = angular_to_goal.item()
        extras["Metrics/joint_pos_to_goal"] = joint_pos_to_goal.item()
        extras["Metrics/contact"] = contact.item()
        extras["Metrics/joint_contact"] = joint_contact.item()

        extras["Metrics/action_thrust_rate"] = action_thrust_rate.item()
        extras["Metrics/thrusts_average"] = thrust_average.item()
        extras["Metrics/thrust_limit_violation"] = thrust_limit_violation.item()
        extras["Metrics/action_joint_rate"] = action_joint_rate.item()
        extras["Metrics/joint_vel"] = joint_vel.item()
        extras["Metrics/joint_acc"] = joint_acc.item()
        extras["Metrics/joint_limit_violation"] = joint_limit_violation.item()
        extras["Metrics/joint_smooth"] = joint_smooth.item()
        extras["Metrics/joint_vibration"] = joint_vibration.item()
        extras["Metrics/joint_error"] = joint_error.item()
        extras["Metrics/joint_smooth"] = joint_smooth.item()
        extras["Metrics/joint_vibration"] = joint_vibration.item()
        extras["Metrics/joint_torque"] = joint_torque.item()
        extras["Metrics/joint_torque_limit_violation"] = joint_torque_limit_violation.item()
        extras["Metrics/joint_power"] = joint_torque_power.item()
        extras["Metrics/action_gimbal_rate"] = action_gimbal_rate.item()
        extras["Metrics/gimbal_acc"] = gimbal_acc.item()
        extras["Metrics/gimbal_vel"] = gimbal_vel.item()
        extras["Metrics/gimbal_limit_violation"] = gimbal_limit_violation.item()
        extras["Metrics/gimbal_error"] = gimbal_error.item()
        extras["Metrics/gimbal_smooth"] = gimbal_smooth.item()
        extras["Metrics/gimbal_vibration"] = gimbal_vibration.item()
        extras["Metrics/gimbal_torque"] = gimbal_torque.item()
        extras["Metrics/gimbal_torque_limit_violation"] = gimbal_torque_limit_violation.item()

        extras["Stage/quat_sample"] = self._quat_sample_stage * self.randomCfg.rsi_ang_dev
        extras["Stage/pos_sample"] = self._pos_sample_stage * self.randomCfg.rsi_pos_dev
        extras["Stage/vel_lin_random"] = self._vel_random_stage * self.randomCfg.rsi_lin_vel_dev
        extras["Stage/vel_ang_random"] = self._vel_random_stage * self.randomCfg.rsi_ang_vel_dev
        extras["Stage/push"] = self._push_stage * self.cfg.push_lin_vel_range
        extras["Stage/gimbal_random"] = self._gimbal_random_stage * self.randomCfg.rsi_gimbal_dev
        extras["Stage/joint_random"] = self._joint_random_stage * self.randomCfg.rsi_joint_dev
        extras["Stage/static_hover"] = self._static_hover_stage if self.cfg.static_hover_mode else 0.0
        extras["Stage/static_hover_rate"] = self._aars_get_static_hover_rate()
        extras["Stage/static_hover_active"] = torch.count_nonzero(self._static_hover_active).item()
        self.extras["log"].update(extras)

        # # Reset Modules ###########################################################
        self._robot.reset(env_ids)
        self._imu_sensor.reset(env_ids)
        self._body_contact_sensor.reset(env_ids)
        self._foot_contact_sensor.reset(env_ids)
        self._end_contact_sensor.reset(env_ids)
        self._thrust_sensor.reset(env_ids)
        self._joint_contact_sensor.reset(env_ids)
        self._rotors.reset(env_ids)
        if self.cfg.decrease_rotor_thrust:
            # override RotorGroup's per-rotor sample so both rotors of a pair share one scale
            scale_lower, scale_upper = self.cfg.rotorCfg["thrust_scale"]
            pair_scale = torch.empty(len(env_ids), self.cfg.paired_rotor_num, device=self.device).uniform_(
                scale_lower, scale_upper
            )
            self._rotors.thrust_scale[env_ids] = self._aars_expand_paired_rotor_actions(pair_scale)
        self.noiseModel.reset(env_ids)

        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs and not self.cfg.evaluate_mode:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # # Reset States ###########################################################
        # # # # Reference State Initialization (RSI) ===========
        # Evaluation starts from the first demo frame. Training/play sample a random virtual phase in
        # [0, 2*max_index), which also encodes forward/backward playback via the triangle wave.
        m = len(env_ids)
        rcfg = self.randomCfg
        if multi_trajectory_stage_active:
            selected_motion_ids = self._motion_library.sample_motion_ids(m)
        else:
            selected_motion_ids = torch.full(
                (m,), self._stage1_motion_id, device=self.device, dtype=torch.long
            )
        self._motion_ids[env_ids] = selected_motion_ids
        self._motion_phase_step[env_ids] = self._motion_library.phase_steps(selected_motion_ids)

        if self.cfg.evaluate_mode:
            self._motion_virtual[env_ids] = torch.zeros(m, device=self.device, dtype=torch.long)
        else:
            self._motion_virtual[env_ids] = self._motion_library.sample_reset_phases(selected_motion_ids)
        ref, frame_index, play_sign = self._motion_library.query(
            selected_motion_ids, self._motion_virtual[env_ids]
        )
        # Hybrid RSI is defined only for the Stage-1 target, which has a paired dynamically-feasible
        # real clip. Keep using that paired reset source whenever an env selects the Stage-1 motion,
        # including after the multi-trajectory stage starts. Other motions retain their target state.
        stage1_local_ids = torch.nonzero(
            selected_motion_ids == self._stage1_motion_id, as_tuple=False
        ).squeeze(-1)
        stage1_init_ref = None
        use_hybrid_initialization = (
            stage1_local_ids.numel() > 0
            and self.cfg.hybrid_initialization
            and self._real_motion is not None
            and not self.cfg.evaluate_mode
        )
        if use_hybrid_initialization:
            stage1_target_ref = {name: value[stage1_local_ids] for name, value in ref.items()}
            stage1_init_ref = self._aars_hybrid_real_ref(
                frame_index[stage1_local_ids], stage1_target_ref
            )
            for name in ref:
                ref[name][stage1_local_ids] = stage1_init_ref[name]
        elif stage1_local_ids.numel() > 0 and (
            self.cfg.gimbal_init_from_demo or self.cfg.thrust_init_from_demo
        ):
            # Optional gimbal/thrust fields exist only on the primary Stage 1 clip. The common
            # target fields come from the packed library; these reset-only fields use the same
            # discrete frame indices without interpolation.
            stage1_init_ref = self._motion.lookup(frame_index[stage1_local_ids])
        play_sign = play_sign.unsqueeze(-1)

        # base pose: demo frame (+ env origin) with optional RSI deviation
        ref_pos_w = ref["base_pos_w"] + self._terrain.env_origins[env_ids]
        if not self.cfg.evaluate_mode:
            ref_pos_w = ref_pos_w + (
                torch.empty(m, 3, device=self.device).uniform_(-1.0, 1.0)
                * rcfg.rsi_pos_dev
                * self._pos_sample_stage
            )
        ref_quat_w = ref["base_quat_w"]
        if not self.cfg.evaluate_mode:
            ang_axis = torch.randn(m, 3, device=self.device)
            ang_axis = ang_axis / ang_axis.norm(dim=-1, keepdim=True)
            ang_range = (
                torch.empty(m, 1, device=self.device).uniform_(-1.0, 1.0)
                * rcfg.rsi_ang_dev
                * self._quat_sample_stage
            )
            delta_quat = quat_from_angle_axis(ang_range.squeeze(1), ang_axis)
            ref_quat_w = quat_mul(delta_quat, ref_quat_w)

        # base twist: demo velocity (reversed when playing backward) with optional RSI deviation
        ref_lin_vel_w = ref["base_lin_vel_w"] * play_sign
        ref_ang_vel_w = ref["base_ang_vel_w"] * play_sign
        if not self.cfg.evaluate_mode and self.cfg.vel_random_mode and self._vel_random_stage > 0.0:
            ref_lin_vel_w = ref_lin_vel_w + (
                torch.empty(m, 3, device=self.device).uniform_(-1.0, 1.0)
                * rcfg.rsi_lin_vel_dev * self._vel_random_stage
            )
            ref_ang_vel_w = ref_ang_vel_w + (
                torch.empty(m, 3, device=self.device).uniform_(-1.0, 1.0)
                * rcfg.rsi_ang_vel_dev * self._vel_random_stage
            )

        init_root_state = torch.cat((ref_pos_w, ref_quat_w, ref_lin_vel_w, ref_ang_vel_w), dim=-1)

        # # # # Static-Hover Mode ==============================
        # Pick a fraction of the resetting envs to become static-hover: their reference is frozen at
        # this RSI frame (phase + look-ahead handled in _aars_set_motion_targets) and they start at rest.
        self._static_hover_active[env_ids] = False
        if not self.cfg.evaluate_mode and self.cfg.static_hover_mode:
            num_to_hover = int(m * self._aars_get_static_hover_rate())
            if num_to_hover > 0:
                perm = torch.randperm(m, device=self.device)[:num_to_hover]
                self._static_hover_active[env_ids[perm]] = True
                init_root_state[perm, 7:13] = 0.0  # zero linear & angular velocity: a rest setpoint

        # joints: demo joint positions with optional RSI deviation (gimbals handled analytically below)
        init_dof_vel = self._robot.data.default_joint_vel[env_ids]
        init_dof_pos = self._robot.data.default_joint_pos[env_ids]
        init_joint_pos = ref["joint_pos"].clone()
        if not self.cfg.evaluate_mode and self.cfg.joint_random_mode and self._joint_random_stage > 0.0:
            init_joint_pos = init_joint_pos + (
                torch.empty_like(init_joint_pos).uniform_(-1.0, 1.0)
                * rcfg.rsi_joint_dev * self._joint_random_stage
            )
        torch.clamp_(init_joint_pos, self._joint_lower_pos[env_ids], self._joint_upper_pos[env_ids])
        init_joint_vel = init_dof_vel[:, self._joint_ids[0]]
        init_gimbal_pos = init_dof_pos[:, self._gimbal_ids[0]]
        init_gimbal_vel = init_dof_vel[:, self._gimbal_ids[0]]
        # Initial rotor thrust defaults to hover. Envs using the Stage-1 motion retain the paired
        # demo thrust even during the multi-trajectory stage.
        init_thrust_force = torch.ones_like(self._action_thrust_force[env_ids]) * self.ctrlCfg.thrust_hover
        if (
            stage1_init_ref is not None
            and self.cfg.thrust_init_from_demo
            and "rotor_thrust" in stage1_init_ref
        ):
            init_thrust_force[stage1_local_ids] = stage1_init_ref["rotor_thrust"]

        # # # # Write to sim ###########################################################
        self._robot.write_root_pose_to_sim(init_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(init_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel, self._joint_ids[0], env_ids)

        # Initial gimbals default to the analytical hover-ready pose. Envs using the Stage-1 motion
        # retain the paired demo gimbals even during the multi-trajectory stage.
        demo_gimbal_available = (
            stage1_init_ref is not None
            and self.cfg.gimbal_init_from_demo
            and "gimbal_pos" in stage1_init_ref
        )
        analytical_gimbal_mask = torch.ones(m, dtype=torch.bool, device=self.device)
        if demo_gimbal_available:
            init_gimbal_pos[stage1_local_ids] = stage1_init_ref["gimbal_pos"]
            analytical_gimbal_mask[stage1_local_ids] = False
        analytical_local_ids = torch.nonzero(analytical_gimbal_mask, as_tuple=False).squeeze(-1)
        if analytical_local_ids.numel() > 0:
            # Analytical fallback: orient each rotor so its thrust points up (hover-ready).
            analytical_env_ids = env_ids[analytical_local_ids]
            _ = self._robot.data.body_state_w[analytical_env_ids, :, :]  # reset env kinematics
            links_quats = self._robot.data.body_link_quat_w[analytical_env_ids][:, self._link_ids[0]]
            gimbal_roll, gimbal_pitch = self._aars_compute_analytical_gimbal_pos(links_quats)
            analytical_gimbal_pos = init_gimbal_pos[analytical_local_ids].clone()
            analytical_gimbal_pos[:, 0] = gimbal_roll[:, 0]
            analytical_gimbal_pos[:, 1] = gimbal_pitch[:, 0]
            analytical_gimbal_pos[:, 2] = gimbal_roll[:, 1]
            analytical_gimbal_pos[:, 3] = gimbal_roll[:, 2]
            analytical_gimbal_pos[:, 4] = gimbal_pitch[:, 1]
            analytical_gimbal_pos[:, 5] = gimbal_pitch[:, 2]
            analytical_gimbal_pos[:, 6] = gimbal_roll[:, 3]
            analytical_gimbal_pos[:, 7] = gimbal_pitch[:, 3]
            init_gimbal_pos[analytical_local_ids] = analytical_gimbal_pos
        if not self.cfg.evaluate_mode and self.cfg.gimbal_random_mode and self._gimbal_random_stage > 0.0:
            init_gimbal_pos += (
                torch.empty_like(init_gimbal_pos).uniform_(-1, 1)
                * self.randomCfg.rsi_gimbal_dev
                * self._gimbal_random_stage
            )
            torch.clamp_(init_gimbal_pos, self._gimbal_lower_pos[env_ids], self._gimbal_upper_pos[env_ids])
        self._robot.write_joint_state_to_sim(init_gimbal_pos, init_gimbal_vel, self._gimbal_ids[0], env_ids)

        self._action_joint_pos[env_ids] = init_joint_pos.clone()
        self._action_gimbal_pos[env_ids] = init_gimbal_pos.clone()
        self._action_thrust_force[env_ids] = init_thrust_force.clone()

        self._actions[env_ids, : self.cfg.joint_num] = (
            self._action_joint_pos[env_ids] - self._joint_default_pos[env_ids]
        ) / self.ctrlCfg.joint_action_scale
        action_gimbals = self._action_gimbal_pos[env_ids] - self._gimbal_default_pos[env_ids]
        action_gimbals[:, self.ctrlCfg.gimbal_pitch_ids] = action_gimbals[:, self.ctrlCfg.gimbal_pitch_ids] / self.ctrlCfg.gimbal_pitch_action_scale
        action_gimbals[:, self.ctrlCfg.gimbal_roll_ids] = action_gimbals[:, self.ctrlCfg.gimbal_roll_ids] / self.ctrlCfg.gimbal_roll_action_scale
        self._actions[env_ids, self.cfg.joint_num: self.cfg.joint_num + self.cfg.gimbal_num] = action_gimbals
        rotor_action_start = self.cfg.joint_num + self.cfg.gimbal_num
        paired_rotor_thrust = self._aars_compress_paired_rotor_actions(self._action_thrust_force[env_ids])
        self._actions[env_ids, rotor_action_start:] = (
            paired_rotor_thrust - self.ctrlCfg.thrust_hover
        ) / self.ctrlCfg.thrust_action_scale
        self._actions_scaled[env_ids] = self._aars_scale_actions_for_observation(
            self._actions[env_ids]
        )
        self._last_actions[env_ids] = self._actions[env_ids]
        self._last_last_actions[env_ids] = self._actions[env_ids]
        self._last_action_joint_pos[env_ids] = self._action_joint_pos[env_ids]
        self._last_action_gimbal_pos[env_ids] = self._action_gimbal_pos[env_ids]
        self._last_action_thrust_force[env_ids] = self._action_thrust_force[env_ids]

        self._action_joint_rate[env_ids] = 0.0
        self._action_gimbal_rate[env_ids] = 0.0
        self._action_thrust_rate[env_ids] = 0.0
        # # # # Reset Gimbal State =============================
        # init_gimbal_pos += torch.empty_like(init_gimbal_pos).uniform_(-1, 1) \
        #     * self.randomCfg.rsi_gimbal_dev * self._gimbal_random_stage
        # torch.clamp_(init_gimbal_pos, self._gimbal_lower_pos, self._gimbal_upper_pos)

        init_joint_pos_rel = init_joint_pos - self._joint_default_pos[env_ids]
        init_gimbal_pos_rel = init_gimbal_pos - self._gimbal_default_pos[env_ids]
        self._joint_pos[env_ids] = init_joint_pos_rel
        self._joint_pos_last[env_ids] = init_joint_pos_rel.clone()
        self._joint_vel[env_ids] = init_joint_vel
        self._joint_vel_last[env_ids] = init_joint_vel
        self._joint_acc[env_ids] = 0.0
        self._joint_error[env_ids] = 0.0
        self._joint_smooth[env_ids] = 0.0
        self._joint_vibration[env_ids] = 0.0
        self._joint_limit_violation[env_ids] = 0.0

        self._gimbal_pos[env_ids] = init_gimbal_pos_rel
        self._gimbal_pos_last[env_ids] = init_gimbal_pos_rel.clone()
        self._gimbal_vel[env_ids] = init_gimbal_vel
        self._gimbal_vel_last[env_ids] = init_gimbal_vel
        self._gimbal_acc[env_ids] = 0.0
        self._gimbal_error[env_ids] = 0.0
        self._gimbal_smooth[env_ids] = 0.0
        self._gimbal_vibration[env_ids] = 0.0
        self._gimbal_limit_violation[env_ids] = 0.0

        self._target_thrust_force[env_ids] = 0.0
        self._target_rotor_torque[env_ids] = 0.0
        self._thrust_power[env_ids] = 0.0
        self._thrust_limit_violation[env_ids] = 0.0

        # # # # Reset Initial Velocity =========================
        # init_root_state[:, 7:10] = (
        #     torch.empty_like(init_root_state[:, 7:10]).uniform_(-1, 1) * self.randomCfg.lin_vel
        # )
        # # Angular velocity
        # init_root_state[:, 10:13] = (
        #     torch.empty_like(init_root_state[:, 10:13]).uniform_(-1, 1) * self.randomCfg.ang_vel
        # )
        # # # # Reset reference targets ========================
        # _desired_* and the look-ahead target buffers are populated from the demo clip at the
        # freshly sampled phase (see end of this method, after the proprio buffers are reset).

        # # # # Reset Observations ####################################################

        # roll, pitch, yaw = euler_xyz_from_quat(self._desired_quat_w[env_ids])
        # self._desired_zyx_euler_w[env_ids] = torch.stack([roll, pitch, yaw], dim=-1)
        # self._init_pos_w[env_ids] = init_root_state[:, :3]
        # self._init_quat_w[env_ids] = init_root_state[:, 3:7]
        # roll, pitch, yaw = euler_xyz_from_quat(self._init_quat_w[env_ids])
        # self._init_zyx_euler_w[env_ids] = torch.stack([roll, pitch, yaw], dim=-1)
        self._position_error[env_ids] = 0.0
        self._angle_error[env_ids] = 0.0

        # self._obsPosDelayModel.reset(env_ids, default_root_state[:, :3])
        # self._obsQuatDelayModel.reset(env_ids, init_root_state[:, 3:7])
        # self._obsAngVelDelayModel.reset(env_ids)
        # self._obsLinVelDelayModel.reset(env_ids)
        # self._obsGimbalPosDelayModel.reset(env_ids)
        reset_quat = init_root_state[:, 3:7]
        reset_lin_vel = init_root_state[:, 7:10]
        reset_ang_vel = init_root_state[:, 10:13]
        reset_lin_vel_b = quat_apply_inverse(reset_quat, reset_lin_vel)
        reset_ang_vel_b = quat_apply_inverse(reset_quat, reset_ang_vel)
        self._obsLinVelDelayModel.reset(env_ids, reset_lin_vel_b)
        self._obsAngVelDelayModel.reset(env_ids, reset_ang_vel_b)
        # root link is the IMU link (fc), so the root state is the estimator's reset state
        self._stateEstimator.reset(
            env_ids, init_root_state[:, :3], reset_quat, init_root_state[:, 7:10], init_root_state[:, 10:13]
        )
        self._est_lin_vel_b[env_ids] = reset_lin_vel_b
        self._est_ang_vel_b[env_ids] = reset_ang_vel_b
        # obs joint/gimbal are observed relative to the default pose (matches _get_observations)
        self._obsJointPosDelayModel.reset(env_ids, init_joint_pos_rel)
        self._obsJointVelDelayModel.reset(env_ids, init_joint_vel)
        self._obsGimbalPosDelayModel.reset(env_ids, init_gimbal_pos_rel)
        self._obsGimbalVelDelayModel.reset(env_ids, init_gimbal_vel)
        # action delay models track absolute joint/gimbal targets
        self._actJointDelayModel.reset(env_ids, init_joint_pos)
        self._actGimbalDelayModel.reset(env_ids, init_gimbal_pos)
        self._actThrustDelayModel.reset(env_ids, init_thrust_force)

        # populate _desired_* and the world-frame look-ahead target buffers for the new phase
        self._aars_set_motion_targets(env_ids)

        # Seed history from the clean reset frame, with no prior-frame position delta.
        if self.cfg.replace_dof_vel_with_last_pos:
            reset_joint_dof_obs = torch.zeros_like(init_joint_pos)
            reset_gimbal_dof_obs = torch.zeros_like(init_gimbal_pos)
        else:
            reset_joint_dof_obs = init_joint_vel * self.obsScales.dof_vel
            reset_gimbal_dof_obs = init_gimbal_vel * self.obsScales.dof_vel

        reset_root_rot_vec = matrix_from_quat(reset_quat)[:, :2, :].reshape(len(env_ids), 6)
        reset_current_obs = torch.cat(
            (
                init_joint_pos_rel,
                reset_joint_dof_obs,
                init_gimbal_pos_rel,
                reset_gimbal_dof_obs,
                reset_root_rot_vec,
                reset_lin_vel_b * self.obsScales.lin_vel,
                reset_ang_vel_b * self.obsScales.ang_vel,
                self._actions_scaled[env_ids],
            ),
            dim=-1,
        )
        reset_current_obs = torch.clamp(
            reset_current_obs, -self.ctrlCfg.clip_observations, self.ctrlCfg.clip_observations
        )
        self._observation_buffers[env_ids] = reset_current_obs.unsqueeze(1).expand(
            -1, self.cfg.obs_history_len, -1
        )

        # print(f"Debug Envent {env_ids}  Reset: ==========================================================")
        # robot_mass = self._robot.root_physx_view.get_masses()[env_ids].sum()
        # print(f"Robot [{env_ids}] mass: {robot_mass:.4f}, default mass: {self._robot_mass:.4f}")
        # root_com = self._robot.data.body_com_pose_w[env_ids, self._body_id].clone().cpu().numpy()
        # root_com_str = ", ".join(f"{x:.4f}" for x in root_com.flatten())
        # default_com_str = ", ".join(f"{x:.4f}" for x in self._robot_default_com.flatten())
        # print(f"Robot [{env_ids}] root com: [{root_com_str}], default com: [{default_com_str}]")

    def _refresh_debug_vis(self) -> None:
        """Refresh the shared debug visualization subscription based on per-group visibility flags."""
        debug_vis = (
            getattr(self, "_vis_targets", False)
            or getattr(self, "_vis_thrusts", False)
            or getattr(self, "_vis_track_ends", False)
            or getattr(self, "_vis_actual_ends", False)
        )
        self.set_debug_vis(debug_vis)

    def set_targets_vis(self, debug_vis: bool) -> None:
        self._vis_targets = bool(debug_vis)
        self._refresh_debug_vis()

    def set_thrusts_vis(self, debug_vis: bool) -> None:
        self._vis_thrusts = bool(debug_vis)
        self._refresh_debug_vis()

    def set_track_ends_vis(self, debug_vis: bool) -> None:
        self._vis_track_ends = bool(debug_vis)
        self._refresh_debug_vis()

    def set_actual_ends_vis(self, debug_vis: bool) -> None:
        self._vis_actual_ends = bool(debug_vis)
        self._refresh_debug_vis()

    def _setup_end_fk(self) -> None:
        """Measure the fixed FK offsets needed to place the target front_end / back_end links.

        The two end links are separate articulation bodies (front_end rigidly fixed to link1,
        back_end rigidly fixed to link4) reached from the tracked base frame (fc) through the
        three pitch/yaw universal joints. All fixed joint origins are captured once from the
        current articulation body poses; the measurement-time joint angles are removed so the
        offsets are independent of the configuration the robot happens to be in when this runs.
        Called lazily on first enable, so it costs nothing while the visualization is off.
        """
        def _bid(name: str) -> int:
            return self._robot.find_bodies(name)[0][0]

        # body-frame world poses (env 0 suffices -- the fixed offsets are identical across envs)
        pos_w = self._robot.data.body_pos_w[0]
        quat_w = self._robot.data.body_quat_w[0]

        def _rel(parent: str, child: str):
            """pose of `child` body expressed in `parent` body frame (measured, current config)."""
            p, q = subtract_frame_transforms(
                pos_w[_bid(parent)].unsqueeze(0), quat_w[_bid(parent)].unsqueeze(0),
                pos_w[_bid(child)].unsqueeze(0), quat_w[_bid(child)].unsqueeze(0),
            )
            return p, q  # (1,3), (1,4)

        # measurement-time joint angles, to strip the joint rotation out of a revolute origin
        def _q0(joint_name: str) -> torch.Tensor:
            jid = self._robot.find_joints(joint_name)[0][0]
            return self._robot.data.joint_pos[0, jid].reshape(1)  # (1,)

        y_axis = torch.tensor([[0.0, 1.0, 0.0]], device=self.device)
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)

        def _fixed_origin(parent: str, child: str, axis: torch.Tensor, joint_name: str):
            """Pure fixed origin O such that pose(child in parent) = O then R(axis, q)."""
            rel_p, rel_q = _rel(parent, child)
            neg_rot = quat_from_angle_axis(-_q0(joint_name), axis)  # (1,4)
            zero = torch.zeros(1, 3, device=self.device)
            return combine_frame_transforms(rel_p, rel_q, zero, neg_rot)

        # revolute-joint fixed origins (parent -> child, rotation axis, joint name)
        self._fk_O_j1p = _fixed_origin("link1", "inter_joint1", y_axis, "joint1_pitch")
        self._fk_O_j1y = _fixed_origin("inter_joint1", "link2", z_axis, "joint1_yaw")
        self._fk_O_j2p = _fixed_origin("link2", "inter_joint2", y_axis, "joint2_pitch")
        self._fk_O_j2y = _fixed_origin("inter_joint2", "link3", z_axis, "joint2_yaw")
        self._fk_O_j3p = _fixed_origin("link3", "inter_joint3", y_axis, "joint3_pitch")
        self._fk_O_j3y = _fixed_origin("inter_joint3", "link4", z_axis, "joint3_yaw")
        # fixed-joint offsets (no joint rotation to remove)
        self._fk_O_fc = _rel("link2", "fc")           # fc (tracked base) in link2
        self._fk_O_front = _rel("link1", "front_end")  # front_end link in link1
        self._fk_O_back = _rel("link4", "back_end")    # back_end link in link4

        # column index of each joint in the _joint_ids / _desired_joint_pos_w ordering
        names = list(self._joint_ids[1])
        self._fk_joint_col = {n: names.index(n) for n in (
            "joint1_pitch", "joint1_yaw", "joint2_pitch",
            "joint2_yaw", "joint3_pitch", "joint3_yaw",
        )}
        self._fk_y_axis = y_axis
        self._fk_z_axis = z_axis

        # visual-mesh offset of each end link inside its own body frame, parsed from the URDF so
        # the marker lands on the *visible* end sphere (e.g. back_end's sphere sits at +0.45x)
        # rather than on the bare body-frame origin. Robust to future geometry edits in the URDF.
        self._fk_front_visual_offset = self._read_link_visual_offset("front_end")
        self._fk_back_visual_offset = self._read_link_visual_offset("back_end")
        self._end_fk_ready = True

    def _read_link_visual_offset(self, link_name: str) -> torch.Tensor:
        """Return the <visual><origin xyz> of a URDF link as a (1,3) tensor (zeros if unavailable)."""
        offset = [0.0, 0.0, 0.0]
        try:
            import xml.etree.ElementTree as ET

            urdf_path = getattr(self.cfg.robot_cfg.spawn, "asset_path", None)
            if urdf_path is not None and str(urdf_path).endswith(".urdf"):
                root = ET.parse(urdf_path).getroot()
                for link in root.findall("link"):
                    if link.get("name") == link_name:
                        origin = link.find("visual/origin")
                        if origin is not None and origin.get("xyz") is not None:
                            offset = [float(v) for v in origin.get("xyz").split()]
                        break
        except Exception as e:  # noqa: BLE001 -- visualization must never break the sim
            print(f"[dragon_track_env] could not read visual offset for '{link_name}': {e}")
        return torch.tensor([offset], device=self.device, dtype=torch.float32)

    def _fk_end_pos(self, fc_p: torch.Tensor, fc_q: torch.Tensor, q_abs: torch.Tensor):
        """FK the visible front_end / back_end tip positions for one reference frame.

        Args:
            fc_p, fc_q: target base (fc) pose in world, (n,3) / (n,4).
            q_abs: absolute target joint angles, (n, 6), ordered like ``_joint_ids``.
        Returns:
            (fe_pos, be_pos): world positions of the front/back visible end tips, each (n,3).
        """
        n = fc_p.shape[0]
        col = self._fk_joint_col

        def _fixed(O):
            return O[0].expand(n, -1), O[1].expand(n, -1)

        def _joint_tf(O, axis: torch.Tensor, joint_name: str):
            """pose of child in parent at the target joint angle: O then R(axis, q)."""
            rot = quat_from_angle_axis(q_abs[:, col[joint_name]], axis.expand(n, -1))  # (n,4)
            zero = torch.zeros(n, 3, device=self.device)
            return combine_frame_transforms(O[0].expand(n, -1), O[1].expand(n, -1), zero, rot)

        def _chain(a, b):
            return combine_frame_transforms(a[0], a[1], b[0], b[1])

        y, z = self._fk_y_axis, self._fk_z_axis
        # link1 -> link2 (through joint1 pitch/yaw)
        l1_ij1 = _joint_tf(self._fk_O_j1p, y, "joint1_pitch")
        ij1_l2 = _joint_tf(self._fk_O_j1y, z, "joint1_yaw")
        l1_l2 = _chain(l1_ij1, ij1_l2)
        # fc (tracked base) in link1 = link1->link2 then fixed link2->fc
        fc_in_l1 = _chain(l1_l2, _fixed(self._fk_O_fc))

        # invert to get link1 world pose from the known fc target world pose
        inv_q = quat_conjugate(fc_in_l1[1])
        inv_p = -quat_apply(inv_q, fc_in_l1[0])
        l1_p, l1_q = combine_frame_transforms(fc_p, fc_q, inv_p, inv_q)

        # front_end (fixed on link1)
        fe_p, fe_q = combine_frame_transforms(l1_p, l1_q, *_fixed(self._fk_O_front))

        # forward from link1 to link4, then back_end (fixed on link4)
        l2_p, l2_q = combine_frame_transforms(l1_p, l1_q, l1_l2[0], l1_l2[1])
        l2_ij2 = _joint_tf(self._fk_O_j2p, y, "joint2_pitch")
        ij2_l3 = _joint_tf(self._fk_O_j2y, z, "joint2_yaw")
        l2_l3 = _chain(l2_ij2, ij2_l3)
        l3_p, l3_q = combine_frame_transforms(l2_p, l2_q, l2_l3[0], l2_l3[1])
        l3_ij3 = _joint_tf(self._fk_O_j3p, y, "joint3_pitch")
        ij3_l4 = _joint_tf(self._fk_O_j3y, z, "joint3_yaw")
        l3_l4 = _chain(l3_ij3, ij3_l4)
        l4_p, l4_q = combine_frame_transforms(l3_p, l3_q, l3_l4[0], l3_l4[1])
        be_p, be_q = combine_frame_transforms(l4_p, l4_q, *_fixed(self._fk_O_back))

        # shift each marker onto the visible end-sphere (visual offset in the end-link frame)
        fe_pos = fe_p + quat_apply(fe_q, self._fk_front_visual_offset.expand(n, -1))
        be_pos = be_p + quat_apply(be_q, self._fk_back_visual_offset.expand(n, -1))
        return fe_pos, be_pos

    def _compute_target_end_pos_w(self) -> torch.Tensor:
        """FK the target end tips over sparsely-sampled future frames of the look-ahead horizon.

        The `track_horizon` future reference frames are subsampled with stride
        ``TRACK_ENDS_FRAME_STRIDE`` (frame 0 = current desired frame is always included).
        Returns a (num_envs, num_sampled_frames * 2, 3) tensor, each frame contributing
        [front_end, back_end].
        """
        stride = max(1, TRACK_ENDS_FRAME_STRIDE)
        frames = list(range(0, self.cfg.track_horizon, stride))
        q_default = self._joint_default_pos
        outs = []
        for f in frames:
            fc_p = self._target_pos_w[:, f]
            fc_q = self._target_quat_w[:, f]
            q_abs = q_default + self._target_joint_pos_w[:, f]
            fe_pos, be_pos = self._fk_end_pos(fc_p, fc_q, q_abs)
            outs.append(torch.stack((fe_pos, be_pos), dim=1))  # (n, 2, 3)
        return torch.cat(outs, dim=1)  # (n, num_frames*2, 3)

    def _setup_actual_ends(self) -> None:
        """Build the per-end-body visual offset (ordered by the contact sensor's body list) so the
        red 'actual' markers land on the same visible end spheres as the green target markers."""
        names = list(self._end_contact_sensor.body_names)
        offsets = [self._read_link_visual_offset(nm)[0] for nm in names]  # each (3,)
        self._actual_ends_visual_offset = torch.stack(offsets, dim=0).unsqueeze(0)  # (1, B, 3)
        self._actual_ends_ready = True

    def _compute_actual_end_pos_w(self) -> torch.Tensor:
        """Actual visible end-tip positions from the end contact sensor, (num_envs, B, 3)."""
        pos_w = self._end_contact_sensor.data.pos_w      # (n, B, 3)
        quat_w = self._end_contact_sensor.data.quat_w    # (n, B, 4)
        off = self._actual_ends_visual_offset.expand(pos_w.shape[0], -1, -1)  # (n, B, 3)
        return pos_w + quat_apply(quat_w, off)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = VisualizationMarkersCfg(
                    markers={
                        "frame": sim_utils.UsdFileCfg(
                            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                            scale=(0.25, 0.25, 0.25),
                        ),
                    }
                )
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_pose"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "thrusts_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.02, 0.02, 0.02)
                marker_cfg.prim_path = "/Visuals/Command/thrusts_visualization"
                self.thrusts_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "target_ends_visualizer"):
                marker_cfg = VisualizationMarkersCfg(
                    markers={
                        "end": sim_utils.SphereCfg(
                            radius=0.04,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.2)),
                        ),
                    }
                )
                marker_cfg.prim_path = "/Visuals/Command/target_ends"
                self.target_ends_visualizer = VisualizationMarkers(marker_cfg)
            if not hasattr(self, "actual_ends_visualizer"):
                marker_cfg = VisualizationMarkersCfg(
                    markers={
                        "end": sim_utils.SphereCfg(
                            radius=0.04,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                        ),
                    }
                )
                marker_cfg.prim_path = "/Visuals/Command/actual_ends"
                self.actual_ends_visualizer = VisualizationMarkers(marker_cfg)
            # set visibility per group so each checkbox controls its own marker
            self.goal_pos_visualizer.set_visibility(getattr(self, "_vis_targets", False))
            self.thrusts_visualizer.set_visibility(getattr(self, "_vis_thrusts", False))
            self.target_ends_visualizer.set_visibility(getattr(self, "_vis_track_ends", False))
            self.actual_ends_visualizer.set_visibility(getattr(self, "_vis_actual_ends", False))
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "thrusts_visualizer"):
                self.thrusts_visualizer.set_visibility(False)
            if hasattr(self, "target_ends_visualizer"):
                self.target_ends_visualizer.set_visibility(False)
            if hasattr(self, "actual_ends_visualizer"):
                self.actual_ends_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        if getattr(self, "_vis_targets", False):
            self.goal_pos_visualizer.visualize(self._desired_pos_w, self._desired_quat_w)

        if getattr(self, "_vis_thrusts", False):
            thrusts = self._target_thrust_force
            thrusts_pos = self._thrust_sensor.data.pos_w
            thrusts_quat = self._thrust_sensor.data.quat_w

            if thrusts_pos is None or thrusts_quat is None:
                print("Thrust positions not available yet for debug visualization.")
                return

            thrusts = thrusts.reshape(-1, 3)
            thrusts_pos = thrusts_pos.reshape(-1, 3)
            thrusts_quat = thrusts_quat.reshape(-1, 4)
            local_offset = torch.zeros_like(thrusts_pos)
            local_offset[:, 2] = 0.03
            thrusts_pos += quat_apply(thrusts_quat, local_offset)

            thrusts_mag = torch.linalg.norm(thrusts, dim=-1)

            unit_dir = normalize(thrusts)

            orientations = quat_from_angle_axis(torch.zeros(thrusts_mag.shape[0], device=self.device), unit_dir)
            x2z_quat = quat_from_euler_xyz(
                torch.zeros(unit_dir.shape[0], device=self.device, dtype=unit_dir.dtype),
                -math.pi / 2 * torch.ones(unit_dir.shape[0], device=self.device, dtype=unit_dir.dtype),
                torch.zeros(unit_dir.shape[0], device=self.device, dtype=unit_dir.dtype),
            )  # (N,4)
            # orientations = quat_mul(thrusts_quat, orientations)
            # orientations = quat_mul(orientations, x2z_quat)

            orientations = quat_mul(thrusts_quat, x2z_quat)

            scales = torch.ones_like(thrusts)
            scales[:, 0] = thrusts_mag * 0.5
            self.thrusts_visualizer.visualize(translations=thrusts_pos, orientations=orientations, scales=scales)

        if getattr(self, "_vis_track_ends", False):
            # FK the target end-link (front_end / back_end) positions over the sampled future frames
            if not self._end_fk_ready:
                self._setup_end_fk()
            end_pos_w = self._compute_target_end_pos_w()  # (num_envs, num_frames*2, 3)
            self.target_ends_visualizer.visualize(translations=end_pos_w.reshape(-1, 3))

        if getattr(self, "_vis_actual_ends", False):
            # draw the robot's real end tips (red) every frame for direct target-vs-actual comparison
            if not self._actual_ends_ready:
                self._setup_actual_ends()
            actual_pos_w = self._compute_actual_end_pos_w()  # (num_envs, B, 3)
            self.actual_ends_visualizer.visualize(translations=actual_pos_w.reshape(-1, 3))
