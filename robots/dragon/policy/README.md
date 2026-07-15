# Dragon Tracker RL-Agent

This is the deployment-side whole-body RL tracker for DRAGON. It runs the ONNX
policy exported from `dragon_track_env.py`, consumes online
`aerial_robot_msgs/FullStateTarget` messages, and publishes link-joint, gimbal,
and paired-thrust commands through the normal JSK aerial robot stack.

The current tracker does **not** load a CSV trajectory internally. A separate
target publisher, for example `scripts/keyframe_tracking.py`, sends one
`FullStateTarget` at a time.

## Current Runtime Flow

1. `launch/bringup_rl_tracking.launch` loads the robot model, `RLServo.yaml`,
   `TrackerRLAgentConfig.yaml`, `dragon_tracker_rlagent`, `servo_bridge`, and
   the simulator helpers.
2. `scripts/keyframe_tracking.py` publishes target states to
   `/dragon/full_state_target`. Before takeoff it keeps publishing the first
   keyframe; after `/dragon/teleop_command/takeoff` it advances through the
   spline at 40 Hz.
3. The RL agent keeps a FIFO of the latest target frames. The FIFO length is
   configured by `rlagent/target_obs_horizon`.
4. The policy always runs once enough state is available, including on the
   ground. Command publication is gated per channel by safety flags.

Recommended simulation bringup:

```bash
roslaunch dragon bringup_rl_tracking.launch sim:=True headless:=False
roslaunch dragon demo/keyframe_tracking.launch robot_ns:=dragon mode:=sim
```

Typical remote command order:

```bash
rostopic pub -1 /dragon/teleop_command/enable_joint std_msgs/Empty
rostopic pub -1 /dragon/teleop_command/enable_gimbal std_msgs/Empty
rostopic pub -1 /dragon/teleop_command/start std_msgs/Empty
rostopic pub -1 /dragon/teleop_command/takeoff std_msgs/Empty
```

`keyframe_tracking.py` also listens to:

```text
/dragon/teleop_command/pause_tracking
/dragon/teleop_command/disable_pause_tracking
/dragon/teleop_command/land
/dragon/teleop_command/halt
/dragon/teleop_command/force_landing
```

`pause_tracking`, `land`, `halt`, and `force_landing` pause target-state
publishing. `disable_pause_tracking` resumes it.

## ROS Interface

Main subscriptions used by `dragon_tracker_rlagent`:

```text
/dragon/joint_states
/dragon/full_state_target                 # configurable by rlagent/target_state_topic
/dragon/uav/baselink/odom                 # only when rlagent/ideal_obs=true
/dragon/teleop_command/enable_joint
/dragon/teleop_command/enable_gimbal
/dragon/teleop_command/halt
/dragon/teleop_command/force_landing
```

Main publications:

```text
/dragon/four_axes/command
/dragon/joints_ctrl                       # only in position-control mode
/dragon/gimbals_ctrl                      # only in position-control mode
/dragon/rlagent/thrust_cmd_debug          # debug mirror, not proof of real thrust output
/dragon/rlagent/joint_cmd_debug
/dragon/rlagent/gimbal_cmd_debug
/dragon/rlagent/observation_debug
```

When `joint_effort_ctrl` or `gimbal_effort_ctrl` is true, the RL agent publishes
effort commands directly to each Gazebo controller:

```text
/dragon/servo_controller/joints/controller*/simulation/command
/dragon/servo_controller/gimbals/controller*/simulation/command
```

Runtime services:

```text
/dragon/rlagent/reload_policy             # aerial_robot_model/ReloadPolicy
/dragon/rlagent/reload_params             # std_srvs/Trigger
```

## Observation Layout

The observation is fully configured by
`config/quad/v1_5_202601/control/TrackerRLAgentConfig.yaml`:

```yaml
target_obs_horizon: 5
target_obs: [root_pos, root_rot_vec, joint_pos]
observations:
  - target_obs_history
  - joint_pos
  - joint_vel
  - gimbal_pos
  - gimbal_vel
  - root_rot_vec
  - lin_vel
  - ang_vel
  - last_action
```

`target_obs_history` is built from the target FIFO in arrival order
(oldest-to-newest). If fewer than `target_obs_horizon` target frames have been
received, the latest frame is repeated until the horizon is full. If no target
has arrived yet, the agent warns:

```text
[RL-Tracker] No target state received yet!
```

`FullStateTarget.joint_state.name` should be populated. When names are present,
the RL agent reorders target joints by name into the configured training order.
If names are missing, the incoming position array is assumed to already be in
`rlagent/joint_names` order.

Current deployment orders, matching `dragon_track_env.py`:

```yaml
joint_names:
  [joint1_yaw, joint2_pitch, joint1_pitch, joint2_yaw, joint3_pitch, joint3_yaw]

gimbal_names:
  [gimbal2_roll, gimbal2_pitch, gimbal1_roll, gimbal3_roll,
   gimbal1_pitch, gimbal3_pitch, gimbal4_roll, gimbal4_pitch]

thrust_pair_names:
  [edf2, edf1, edf3, edf4]
```

Note that `keyframe_tracking.py` publishes joint names in physical controller
order:

```text
[joint1_pitch, joint1_yaw, joint2_pitch, joint2_yaw, joint3_pitch, joint3_yaw]
```

This is safe because the target callback maps by joint name.

## Action Parsing

The action vector is parsed according to `rlagent/actions`:

```yaml
actions: [joint, gimbal, thrust]
```

With the current policy this means:

```text
action[0:6]    -> link joints
action[6:14]   -> gimbals
action[14:18]  -> paired EDF thrust
```

Joint and gimbal actions are converted to position targets:

```text
joint_target  = action * scales/joint_act  + joint_default
gimbal_target = action * scales/gimbal_act + gimbal_default
```

They are then clamped by `joint_limit` and `gimbal_limit`. In effort mode the
position target is converted to torque:

```text
effort = kp * (target_position - measured_position) - kd * measured_velocity
effort = clamp(effort, -effort_limit, effort_limit)
```

Thrust action follows the paired-rotor convention used in training:

```text
per_edf_thrust = action * scales/thrust_act + thrust_hover
per_edf_thrust = clamp(per_edf_thrust, 0, thrust_limit)
target_thrust  = per_edf_thrust * thrust_scale
base_thrust[module_index] = 2 * target_thrust
```

`thrust_limit` is therefore a per-EDF limit. The value published to
`four_axes/command.base_thrust` is doubled because each DRAGON module is a
dual-EDF pair.

## Effort Control Parameters

`RLServo.yaml` is used in two places:

1. `servo_bridge` loads and starts the Gazebo effort controllers.
2. `dragon_tracker_rlagent` reads `kp`, `kd`, and `effort_limit` from the same
   parameter tree and computes the effort command itself.

The active simulation values are:

```text
link joints:
  kp = 6.39763
  kd = 0.399211
  effort_limit = 8.2

gimbal roll:
  kp = 2.34079
  kd = 0.146053
  effort_limit = 3.0

gimbal pitch:
  kp = 0.593
  kd = 0.037
  effort_limit = 0.76
```

To tune these values, edit:

```text
config/quad/v1_5_202601/RLServo.yaml
```

Then restart the launch, or reload the RL-side copy with:

```bash
rosservice call /dragon/rlagent/reload_params
```

Changing controller type or Gazebo controller parameters still requires a
restart because `servo_bridge` creates those controllers during bringup.

## Initial Joint Pose

In simulation, `bringup_rl_tracking.launch` starts:

```text
scripts/gazebo_quick_init_pose.py
```

This script reads `servo_controller/joints/*/simulation/init_pose` from
`RLServo.yaml` and calls Gazebo `set_model_configuration` until the joint state
matches the configured initial pose.

For effort-control mode:

```text
init_pose   -> Gazebo model initial configuration
init_value  -> first effort command sent to the controller
```

The current physical controller order is:

```text
[joint1_pitch, joint1_yaw, joint2_pitch, joint2_yaw, joint3_pitch, joint3_yaw]
```

The configured initial pose is:

```text
[0.0, 1.57, 0.0, 1.57, 0.0, 1.57]
```

Do not compare this array directly with `rlagent/joint_names`; the RL policy
uses a different, training-specific order and maps by name.

## Safety Gates

The policy can forward on the ground, but outputs are only published when the
corresponding gate is open:

```text
joint command  -> enable_joint && action_seeded
gimbal command -> enable_gimbal && action_seeded
thrust command -> enable_thrust && control_timestamp > 0 && action_seeded && enable_gimbal
```

`control_timestamp > 0` becomes true after the aerial robot navigation stack
enters the takeoff/control phase. `enable_gimbal` also gates thrust on purpose:
it is the hardware safety interlock used before arming and takeoff.

`halt` and `force_landing` disable both joint and gimbal commands. Because
thrust is gated by `enable_gimbal`, they also stop real thrust publication from
the RL agent.

If `/dragon/four_axes/command` is empty after takeoff, check the periodic log:

```text
[RL-Tracker] Thrust gated: navi_state=... control_started=... control_timestamp=... action_seeded=... enable_gimbal=...
```

Also check the real command topic, not only the debug mirror:

```bash
rostopic echo /dragon/four_axes/command
rostopic echo /dragon/rlagent/thrust_cmd_debug
```

`rlagent/thrust_cmd_debug` is published in debug mode even when the real thrust
topic is gated.

## Logging

`rlagent_forward_info: true` prints a compact green summary every two seconds:

```text
[RL-Tracker] ORT 2.00s: calls=80 avg=0.102ms min=0.024ms max=3.280ms | target fifo 5/5 HOLD
```

When `rlagent_debug: true`, extra tracking diagnostics are printed below it:

```text
root err: xyz [...] |norm ...  rpy [...] |norm ...
joint err: Y1 ... P2 ...
gimbal err: GR2 ... GP2 ...
rotor: R1 ... R2 ...
```

Use `rlagent_debug: false` when only the compact ORT/timing line is needed.

## Key Files

```text
robots/dragon/src/control/dragon_tracker_rlagent.cpp
robots/dragon/include/dragon/control/dragon_tracker_rlagent.h
robots/dragon/config/quad/v1_5_202601/control/TrackerRLAgentConfig.yaml
robots/dragon/config/quad/v1_5_202601/RLServo.yaml
robots/dragon/launch/bringup_rl_tracking.launch
robots/dragon/launch/demo/keyframe_tracking.launch
robots/dragon/scripts/keyframe_tracking.py
robots/dragon/scripts/gazebo_quick_init_pose.py
robots/dragon/policy/dragon_track_env.py
aerial_robot_model/srv/ReloadPolicy.srv
```
