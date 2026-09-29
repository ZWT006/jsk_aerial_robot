# end_leg

Single-joint real-to-simulation calibration package.

## Bringup

Gazebo environment and calibration are started separately:

```bash
# Effort mode: simple_leg.urdf + MITServo.yaml
roslaunch end_leg bringup_leg.launch effort_ctrl:=true

# Position mode: simple_servo_leg.urdf + SimServo.yaml
roslaunch end_leg bringup_leg.launch effort_ctrl:=false
```

For hardware, start the minimal end-leg bringup first. It starts only the
serial connection and `servo_bridge`, using the four-servo configuration in
`config/Servo.yaml`:

```bash
roslaunch end_leg bringup.launch \
  serial_port:=/dev/flight_controller serial_baud:=2000000
```

Then run calibration against `gimbal1` (or select another configured servo via
`real_joint_name`):

```bash
roslaunch end_leg servo_calibration.launch \
  simulation:=false effort_ctrl:=false real_joint_name:=gimbal1
```

The default real-hardware topics are `/end_leg/joint_states` and
`/end_leg/gimbals_ctrl`. Override `real_robot_ns` if `bringup.launch` was
started with a different `robot_ns`.

**Set Joint Position in Gazebo**
```bash
rosservice call /gazebo/pause_physics

rosservice call /gazebo/set_model_configuration \
"{model_name: 'end_leg',
  urdf_param_name: '',
  joint_names: ['joint_pitch_servo'],
  joint_positions: [0.9]}"

rosservice call /gazebo/unpause_physics
```

## Four datasets

| Dataset | Signal | Required parameters |
|---|---|---|
| `PassiveSet` | Passive/free motion | `auto_swing:=false` |
| `StepSet` | Step | `auto_swing:=true step_mode:=true sweep_mode:=false cmd_period:=0.2 step_height:=0.15` |
| `IdentSet` | SweepSinWave | `auto_swing:=true step_mode:=false sweep_mode:=true cmd_period:=0.04 period_scale:=50` |
| `ValSet` | MultipleWave | `auto_swing:=true step_mode:=false sweep_mode:=false cmd_period:=4.0` |

Only these four collection modes are used. With both signal flags false, the
node selects `MultipleWave`.

### PassiveSet

Calibration only records feedback and never publishes a command:

```bash
roslaunch end_leg servo_calibration.launch \
  auto_swing:=false enable_save:=true
```

For a Gazebo release test, use effort mode with zero commanded torque. The
fixed root is at `z=0.8 m` with `+90 deg` pitch, so `q=0` is the hanging pose.
Set a nonzero initial angle with `/gazebo/set_model_configuration`, then release
it. A position controller cannot be used because it holds the joint.

### Effort margin

The movable links total `0.712 kg` with approximately `0.06394 kg m^2` inertia
about the joint. Gravity requires at most `1.76 Nm`, or about `1.41 Nm` inside
the default `joint_range=+/-0.9 rad`; the `6.6 Nm` limit is sufficient to move
through this range. The effort-mode controller is PD-only, so sufficient effort
does not guarantee zero steady-state tracking error under gravity.

### StepSet

The staircase starts at `default_position`, sweeps to the positive limit, then
through the default position to the negative limit, and finally returns to the
default position.

```bash
roslaunch end_leg servo_calibration.launch \
  auto_swing:=true enable_save:=true \
  step_mode:=true sweep_mode:=false cmd_period:=1.0 step_height:=0.05
```

### IdentSet

```bash
roslaunch end_leg servo_calibration.launch \
  auto_swing:=true enable_save:=true \
  step_mode:=false sweep_mode:=true cmd_period:=0.1 period_scale:=50
```

### ValSet

```bash
roslaunch end_leg servo_calibration.launch \
  auto_swing:=true enable_save:=true \
  step_mode:=false sweep_mode:=false cmd_period:=0.0
```

For hardware, use the same four dataset parameters with
`simulation:=false effort_ctrl:=false` and the required `real_robot_ns` and
`real_joint_name`.

## Communication delay

Simulation supports integer delays tied to the calibration update rate:

```text
delay_seconds = delay_step / control_freq
```

- `feedback_delay_step` delays position, velocity, and measured effort used by
  the controller and CSV.
- `command_delay_step` delays the command actually published.
- `target_rad` remains the current desired target before command delay.
- Hardware mode forces both delay steps to zero.

## CSV

Set `enable_save:=true` to write data under `save_path` (`/home/wentao` by
default in the launch file):

```text
time_s,target_rad,position_rad,velocity_rad_s,effort_Nm
```

Position, velocity, and effort must all exist and be finite in `JointState`.
`velocity_rad_s` comes directly from `JointState.velocity`; missing velocity is
a fatal error, with no finite-difference fallback. `effort_Nm` is measured
feedback effort, not the generated effort command.
