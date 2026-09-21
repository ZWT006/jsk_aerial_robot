## Setting
1. Very important: set `NERVE_COMM = 0` and write firmware to spinal.

2. Connect like this:

 ``````````````````````````````````
 ``````````````````````````````````
 ```  PC --spinal--ESC---rotor  ```
 ```  |              |          ```
 ```  |-- power supply          ```
 ```  |                         ```
 ```  |--force sensor           ```
 ``````````````````````````````````
 ``````````````````````````````````
3. Create new wired setting for power supply. Set IP address of the power supply to `192.168.0.2`, Netmask to '255.255.255.0', and Gateway to 192.168.0.1.
4. Power on the power supply and set the voltage as you like (e.g. 25.2V).
5. In your terminal, do:
   ```
   $ roslaunch motor_test test.launch
   ```

   **parameter**:
   - `test_mode`: the test mode. Step mode (:=0) is to give continous steps command to the ESC, the problem is the bias of the force sensor increases during the steps commands. Thus, we recommand to use **one-shot** mode (:=1, default mode), which iteratively gives an increasing PWM value. The difference is this mode stops the motor rotation after each step, and re-calibrate the force sensor before the next step. So the bias of force sensor can be considered to be zero.
   - `run_duration`: the duration of motor rotation during each step.
   - `pwm_incremental_value`: the increamental value between two steps.
   - `min_pwm_value`/`max_pwm_value`: the min/max of the pwm value
   - `raise_duration`/`brake_duration`: these are the special parameters for one-shot mode, since we have to consider the raise-up phase after staring rotation and brake-down phase after stopping rotation. For small propellers (e.g. 5inch), these value can be small. For large propellers (e.g. 14inch), please increase there values
   - `force_sensor`: the type of force sensor. Options are CFS034CA301U (default); PFS055YA501U6 (larger type)

   **sample**:
   - for 5inch propeller:
     ```
     $ roslaunch motor_test test.launch max_pwm_value:=1900 raise_duration:=1.0 brake_duration:=2.0 run_duration:=2.0
     ```

   - for 5inch propeller:
     ```
     $ roslaunch motor_test test.launch max_pwm_value:=1200 raise_duration:=2  brake_duration:=5 run_duration:=2
     ```

6. Start logging by `rostopic pub -1 /start_log_cmd std_msgs/Empty "{}" `.

7. Cool down your motor by sending 'rostopic pub -1 /pwm_test std_msgs/Float32 "data: 0.54" ' and when it is cool enough, send 'rostopic pub -1 /pwm_test std_msgs/Float32 "data: 0.5" '，and then start a next test.

## General calib for ESC:

Low PWM: 1000 (start with 1050)
High PWM: 1900

## Servo response logging

`servo_response_node` collects joint/gimbal response data for system
identification. It is robot-agnostic: pass the robot namespace, state topic,
command topic, and axis names from launch.

### Signal sets

Use only these three `signal_type` values:

- `step`: step-response set. The command holds each level for `cmd_period`,
  jumps by `step_height`, sweeps the full `[-position_limit, position_limit]`
  range, and then reverses.
- `sweep`: identification set. This is the beetle_omni SweepSinWave: a sine
  wave with short-long-short period sweep and a matching low-high-low amplitude
  envelope. Use `cmd_period` for the shortest period and `period_scale` for the
  longest period.
- `multi`: validation set. This fixed sequence runs sine, triangle, square,
  trapezoid, and sine-with-triangle-envelope waves. Only `cmd_period` is tuned.

### Common parameters

- `robot_ns`: robot namespace. Relative topics become `/robot_ns/topic`.
- `state_topic`: feedback `sensor_msgs/JointState` topic.
- `command_topic`: position command `sensor_msgs/JointState` topic, usually
  `gimbals_ctrl` or `joints_ctrl`.
- `axis_names`: ordered target joint/gimbal names. If empty, the first
  `axis_count` feedback axes are used.
- `enabled_axes`: per-axis mask. Disabled axes are still published at
  `default_position`.
- `default_position`: scalar or array added to the generated relative command.
- `control_freq`: command publish rate [Hz].
- `duration`: data collection duration [s].
- `position_limit`: non-negative symmetric work range [rad]. Set it to `0` to
  keep the generated relative command at zero.
- `enable_save`, `save_path`, `file_prefix`: CSV logging controls.
- `effort_ctrl`: if false, publish position command to `command_topic`; if true,
  publish Float64 effort commands using `effort_topics` or auto-built
  `servo_controller/<servo_group>/controllerN/simulation/command` topics.

### Attention: real-hardware servo safety

Sparse commands do not make the other servos safe automatically:

- Every servo listed in `axis_names` is included in every position command.
  Setting its `enabled_axes` entry to `false` only disables the test signal;
  that servo still receives `default_position`. It is **not** torque-disabled.
- Servos omitted from `axis_names` do not receive `default_position` from this
  node. `ServoBridge` forwards only the named servos, but Spinal retains the
  previous target of every other servo and continues transmitting those cached
  targets to the hardware. An omitted servo can therefore remain torque-enabled
  and hold or move toward an earlier target.
- Before running on real hardware, explicitly torque-disable every servo that is
  not under test, and verify its torque state. Do not rely on omission from
  `axis_names` or on `enabled_axes: false` as a safety mechanism. Also ensure
  that no other controller is publishing commands to the same control topic.

Use the platform's per-servo torque-control procedure (for example,
`spinal/ServoTorqueCmd` on `<robot_ns>/servo/torque_enable` or the servo monitor)
when only selected servos should remain enabled. The
`<robot_ns>/<servo_group>/torque_enable` service switches the entire configured
group and cannot isolate individual axes.

### Beetle examples

StepSet:

```bash
roslaunch motor_test servo_response.launch \
  robot_ns:=beetle_omni \
  signal_type:=step \
  cmd_period:=0.2 \
  step_height:=0.15 \
  enable_save:=true \
  save_path:=/tmp
```

IdentSet:

```bash
roslaunch motor_test servo_response.launch \
  robot_ns:=beetle_omni \
  signal_type:=sweep \
  cmd_period:=0.04 \
  period_scale:=50 \
  enable_save:=true \
  save_path:=/tmp
```

ValSet:

```bash
roslaunch motor_test servo_response.launch \
  robot_ns:=beetle_omni \
  signal_type:=multi \
  cmd_period:=4.0 \
  enable_save:=true \
  save_path:=/tmp
```

DRAGON gimbal example:

```bash
roslaunch motor_test servo_response.launch \
  robot_ns:=dragon \
  servo_group:=gimbals \
  command_topic:=gimbals_ctrl \
  axis_names:="[gimbal2_roll, gimbal2_pitch, gimbal1_roll, gimbal3_roll, gimbal1_pitch, gimbal3_pitch, gimbal4_roll, gimbal4_pitch]" \
  enabled_axes:="[true, true, true, true, true, true, true, true]" \
  position_limit:=3.14 \
  enable_save:=true
```

For DRAGON joints, use `servo_group:=joints`, `command_topic:=joints_ctrl`, and
the joint name list used by that platform.
