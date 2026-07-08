# Motor Test

This package runs an open-loop motor/propeller thrust test. It publishes a PWM test command to spinal, logs force sensor data, logs power-supply current, and can optionally log DShot telemetry.

## Hardware Setup

1. Set `NERVE_COMM = 0` in spinal firmware and write the firmware to the board.
2. Connect the test bench:

```text
PC -- spinal -- ESC -- motor -- propeller
 |                |
 |                +-- power supply output
 |
 +-- force sensor over USB
 |
 +-- power supply control over Ethernet
```

3. Connect the force sensor to the PC by USB.
4. Connect the power supply to the PC by Ethernet. The `takasako_sps` node opens/closes the output and reads current; it does not set the voltage.
5. Set the power-supply voltage manually before each test point or voltage sweep. For example, set `25.2 V` or `26.2 V` on the supply front panel/software.
6. Check the power-supply IP settings. The current `takasako_sps` driver default target IP is `192.168.0.1` unless the node private parameter `ip_address` is changed.

## Safety Checklist

Do these checks before publishing `/start_log_cmd`; that command starts the actual PWM sequence.

- Mount the motor rigidly, keep the test area clear, and wear eye protection.
- Confirm the propeller type, propeller mounting direction, and motor rotation direction.
- For the first bring-up, run a no-propeller dry test with a very small PWM range to confirm the whole software and wiring flow.
- Confirm that `/pwm_test` value `0.5` stops the motor before installing a propeller.
- Confirm the power-supply voltage and current limit before each run.
- Keep a fast way to stop power output, for example power-supply emergency stop or `/power_off_cmd`.
- Confirm the force sensor sign. If thrust appears as negative `fz`, use `--if_reverse` when analyzing data or remount the sensor.

## What `/start_log_cmd` Does

Launch starts the nodes and performs the initial force-sensor calibration. In one-shot mode, the node also recalibrates the force sensor between PWM steps.

The test itself does not start until this command is sent:

```bash
rostopic pub -1 /start_log_cmd std_msgs/Empty "{}"
```

That callback creates `motor_test_<timestamp>.txt`, publishes the first PWM command, starts the timer sequence, and enables force logging. Without this command, the force callback and PWM timer return immediately.

## Force Sensor Zero Drift

Current handling of force-sensor zero drift is simple and practical:

- At launch, `motor_test_node` calls `/cfs_sensor_calib` once before the test starts. The `cfs_sensor` node estimates a zero offset by averaging recent raw samples, then subtracts that offset from all later wrench outputs.
- In one-shot mode, the motor is stopped after each PWM step, a brake/cool-down period is inserted, and `/cfs_sensor_calib` is called again before the next PWM step. This is the main protection against zero drift during a sweep.
- During logging, one-shot mode labels samples as `raise`, `valid`, and `brake`. The analysis script uses only the `valid` window and averages samples for each PWM value, which helps reject spin-up/spin-down transients and reduce short-term noise.

Limits of the current approach:

- There is no explicit long-term drift model, temperature compensation, or post-processing bias estimation inside each valid window.
- Step mode does not recalibrate between PWM levels, so it is more sensitive to zero drift than one-shot mode.

For best repeatability, use one-shot mode and avoid touching the sensor during calibration.

## PWM Units

`min_pwm_value`, `max_pwm_value`, and the first log column use raw PWM-style values such as `1000`, `1200`, or `1800`.

Internally, `motor_test` publishes a normalized command:

```text
/pwm_test = pwm_value / pwm_range
```

With the default `pwm_range = 2000`, this means:

```text
1000 -> 0.50
1050 -> 0.525
1800 -> 0.90
1900 -> 0.95
2000 -> 1.00
```

Choose the raw PWM range according to the motor, ESC, propeller, and robot configuration being tested.

## Power Supply Node

`power_node` from `takasako_sps` is useful, but limited:

- `/power_on_cmd` sends `OUTP ON`.
- `/power_off_cmd` sends `OUTP OFF`.
- `/power_info` currently fills only `currency` from `MEAS:CURR?`.
- It does not set output voltage.
- It does not currently fill `PowerInfo.voltage` or `PowerInfo.power`.

If DShot telemetry is enabled, the logged `voltage` column comes from ESC telemetry, not from the power-supply node.

## Launch Parameters

- `test_mode`: `0` is step mode; `1` is one-shot mode. One-shot mode is recommended because it stops the motor after each PWM step and recalibrates the force sensor before the next step.
- `run_duration`: valid rotation duration for each PWM step.
- `pwm_incremental_value`: raw PWM increment between steps.
- `min_pwm_value` and `max_pwm_value`: raw PWM sweep range.
- `raise_duration`: time allowed for spin-up before the valid window in one-shot mode.
- `brake_duration`: time allowed for spin-down before the next calibration in one-shot mode.
- `force_sensor`: force sensor config name. Common options are `CFS034CA301U` and `PFS055YA501U6`.
- `has_dshot_telemetry`: set true to log ESC RPM, temperature, and voltage.
- `dshot_telemetry_id`: ESC telemetry index, from `1` to `4`.

## Example Commands

No-propeller first check:

```bash
roslaunch motor_test test.launch \
  test_mode:=1 \
  min_pwm_value:=1000 \
  max_pwm_value:=1100 \
  pwm_incremental_value:=50 \
  run_duration:=0.5 \
  raise_duration:=0.5 \
  brake_duration:=1.0 \
  has_dshot_telemetry:=true \
  dshot_telemetry_id:=1
```

Generic propeller test example:

```bash
roslaunch motor_test test.launch \
  test_mode:=1 \
  min_pwm_value:=1050 \
  max_pwm_value:=1700 \
  pwm_incremental_value:=50 \
  run_duration:=2.0 \
  raise_duration:=2.0 \
  brake_duration:=4.0 \
  has_dshot_telemetry:=true \
  dshot_telemetry_id:=1 \
  force_sensor:=CFS034CA301U
```

Start the test after all safety checks:

```bash
rostopic pub -1 /start_log_cmd std_msgs/Empty "{}"
```

Cool down the motor between tests:

```bash
rostopic pub -1 /pwm_test std_msgs/Float32 "data: 0.54"
```

Stop the motor after cooling:

```bash
rostopic pub -1 /pwm_test std_msgs/Float32 "data: 0.5"
```

## Analyze Data

Logs are usually written under `~/.ros/` when launched with `roslaunch`, unless the node working directory is changed.

For a log with DShot telemetry and negative `fz` thrust direction, use:

```bash
python3 aerial_robot_nerve/motor_test/scripts/analyze_data.py \
  motor_test_<timestamp>.txt \
  --folder_path ~/.ros/ \
  --has_telemetry \
  --order 3 \
  --if_reverse
```

Use `--order` to choose the polynomial order. Add `--has_telemetry` when the log contains RPM, temperature, and ESC voltage columns. Add `--if_reverse` when the force sensor sign makes thrust negative.

## General ESC Calibration

Low PWM: `1000` (start carefully from `1050` if needed)
High PWM: `1900`
