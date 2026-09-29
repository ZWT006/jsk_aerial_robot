# Neuron Servo Control

This context describes how a neuron board receives servo targets, maintains its local low-level control rhythm, and reports coherent feedback to spinal.

## Language

**Target frame**:
A CAN message carrying the latest desired servo state and its sequence from spinal to a neuron. Receiving it updates intent but does not itself execute a Dynamixel control cycle.
_Avoid_: Control trigger, UART trigger

**Pending target frame**:
The single coherent board-level copy of the newest valid target frame. CAN RX replaces it atomically, and the servo scheduler consumes it once at a high-rate deadline; it is not an instruction-queue entry.
_Avoid_: Target queue, control event

**Applied target**:
The coherent target snapshot frozen for use by a control cycle. Its sequence identifies what the low-level controller actually attempted to apply, rather than what neuron most recently received.
_Avoid_: Latest target, received target

**Control cycle**:
A neuron-local 5 ms servo iteration that obtains a coherent measurement and applies the selected target according to each servo's control mode.
_Avoid_: CAN cycle, target cycle, event sequence

**Servo instruction**:
A bounded FIFO record containing a command, servo mask, and optional final-feedback-group mask. It describes one logical bus operation and does not carry target values.
_Avoid_: Target event, UART packet

**Pending transaction**:
The single read request that has been transmitted but whose response has not yet been completed. It preserves the original absolute response deadline across servo-task ticks.
_Avoid_: Queue item, background thread

**Phase sync**:
The one-time alignment of a neuron's local control-cycle epoch to the arrival phase of a valid target frame from spinal.
_Avoid_: CAN-triggered control

**Holdover**:
Continued execution of the local control cycle from its established epoch while target frames are temporarily absent.
_Avoid_: Free running, unsynchronised mode

**Relock**:
Realignment of the local control-cycle epoch to the first valid target frame received after a prolonged target interruption. It occurs at a control-cycle boundary without replaying missed cycles.
_Avoid_: Restart, catch-up

**Fresh measurement**:
A coherent set of current, velocity, and position values successfully obtained for every servo expected in one feedback group. It does not imply that the following goal write succeeded.
_Avoid_: Cycle complete, write complete

**Control write status**:
The outcome of attempting to write goals for a control cycle, tracked separately from measurement freshness.
_Avoid_: Feedback status

**Maintenance state**:
An explicitly entered, torque-disabled state in which blocking configuration or Flash operations may suspend the real-time control guarantee.
_Avoid_: Normal control, service cycle

**Torque-off event**:
A safety command that must be executed at least once and cannot be discarded by latest-wins target coalescing.
_Avoid_: Torque target
