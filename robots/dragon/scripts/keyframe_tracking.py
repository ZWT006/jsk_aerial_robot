#!/usr/bin/env python3
"""Keyframe trajectory tracking for DRAGON robot.

Reads keyframes from YAML, computes a cubic Hermite spline at 40 Hz,
publishes FullStateTarget with child_frame_id='fc' (baselink), and
saves the reference trajectory to a .npz file beside the YAML.

Usage:
    rosrun dragon keyframe_tracking.py <yaml_path>
    rosrun dragon keyframe_tracking.py _config:=<yaml_path>

Modes (~mode):
    real (default): sample ~init_sample_num frames of odometry + joint states and
        replace the first YAML keyframe with the averaged measured state; after
        takeoff the spline reaches the second YAML keyframe within its 'duration',
        exactly like the sim interpolation.
    sim: start directly from the first YAML keyframe (robot is assumed to already
        be there, e.g. Gazebo quick-init).

Topics default to /<robot_ns>/... with robot_ns from the ~robot_ns parameter
(falling back to the node's namespace).

YAML format:
    keyframes:
      - pos: [x, y, z]
        rpy: [roll, pitch, yaw]
        joint_pos: [j1, j2, j3, j4, j5, j6]
        duration: 0.0   # first keyframe: ignored (robot starts here)
        # optional vel, omega, joint_vel default to [0,0,0] (stop at keyframe)
      - pos: [...]
        duration: 5.0   # time (s) to travel from previous keyframe
        vel: [vx, vy, vz]      # optional: velocity at this keyframe
        omega: [wx, wy, wz]    # optional: angular velocity at this keyframe
        joint_vel: [...]       # optional: joint velocity at this keyframe
"""

import sys
import os
import yaml
import numpy as np
import rospy
from aerial_robot_msgs.msg import FullStateTarget
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
import tf.transformations as tft

FREQ = 40.0
DT = 1.0 / FREQ
JOINT_NAMES = ['joint1_pitch', 'joint1_yaw', 'joint2_pitch', 'joint2_yaw', 'joint3_pitch', 'joint3_yaw']


def _wrap(a):
    return (np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def _hermite(p0, p1, m0, m1, T, t):
    s = t / T
    return ((2*s**3 - 3*s**2 + 1)*p0 + (s**3 - 2*s**2 + s)*T*m0 +
            (-2*s**3 + 3*s**2)*p1 + (s**3 - s**2)*T*m1)


def _hermite_dot(p0, p1, m0, m1, T, t):
    s = t / T
    return (((6*s**2 - 6*s) / T)*p0 + (3*s**2 - 4*s + 1)*m0 +
            ((6*s - 6*s**2) / T)*p1 + (3*s**2 - 2*s)*m1)


def load_keyframes(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return [{
        'pos':       np.array(kf['pos'],       dtype=float),
        'rpy':       np.array(kf['rpy'],       dtype=float),
        'joint_pos': np.array(kf['joint_pos'], dtype=float),
        'duration':  float(kf.get('duration', 0.0)),
        'vel':       np.array(kf.get('vel',       [0., 0., 0.]), dtype=float),
        'omega':     np.array(kf.get('omega',     [0., 0., 0.]), dtype=float),
        'joint_vel': np.array(kf.get('joint_vel', [0.] * 6),    dtype=float),
    } for kf in data['keyframes']]


def collect_initial_keyframe(odom_topic, joint_topic, n_samples, timeout=10.0):
    """Average n_samples of odometry + joint states into a start keyframe."""
    odoms, joints = [], []

    def odom_cb(msg):
        if len(odoms) < n_samples:
            odoms.append(msg)

    def joint_cb(msg):
        if len(joints) < n_samples and msg.name:
            joints.append(msg)

    odom_sub = rospy.Subscriber(odom_topic, Odometry, odom_cb, queue_size=n_samples)
    joint_sub = rospy.Subscriber(joint_topic, JointState, joint_cb, queue_size=n_samples)
    rospy.loginfo("Collecting %d frames from %s and %s ...", n_samples, odom_topic, joint_topic)

    deadline = rospy.Time.now() + rospy.Duration(timeout)
    poll = rospy.Rate(100)
    while not rospy.is_shutdown() and (len(odoms) < n_samples or len(joints) < n_samples):
        if rospy.Time.now() > deadline:
            odom_sub.unregister()
            joint_sub.unregister()
            raise RuntimeError(
                "Timed out collecting initial state: %d/%d odom, %d/%d joint msgs "
                "(check %s and %s)" % (len(odoms), n_samples, len(joints), n_samples,
                                       odom_topic, joint_topic))
        poll.sleep()
    odom_sub.unregister()
    joint_sub.unregister()

    pos = np.mean([[m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z]
                   for m in odoms], axis=0)
    rpys = np.array([tft.euler_from_quaternion(
        [m.pose.pose.orientation.x, m.pose.pose.orientation.y,
         m.pose.pose.orientation.z, m.pose.pose.orientation.w]) for m in odoms])
    # average angles around the first sample to stay safe near +-pi
    rpy = _wrap(rpys[0] + np.mean(_wrap(rpys - rpys[0]), axis=0))

    joint_pos = np.zeros(len(JOINT_NAMES))
    for j, name in enumerate(JOINT_NAMES):
        vals = [m.position[m.name.index(name)] for m in joints if name in m.name]
        if not vals:
            raise RuntimeError(f"Joint '{name}' not found in {joint_topic} messages")
        joint_pos[j] = np.mean(vals)

    rospy.loginfo("Measured start state: pos=%s rpy=%s joint_pos=%s",
                  np.array2string(pos, precision=3, suppress_small=True),
                  np.array2string(rpy, precision=3, suppress_small=True),
                  np.array2string(joint_pos, precision=3, suppress_small=True))
    return {
        'pos': pos,
        'rpy': rpy,
        'joint_pos': joint_pos,
        'duration': 0.0,
        'vel': np.zeros(3),
        'omega': np.zeros(3),
        'joint_vel': np.zeros(len(JOINT_NAMES)),
    }


def build_trajectory(keyframes):
    n_seg = len(keyframes) - 1
    assert n_seg >= 1, "Need at least 2 keyframes"

    # Unwrap RPY across keyframes to avoid discontinuities at ±π
    rpys = np.array([kf['rpy'] for kf in keyframes])
    for i in range(1, len(rpys)):
        rpys[i] = rpys[i - 1] + _wrap(rpys[i] - rpys[i - 1])

    states, vels, times = [], [], []
    t0 = 0.0

    for seg in range(n_seg):
        kf0, kf1 = keyframes[seg], keyframes[seg + 1]
        T = kf1['duration']
        assert T > 0, f"Keyframe {seg + 1} duration must be > 0"

        p0 = np.r_[kf0['pos'], rpys[seg],     kf0['joint_pos']]
        p1 = np.r_[kf1['pos'], rpys[seg + 1], kf1['joint_pos']]
        m0 = np.r_[kf0['vel'], kf0['omega'],  kf0['joint_vel']]
        m1 = np.r_[kf1['vel'], kf1['omega'],  kf1['joint_vel']]

        n_steps = max(1, int(round(T * FREQ)))
        for k in range(n_steps):
            t = k * DT
            s = _hermite(p0, p1, m0, m1, T, t)
            s[3:6] = _wrap(s[3:6])
            states.append(s)
            vels.append(_hermite_dot(p0, p1, m0, m1, T, t))
            times.append(t0 + t)
        t0 += T

    # Append final keyframe
    states.append(np.r_[keyframes[-1]['pos'], _wrap(rpys[-1]), keyframes[-1]['joint_pos']])
    vels.append(np.zeros(12))
    times.append(t0)

    return np.array(states), np.array(vels), np.array(times)


def _make_msg(state, vel):
    msg = FullStateTarget()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = 'world'

    odom = msg.root_state
    odom.header = msg.header
    odom.child_frame_id = 'fc'

    p = odom.pose.pose
    p.position.x, p.position.y, p.position.z = state[0], state[1], state[2]
    q = tft.quaternion_from_euler(state[3], state[4], state[5])
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q

    tw = odom.twist.twist
    tw.linear.x,  tw.linear.y,  tw.linear.z  = vel[0], vel[1], vel[2]
    tw.angular.x, tw.angular.y, tw.angular.z = vel[3], vel[4], vel[5]

    msg.joint_state.name     = JOINT_NAMES
    msg.joint_state.position = state[6:12].tolist()
    msg.joint_state.velocity = vel[6:12].tolist()
    return msg


def build_landing_trajectory(start_state, start_vel, target_z, duration):
    duration = max(float(duration), DT)
    n_steps = max(1, int(round(duration * FREQ)))
    start_state = np.asarray(start_state, dtype=float).copy()
    start_vel = np.asarray(start_vel, dtype=float).copy()
    target_z = float(target_z)

    # Keep x/y, attitude, and yaw joints at the current target keyframe while
    # descending. Fold the pitch joints to zero for a level landing shape.
    vz0 = min(float(start_vel[2]), 0.0)
    landing_states, landing_vels, landing_times = [], [], []
    for k in range(n_steps + 1):
        t = min(k * DT, duration)
        state = start_state.copy()
        vel = np.zeros_like(start_vel)
        state[2] = _hermite(start_state[2], target_z, vz0, 0.0, duration, t)
        vel[2] = _hermite_dot(start_state[2], target_z, vz0, 0.0, duration, t)
        for joint_index, name in enumerate(JOINT_NAMES):
            if 'pitch' not in name:
                continue
            state_index = 6 + joint_index
            state[state_index] = _hermite(
                start_state[state_index], 0.0, start_vel[state_index], 0.0, duration, t)
            vel[state_index] = _hermite_dot(
                start_state[state_index], 0.0, start_vel[state_index], 0.0, duration, t)
        landing_states.append(state)
        landing_vels.append(vel)
        landing_times.append(t)
    return np.array(landing_states), np.array(landing_vels), np.array(landing_times)


def _format_state(states, times, index):
    return (
        "step=%d/%d t=%.2fs pos=%s rpy=%s joint_pos=%s"
        % (
            index + 1,
            len(states),
            times[index],
            np.array2string(states[index, 0:3], precision=3, suppress_small=True),
            np.array2string(states[index, 3:6], precision=3, suppress_small=True),
            np.array2string(states[index, 6:12], precision=3, suppress_small=True),
        )
    )


def main():
    rospy.init_node('keyframe_tracking')

    args = rospy.myargv(argv=sys.argv)
    yaml_path = rospy.get_param('~config', args[1] if len(args) > 1 else None)
    if yaml_path is None:
        rospy.logerr("Provide config: _config:=<path.yaml> or pass as first argument")
        return

    yaml_path = os.path.abspath(yaml_path)
    npz_path  = os.path.splitext(yaml_path)[0] + '.npz'

    rospy.loginfo(f"Config: {yaml_path}")
    keyframes = load_keyframes(yaml_path)
    rospy.loginfo(f"Loaded {len(keyframes)} keyframes")

    mode = rospy.get_param('~mode', 'real')
    if mode not in ('real', 'sim'):
        rospy.logerr("Unknown mode '%s' (use real or sim)", mode)
        return
    rospy.loginfo("Mode: %s", mode)

    robot_ns = rospy.get_param('~robot_ns', rospy.get_namespace().strip('/')) or 'dragon'
    ns = '/' + robot_ns.strip('/')
    rospy.loginfo("Robot namespace: %s", ns)

    if mode == 'real':
        odom_topic = rospy.get_param('~odom_topic', ns + '/uav/baselink/odom')
        joint_topic = rospy.get_param('~joint_state_topic', ns + '/joint_states')
        init_sample_num = int(rospy.get_param('~init_sample_num', 10))
        try:
            init_kf = collect_initial_keyframe(odom_topic, joint_topic, init_sample_num)
        except RuntimeError as e:
            rospy.logerr("%s", e)
            return
        # the measured state replaces the first YAML keyframe; the spline then reaches
        # the second YAML keyframe within that keyframe's own duration, as in sim
        keyframes[0] = init_kf
        rospy.loginfo("Replaced keyframe 1 with measured state; reach keyframe 2 in %.1fs "
                      "after takeoff.", keyframes[1]['duration'])

    states, vels, times = build_trajectory(keyframes)
    rospy.loginfo(f"Trajectory: {len(times)} steps, {times[-1]:.2f}s total")
    land_height = float(rospy.get_param('~land_height', 0.15))
    land_duration = float(rospy.get_param('~land_duration', 10.0))

    target_topic = rospy.get_param('~target_topic', ns + '/full_state_target')
    command_ns = rospy.get_param('~command_ns', ns + '/teleop_command').rstrip('/')
    pub = rospy.Publisher(target_topic, FullStateTarget, queue_size=10)

    state = {
        'started': False,
        'paused': False,
        'pause_reason': '',
        'target_index': 0,
        'landing': False,
        'landed': False,
        'landing_index': 0,
        'landing_states': None,
        'landing_vels': None,
        'landing_times': None,
        'land_done_logged': False,
    }

    def start_tracking_cb(_msg):
        state['started'] = True
        state['paused'] = False
        state['pause_reason'] = ''
        state['landing'] = False
        state['landed'] = False
        rospy.loginfo("Received takeoff command: start keyframe tracking.")

    def pause_tracking(reason):
        state['paused'] = True
        state['pause_reason'] = reason
        state['landing'] = False
        state['landed'] = False
        rospy.loginfo("Received %s command: pause target-state publishing.", reason)

    def resume_tracking_cb(_msg):
        state['paused'] = False
        state['pause_reason'] = ''
        rospy.loginfo("Received disable_pause_tracking command: resume target-state publishing.")

    def land_cb(_msg):
        if state['landing']:
            rospy.loginfo("Landing target trajectory is already active.")
            return
        if state['landed'] and state['landing_states'] is not None:
            start_state = state['landing_states'][-1]
            start_vel = state['landing_vels'][-1]
        else:
            current_index = state['target_index'] - 1 if state['started'] else 0
            current_index = min(max(current_index, 0), len(states) - 1)
            start_state = states[current_index]
            start_vel = vels[current_index]

        landing_states, landing_vels, landing_times = build_landing_trajectory(
            start_state, start_vel, land_height, land_duration)
        state['landing_states'] = landing_states
        state['landing_vels'] = landing_vels
        state['landing_times'] = landing_times
        state['landing_index'] = 0
        state['landing'] = True
        state['landed'] = False
        state['land_done_logged'] = False
        state['started'] = True
        state['paused'] = False
        state['pause_reason'] = ''
        rospy.loginfo("Received land_tracking command: descend target from z=%.3f to z=%.3f in %.1fs.",
                      start_state[2], land_height, land_duration)

    rospy.Subscriber(command_ns + '/start_tracking', Empty, start_tracking_cb, queue_size=1)
    rospy.Subscriber(command_ns + '/pause_tracking', Empty,
                     lambda _msg: pause_tracking('pause_tracking'), queue_size=1)
    rospy.Subscriber(command_ns + '/disable_pause_tracking', Empty, resume_tracking_cb, queue_size=1)
    rospy.Subscriber(command_ns + '/land_tracking', Empty, land_cb, queue_size=1)
    rospy.Subscriber(command_ns + '/halt', Empty, lambda _msg: pause_tracking('halt'), queue_size=1)
    rospy.Subscriber(command_ns + '/force_landing', Empty,
                     lambda _msg: pause_tracking('force_landing'), queue_size=1)

    rospy.loginfo("Target topic: %s", target_topic)
    rospy.loginfo("Command namespace: %s", command_ns)
    rospy.loginfo("Holding first keyframe until %s/takeoff is received.", command_ns)
    rospy.sleep(0.5)

    rate = rospy.Rate(FREQ)
    rospy.loginfo("Keyframe target publisher started.")
    last_state_log_time = rospy.Time(0)
    try:
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if (now - last_state_log_time).to_sec() >= 5.0:
                log_index = min(max(state['target_index'], 0), len(states) - 1)
                if state['paused']:
                    rospy.loginfo("Target-state publishing paused by %s at %s",
                                  state['pause_reason'], _format_state(states, times, log_index))
                elif state['landing']:
                    landing_index = min(max(state['landing_index'], 0),
                                        len(state['landing_states']) - 1)
                    rospy.loginfo("Landing target state %s",
                                  _format_state(state['landing_states'],
                                                state['landing_times'],
                                                landing_index))
                elif state['landed']:
                    rospy.loginfo("Holding landed target %s",
                                  _format_state(state['landing_states'],
                                                state['landing_times'],
                                                len(state['landing_states']) - 1))
                elif not state['started']:
                    rospy.loginfo("Waiting for takeoff; holding %s", _format_state(states, times, 0))
                else:
                    rospy.loginfo("Current target state %s", _format_state(states, times, log_index))
                last_state_log_time = now

            if state['paused']:
                rate.sleep()
                continue

            if state['landing']:
                landing_index = min(state['landing_index'], len(state['landing_states']) - 1)
                pub.publish(_make_msg(state['landing_states'][landing_index],
                                      state['landing_vels'][landing_index]))
                if landing_index >= len(state['landing_states']) - 1:
                    state['landing'] = False
                    state['landed'] = True
                    if not state['land_done_logged']:
                        rospy.loginfo("Finished landing target descent; holding z=%.3f.",
                                      state['landing_states'][-1, 2])
                        state['land_done_logged'] = True
                else:
                    state['landing_index'] += 1
                rate.sleep()
                continue

            if state['landed']:
                pub.publish(_make_msg(state['landing_states'][-1], state['landing_vels'][-1]))
                rate.sleep()
                continue

            if not state['started']:
                pub.publish(_make_msg(states[0], vels[0]))
                rate.sleep()
                continue

            pub.publish(_make_msg(states[state['target_index']], vels[state['target_index']]))
            state['target_index'] += 1
            if state['target_index'] >= len(states):
                rospy.loginfo("Finished keyframe trajectory.")
                break
            rate.sleep()
    finally:
        base_quat_w = np.array([tft.quaternion_from_euler(*rpy) for rpy in states[:, 3:6]])
        np.savez(npz_path,
                 base_pos_w    = states[:, 0:3],
                 base_quat_w   = base_quat_w,        # [x, y, z, w]
                 joint_pos     = states[:, 6:12],
                 joint_names   = np.array(JOINT_NAMES),
                 base_lin_vel_w= vels[:, 0:3],
                 base_ang_vel_w= vels[:, 3:6],
                 framerate     = np.float64(FREQ))
        rospy.loginfo(f"Saved: {npz_path}")


if __name__ == '__main__':
    main()
