#!/usr/bin/env python3
"""Keyframe trajectory tracking for DRAGON robot.

Reads keyframes from YAML, computes a cubic Hermite spline at 40 Hz,
publishes FullStateTarget with child_frame_id='fc' (baselink), and
saves the reference trajectory to a .npz file beside the YAML.

Usage:
    rosrun dragon keyframe_tracking.py <yaml_path>
    rosrun dragon keyframe_tracking.py _config:=<yaml_path>

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

    states, vels, times = build_trajectory(keyframes)
    rospy.loginfo(f"Trajectory: {len(times)} steps, {times[-1]:.2f}s total")

    pub = rospy.Publisher('/dragon/full_state_target', FullStateTarget, queue_size=10)
    rospy.sleep(0.5)

    rate = rospy.Rate(FREQ)
    rospy.loginfo("Tracking started...")
    try:
        for i in range(len(states)):
            if rospy.is_shutdown():
                break
            pub.publish(_make_msg(states[i], vels[i]))
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
