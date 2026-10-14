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
        omega: [dr, dp, dy]    # optional: rpy rate at this keyframe (spline tangent);
                               # output twist / base_ang_vel_w are converted to world-frame omega
        joint_vel: [...]       # optional: joint velocity at this keyframe
      - duration: 5.0          # transition time from the previous keyframe
        geometry:
          type: dragon_dance
          start_phase: 0.0
          cycles: 1
          duration: 60.0       # geometry motion duration, or use motion_duration
"""

import sys
import os
import yaml
import numpy as np
import rospy
from aerial_robot_msgs.msg import FullStateTarget
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, Empty
from visualization_msgs.msg import Marker, MarkerArray
import tf.transformations as tft

from geometry_trajectories import make_geometry_trajectory

FREQ = 40.0
DT = 1.0 / FREQ
JOINT_NAMES = ['joint1_pitch', 'joint1_yaw', 'joint2_pitch', 'joint2_yaw', 'joint3_pitch', 'joint3_yaw']
LINK_LENGTH = 0.474
INTER_JOINT_X_OFFSET = 0.02575
FC_OFFSET_IN_LINK2 = np.array([0.3245, -0.0010, 0.0280])


def _wrap(a):
    return (np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def _normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def _normalize_quat(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def _rot_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def _rot_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _compose_pose(pos_a, rot_a, pos_b, rot_b):
    return pos_a + rot_a.dot(pos_b), rot_a.dot(rot_b)


def _minimal_rotation_quat(src_vec, dst_vec):
    src = _normalize(src_vec)
    dst = _normalize(dst_vec)
    cross = np.cross(src, dst)
    cross_norm = np.linalg.norm(cross)
    dot = np.clip(np.dot(src, dst), -1.0, 1.0)
    if cross_norm < 1e-12:
        if dot > 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3), 0.0
        axis = _normalize(np.cross(src, np.array([1.0, 0.0, 0.0])))
        if np.linalg.norm(axis) < 1e-12:
            axis = _normalize(np.cross(src, np.array([0.0, 1.0, 0.0])))
        angle = np.pi
        return _normalize_quat(tft.quaternion_about_axis(angle, axis)), axis, angle

    axis = cross / cross_norm
    angle = np.arctan2(cross_norm, dot)
    return _normalize_quat(tft.quaternion_about_axis(angle, axis)), axis, angle


def _level_orientation_quat(start_rpy):
    q_start = _normalize_quat(tft.quaternion_from_euler(*start_rpy))
    rot = tft.quaternion_matrix(q_start)[:3, :3]
    body_z_w = rot.dot(np.array([0.0, 0.0, 1.0]))
    q_level_delta, axis, angle = _minimal_rotation_quat(body_z_w, np.array([0.0, 0.0, 1.0]))
    q_target = _normalize_quat(tft.quaternion_multiply(q_level_delta, q_start))
    if np.dot(q_start, q_target) < 0.0:
        q_target = -q_target
    return q_start, q_target, axis, angle


def _hermite(p0, p1, m0, m1, T, t):
    s = t / T
    return ((2*s**3 - 3*s**2 + 1)*p0 + (s**3 - 2*s**2 + s)*T*m0 +
            (-2*s**3 + 3*s**2)*p1 + (s**3 - s**2)*T*m1)


def _hermite_dot(p0, p1, m0, m1, T, t):
    s = t / T
    return (((6*s**2 - 6*s) / T)*p0 + (3*s**2 - 4*s + 1)*m0 +
            ((6*s - 6*s**2) / T)*p1 + (3*s**2 - 2*s)*m1)


def _rpy_rate_to_world_omega(rpy, rpy_dot):
    """Map rpy rates to world-frame angular velocity for R = Rz(y) Ry(p) Rx(r) (tft 'sxyz')."""
    rpy = np.atleast_2d(rpy)
    rpy_dot = np.atleast_2d(rpy_dot)
    p, y = rpy[:, 1], rpy[:, 2]
    dr, dp, dy = rpy_dot[:, 0], rpy_dot[:, 1], rpy_dot[:, 2]
    cp, sp, cy, sy = np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.stack([cy*cp*dr - sy*dp,
                     sy*cp*dr + cy*dp,
                     -sp*dr + dy], axis=-1)


def _as_array(value, length, name):
    arr = np.array(value, dtype=float)
    if arr.shape != (length,):
        raise ValueError("%s must have %d values" % (name, length))
    return arr


def _is_geometry_item(item):
    return 'geometry' in item


def _normalize_keyframe(kf):
    return {
        'pos':       _as_array(kf['pos'],       3, 'pos'),
        'rpy':       _as_array(kf['rpy'],       3, 'rpy'),
        'joint_pos': _as_array(kf['joint_pos'], len(JOINT_NAMES), 'joint_pos'),
        'duration':  float(kf.get('duration', 0.0)),
        'vel':       _as_array(kf.get('vel',       [0., 0., 0.]), 3, 'vel'),
        'omega':     _as_array(kf.get('omega',     [0., 0., 0.]), 3, 'omega'),
        'joint_vel': _as_array(kf.get('joint_vel', [0.] * len(JOINT_NAMES)),
                               len(JOINT_NAMES), 'joint_vel'),
    }


def _normalize_trajectory_item(item):
    if _is_geometry_item(item):
        if item['geometry'] is None:
            raise ValueError("geometry item must contain a geometry spec")
        return {
            'geometry': dict(item['geometry']),
            'duration': float(item.get('transition_duration', item.get('duration', 0.0))),
        }
    return _normalize_keyframe(item)


def load_keyframes(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return [_normalize_trajectory_item(item) for item in data['keyframes']]


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


def _state_from_keyframe(kf):
    return np.r_[kf['pos'], kf['rpy'], kf['joint_pos']]


def _vel_from_keyframe(kf):
    return np.r_[kf['vel'], kf['omega'], kf['joint_vel']]


def _keyframe_from_state(state, vel=None):
    state = np.asarray(state, dtype=float)
    if vel is None:
        vel = np.zeros_like(state)
    vel = np.asarray(vel, dtype=float)
    return {
        'pos': state[0:3].copy(),
        'rpy': _wrap(state[3:6]),
        'joint_pos': state[6:12].copy(),
        'duration': 0.0,
        'vel': vel[0:3].copy(),
        'omega': vel[3:6].copy(),
        'joint_vel': vel[6:12].copy(),
    }


def _append_hermite_segment(states, vels, times, t0, kf0, kf1, duration, segment_label):
    T = float(duration)
    if T <= 0.0:
        raise ValueError("%s duration must be > 0" % segment_label)

    p0 = _state_from_keyframe(kf0)
    p1 = _state_from_keyframe(kf1)
    p1[3:6] = p0[3:6] + _wrap(p1[3:6] - p0[3:6])
    m0 = _vel_from_keyframe(kf0)
    m1 = _vel_from_keyframe(kf1)

    n_steps = max(1, int(round(T * FREQ)))
    for k in range(n_steps):
        t = k * DT
        state = _hermite(p0, p1, m0, m1, T, t)
        state[3:6] = _wrap(state[3:6])
        states.append(state)
        vels.append(_hermite_dot(p0, p1, m0, m1, T, t))
        times.append(t0 + t)

    final_state = p1.copy()
    final_state[3:6] = _wrap(final_state[3:6])
    return _keyframe_from_state(final_state, m1), t0 + T


def build_trajectory(items):
    assert len(items) >= 2, "Need at least 2 trajectory entries"
    if _is_geometry_item(items[0]):
        raise ValueError("First trajectory entry must be a normal keyframe")

    states, vels, times = [], [], []
    current_kf = items[0]
    t0 = 0.0

    for index, item in enumerate(items[1:], start=1):
        if _is_geometry_item(item):
            geometry = make_geometry_trajectory(item['geometry'])
            geom_states, geom_vels, geom_times = geometry.sample(FREQ)
            geom_start = _keyframe_from_state(geom_states[0], geom_vels[0])
            transition_duration = float(item.get('duration', 0.0))

            if transition_duration > 0.0:
                current_kf, t0 = _append_hermite_segment(
                    states, vels, times, t0, current_kf, geom_start,
                    transition_duration, "Geometry entry %d transition" % index)
                start_sample = 0
            elif transition_duration < 0.0:
                raise ValueError("Geometry entry %d transition duration must be >= 0" % index)
            else:
                if not states:
                    states.append(_state_from_keyframe(current_kf))
                    vels.append(_vel_from_keyframe(current_kf))
                    times.append(t0)
                    start_sample = 1
                else:
                    start_sample = 0

            for k in range(start_sample, len(geom_times) - 1):
                states.append(geom_states[k])
                vels.append(geom_vels[k])
                times.append(t0 + geom_times[k])
            t0 += geom_times[-1]
            current_kf = _keyframe_from_state(geom_states[-1], geom_vels[-1])
        else:
            current_kf, t0 = _append_hermite_segment(
                states, vels, times, t0, current_kf, item,
                item['duration'], "Keyframe %d" % index)

    states.append(_state_from_keyframe(current_kf))
    vels.append(_vel_from_keyframe(current_kf))
    times.append(t0)

    # Splines are built in rpy-rate space; convert to world-frame angular velocity
    # so published twist.angular and saved base_ang_vel_w are true omega_w.
    states, vels = np.array(states), np.array(vels)
    vels[:, 3:6] = _rpy_rate_to_world_omega(states[:, 3:6], vels[:, 3:6])
    return states, vels, np.array(times)


def _make_msg(state, vel, orientation_quat=None):
    msg = FullStateTarget()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = 'world'

    odom = msg.root_state
    odom.header = msg.header
    odom.child_frame_id = 'fc'

    p = odom.pose.pose
    p.position.x, p.position.y, p.position.z = state[0], state[1], state[2]
    q = (_normalize_quat(orientation_quat) if orientation_quat is not None
         else tft.quaternion_from_euler(state[3], state[4], state[5]))
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q

    tw = odom.twist.twist
    tw.linear.x,  tw.linear.y,  tw.linear.z  = vel[0], vel[1], vel[2]
    tw.angular.x, tw.angular.y, tw.angular.z = vel[3], vel[4], vel[5]

    msg.joint_state.name     = JOINT_NAMES
    msg.joint_state.position = state[6:12].tolist()
    msg.joint_state.velocity = vel[6:12].tolist()
    return msg


def _point_msg(pos):
    point = Point()
    point.x, point.y, point.z = pos
    return point


def _color(r, g, b, a):
    color = ColorRGBA()
    color.r = r
    color.g = g
    color.b = b
    color.a = a
    return color


def _state_quat(state, orientation_quat=None):
    if orientation_quat is not None:
        return _normalize_quat(orientation_quat)
    return _normalize_quat(tft.quaternion_from_euler(state[3], state[4], state[5]))


def _target_visual_points(state, orientation_quat=None):
    fc_pos = np.asarray(state[0:3], dtype=float)
    fc_quat = _state_quat(state, orientation_quat)
    fc_rot = tft.quaternion_matrix(fc_quat)[:3, :3]
    joints = np.asarray(state[6:12], dtype=float)

    pitch1 = _rot_y(joints[0])
    yaw1 = _rot_z(joints[1])
    link1_to_fc_rot = pitch1.dot(yaw1)
    link1_to_fc_pos = (
        np.array([LINK_LENGTH, 0.0, 0.0]) +
        pitch1.dot(np.array([2.0 * INTER_JOINT_X_OFFSET, 0.0, 0.0]) +
                   yaw1.dot(FC_OFFSET_IN_LINK2))
    )

    link_pos = fc_pos - fc_rot.dot(link1_to_fc_rot.T).dot(link1_to_fc_pos)
    link_rot = fc_rot.dot(link1_to_fc_rot.T)
    first_link_head = link_pos.copy()

    for link_index in range(3):
        pitch = joints[2 * link_index]
        yaw = joints[2 * link_index + 1]
        link_pos, link_rot = _compose_pose(
            link_pos, link_rot, np.array([LINK_LENGTH, 0.0, 0.0]), _rot_y(pitch))
        link_pos, link_rot = _compose_pose(
            link_pos, link_rot,
            np.array([2.0 * INTER_JOINT_X_OFFSET, 0.0, 0.0]), _rot_z(yaw))

    last_link_tail = link_pos + link_rot.dot(np.array([LINK_LENGTH, 0.0, 0.0]))
    return fc_pos, fc_quat, fc_rot, first_link_head, last_link_tail


class TargetVisualizer:
    def __init__(self, marker_pub, frame_id='world'):
        self.marker_pub = marker_pub
        self.frame_id = frame_id

    def publish(self, state, orientation_quat=None, stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()

        fc_pos, fc_quat, fc_rot, first_head, last_tail = _target_visual_points(
            state, orientation_quat)

        markers = MarkerArray()
        markers.markers.extend([
            self._sphere(0, 'target_fc', fc_pos, stamp, 0.075, (0.0, 0.55, 1.0, 0.95)),
            self._sphere(1, 'target_link1_head', first_head, stamp, 0.075, (0.0, 1.0, 0.25, 0.95)),
            self._sphere(2, 'target_link4_tail', last_tail, stamp, 0.075, (1.0, 0.15, 0.85, 0.95)),
            self._axis_arrow(3, 'target_fc_axis_x', fc_pos, fc_rot[:, 0], stamp,
                             0.30, (1.0, 0.15, 0.1, 0.9)),
            self._axis_arrow(4, 'target_fc_axis_y', fc_pos, fc_rot[:, 1], stamp,
                             0.22, (0.0, 0.9, 0.15, 0.9)),
            self._axis_arrow(5, 'target_fc_axis_z', fc_pos, fc_rot[:, 2], stamp,
                             0.22, (0.2, 0.45, 1.0, 0.9)),
        ])
        self.marker_pub.publish(markers)

    def _base_marker(self, marker_id, ns, stamp):
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.frame_id
        marker.ns = ns
        marker.id = marker_id
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _sphere(self, marker_id, ns, pos, stamp, diameter, rgba):
        marker = self._base_marker(marker_id, ns, stamp)
        marker.type = Marker.SPHERE
        marker.pose.position = _point_msg(pos)
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = diameter
        marker.color = _color(*rgba)
        return marker

    def _axis_arrow(self, marker_id, ns, origin, axis, stamp, length, rgba):
        marker = self._base_marker(marker_id, ns, stamp)
        marker.type = Marker.ARROW
        axis = _normalize(axis)
        marker.points = [_point_msg(origin), _point_msg(origin + length * axis)]
        marker.scale.x = 0.025
        marker.scale.y = 0.055
        marker.scale.z = 0.055
        marker.color = _color(*rgba)
        return marker


def build_landing_trajectory(start_state, start_vel, target_z, duration):
    duration = max(float(duration), DT)
    n_steps = max(1, int(round(duration * FREQ)))
    start_state = np.asarray(start_state, dtype=float).copy()
    start_vel = np.asarray(start_vel, dtype=float).copy()
    target_z = float(target_z)

    # Keep x/y and yaw joints at the current target keyframe while descending.
    # Level the root orientation by the shortest rotation that makes body z
    # vertical, then interpolate directly in quaternion space.
    vz0 = min(float(start_vel[2]), 0.0)
    q_start, q_target, level_axis, level_angle = _level_orientation_quat(start_state[3:6])
    landing_states, landing_vels, landing_times, landing_quats = [], [], [], []
    for k in range(n_steps + 1):
        t = min(k * DT, duration)
        s = t / duration
        smooth_s = 3.0 * s**2 - 2.0 * s**3
        smooth_s_dot = (6.0 * s - 6.0 * s**2) / duration
        q = _normalize_quat(tft.quaternion_slerp(q_start, q_target, smooth_s))
        state = start_state.copy()
        vel = np.zeros_like(start_vel)
        state[2] = _hermite(start_state[2], target_z, vz0, 0.0, duration, t)
        vel[2] = _hermite_dot(start_state[2], target_z, vz0, 0.0, duration, t)
        state[3:6] = _wrap(tft.euler_from_quaternion(q))
        vel[3:6] = level_axis * level_angle * smooth_s_dot
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
        landing_quats.append(q)
    return (
        np.array(landing_states),
        np.array(landing_vels),
        np.array(landing_times),
        np.array(landing_quats),
    )


def _format_state(states, times, index, orientation_quats=None):
    rpy = (np.array(tft.euler_from_quaternion(orientation_quats[index]))
           if orientation_quats is not None else states[index, 3:6])
    return (
        "step=%d/%d t=%.2fs pos=%s rpy=%s joint_pos=%s"
        % (
            index + 1,
            len(states),
            times[index],
            np.array2string(states[index, 0:3], precision=3, suppress_small=True),
            np.array2string(rpy, precision=3, suppress_small=True),
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
    try:
        trajectory_items = load_keyframes(yaml_path)
    except (KeyError, TypeError, ValueError) as e:
        rospy.logerr("Failed to load trajectory config: %s", e)
        return
    rospy.loginfo(f"Loaded {len(trajectory_items)} trajectory entries")
    if len(trajectory_items) < 2:
        rospy.logerr("Need at least 2 trajectory entries")
        return
    if _is_geometry_item(trajectory_items[0]):
        rospy.logerr("The first trajectory entry must be a normal keyframe")
        return

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
        # the measured state replaces the first YAML keyframe; the trajectory then
        # reaches the second entry within that entry's own transition duration
        trajectory_items[0] = init_kf
        rospy.loginfo("Replaced keyframe 1 with measured state; reach entry 2 in %.1fs "
                      "after takeoff.", trajectory_items[1]['duration'])

    try:
        states, vels, times = build_trajectory(trajectory_items)
    except (AssertionError, ValueError) as e:
        rospy.logerr("Failed to build trajectory: %s", e)
        return
    rospy.loginfo(f"Trajectory: {len(times)} steps, {times[-1]:.2f}s total")
    land_height = float(rospy.get_param('~land_height', 0.15))
    land_duration = float(rospy.get_param('~land_duration', 10.0))

    target_topic = rospy.get_param('~target_topic', ns + '/full_state_target')
    command_ns = rospy.get_param('~command_ns', ns + '/teleop_command').rstrip('/')
    pub = rospy.Publisher(target_topic, FullStateTarget, queue_size=10)
    target_visualizer = None
    if (rospy.get_param('~visualize', False) or
            rospy.get_param('~visualize', False)):
        marker_topic = rospy.get_param(
            '~target_visualization_topic', ns + '/keyframe_tracking/target_visualization')
        vis_frame_id = rospy.get_param('~target_visualization_frame_id', 'world')
        target_visualizer = TargetVisualizer(
            rospy.Publisher(marker_topic, MarkerArray, queue_size=1),
            vis_frame_id,
        )
        rospy.loginfo("Target visualization enabled: markers=%s", marker_topic)

    def publish_target_state(target_state, target_vel, orientation_quat=None):
        msg = _make_msg(target_state, target_vel, orientation_quat)
        pub.publish(msg)
        if target_visualizer is not None:
            target_visualizer.publish(target_state, orientation_quat, msg.header.stamp)

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
        'landing_quats': None,
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

        landing_states, landing_vels, landing_times, landing_quats = build_landing_trajectory(
            start_state, start_vel, land_height, land_duration)
        state['landing_states'] = landing_states
        state['landing_vels'] = landing_vels
        state['landing_times'] = landing_times
        state['landing_quats'] = landing_quats
        state['landing_index'] = 0
        state['landing'] = True
        state['landed'] = False
        state['land_done_logged'] = False
        state['started'] = True
        state['paused'] = False
        state['pause_reason'] = ''
        rospy.loginfo("Received land_tracking command: descend target from z=%.3f to z=%.3f "
                      "and level body-z with minimal quaternion rotation in %.1fs.",
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
                                                landing_index,
                                                state['landing_quats']))
                elif state['landed']:
                    rospy.loginfo("Holding landed target %s",
                                  _format_state(state['landing_states'],
                                                state['landing_times'],
                                                len(state['landing_states']) - 1,
                                                state['landing_quats']))
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
                publish_target_state(state['landing_states'][landing_index],
                                     state['landing_vels'][landing_index],
                                     state['landing_quats'][landing_index])
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
                publish_target_state(state['landing_states'][-1],
                                     state['landing_vels'][-1],
                                     state['landing_quats'][-1])
                rate.sleep()
                continue

            if not state['started']:
                publish_target_state(states[0], vels[0])
                rate.sleep()
                continue

            publish_target_state(states[state['target_index']], vels[state['target_index']])
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
