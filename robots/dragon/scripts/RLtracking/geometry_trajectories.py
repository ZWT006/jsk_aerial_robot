#!/usr/bin/env python3
"""Deterministic geometry trajectories for DRAGON RL tracking."""

import math

import numpy as np
import tf.transformations as tft


DEFAULT_JOINT_NAMES = (
    'joint1_pitch',
    'joint1_yaw',
    'joint2_pitch',
    'joint2_yaw',
    'joint3_pitch',
    'joint3_yaw',
)

_EPS = 1e-9


def _normalize_quat(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < _EPS:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def _quat_to_matrix(q):
    x, y, z, w = _normalize_quat(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ])


def _rot_y(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def _rot_z(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _euler_from_matrix(rot):
    mat = np.eye(4)
    mat[:3, :3] = rot
    return np.array(tft.euler_from_matrix(mat))


def _finite_difference(states, times):
    velocities = np.zeros_like(states)
    if len(states) < 2:
        return velocities

    continuous = states.copy()
    continuous[:, 3:6] = np.unwrap(continuous[:, 3:6], axis=0)
    edge_order = 2 if len(states) > 2 else 1
    velocities[:] = np.gradient(continuous, times, axis=0, edge_order=edge_order)
    return velocities


class DragonDanceTrajectory:
    """Fixed dragon dance FullStateTarget trajectory in the DRAGON fc frame."""

    PERIOD = 60.0
    CENTER = np.array([0.0, -0.2])
    AXES = np.array([1.8, 1.2])
    Z_OFFSET = 1.25
    Z_AMPLITUDE = 0.25
    Z_CYCLES = 3.0
    LINK_NUM = 4
    LINK_LENGTH = 0.455
    INTER_JOINT_X_OFFSET = 0.02575
    FC_OFFSET_IN_LINK2 = np.array([0.3245, -0.0010, 0.0280])
    LOOKUP_SIZE = 8192

    def __init__(self, spec):
        self.start_phase = float(spec.get('start_phase', 0.0))
        self.cycles = float(spec.get('cycles', 1.0))
        self.duration = float(spec.get(
            'motion_duration',
            spec.get('duration', self.PERIOD * max(abs(self.cycles), 1.0)),
        ))
        self.link_length = float(spec.get('link_length', self.LINK_LENGTH))
        self.inter_joint_x_offset = float(spec.get(
            'inter_joint_x_offset', self.INTER_JOINT_X_OFFSET))
        self.fc_offset_in_link2 = np.array(
            spec.get('fc_offset_in_link2', self.FC_OFFSET_IN_LINK2), dtype=float)
        if self.duration <= 0.0:
            raise ValueError("dragon_dance geometry duration must be > 0")
        if self.link_length <= 0.0:
            raise ValueError("dragon_dance link_length must be > 0")
        if self.fc_offset_in_link2.shape != (3,):
            raise ValueError("dragon_dance fc_offset_in_link2 must have 3 values")

        self._lookup_phase = np.linspace(0.0, 1.0, self.LOOKUP_SIZE + 1)
        self._lookup_pos, _ = self._tail_pose_at_phase(self._lookup_phase)
        segment_length = np.linalg.norm(np.diff(self._lookup_pos, axis=0), axis=1)
        self._lookup_arc = np.r_[0.0, np.cumsum(segment_length)]
        self._total_arc_length = self._lookup_arc[-1]
        if self._total_arc_length <= _EPS:
            raise ValueError("dragon_dance path arc length is too small")

    def sample(self, freq):
        n_steps = max(1, int(round(self.duration * float(freq))))
        times = np.arange(n_steps + 1, dtype=float) / float(freq)
        times[-1] = self.duration
        phases = self.start_phase + self.cycles * (times / self.duration)

        tail_pos, root_quat = self._tail_pose_at_phase(phases)
        target_positions = self._target_positions(phases)

        states = np.zeros((n_steps + 1, 6 + len(DEFAULT_JOINT_NAMES)), dtype=float)
        for i, quat in enumerate(root_quat):
            root_rotation = _quat_to_matrix(quat)
            joints = self._joint_angles(target_positions[i], tail_pos[i], root_rotation)
            fc_pos, fc_rotation = self._fc_pose_from_tail(tail_pos[i], root_rotation, joints)
            states[i, 0:3] = fc_pos
            states[i, 3:6] = _euler_from_matrix(fc_rotation)
            states[i, 6:12] = joints

        velocities = _finite_difference(states, times)
        states[:, 3:6] = (states[:, 3:6] + math.pi) % (2.0 * math.pi) - math.pi
        return states, velocities, times

    def start_keyframe(self):
        states, velocities, _ = self.sample(1.0)
        return {
            'pos': states[0, 0:3],
            'rpy': states[0, 3:6],
            'joint_pos': states[0, 6:12],
            'duration': 0.0,
            'vel': velocities[0, 0:3],
            'omega': velocities[0, 3:6],
            'joint_vel': velocities[0, 6:12],
        }

    def _tail_pose_at_phase(self, phases):
        phases = np.asarray(phases, dtype=float)
        theta = 0.5 * math.pi - 2.0 * math.pi * phases
        z_phase = 2.0 * math.pi * self.Z_CYCLES * phases - 0.5 * math.pi

        pos = np.empty((phases.size, 3), dtype=float)
        pos[:, 0] = self.CENTER[0] + self.AXES[0] * np.cos(theta)
        pos[:, 1] = self.CENTER[1] + self.AXES[1] * np.sin(theta)
        pos[:, 2] = self.Z_OFFSET + self.Z_AMPLITUDE * np.sin(z_phase)

        dx_dphase = 2.0 * math.pi * self.AXES[0] * np.sin(theta)
        dy_dphase = -2.0 * math.pi * self.AXES[1] * np.cos(theta)
        dz_dphase = 2.0 * math.pi * self.Z_CYCLES * self.Z_AMPLITUDE * np.cos(z_phase)
        yaw = np.arctan2(dy_dphase, dx_dphase)
        pitch = -np.arctan2(dz_dphase, np.hypot(dx_dphase, dy_dphase))

        half_pitch = 0.5 * pitch
        half_yaw = 0.5 * yaw
        q_flu = np.empty((phases.size, 4), dtype=float)
        q_flu[:, 0] = -np.sin(half_pitch) * np.sin(half_yaw)
        q_flu[:, 1] = np.sin(half_pitch) * np.cos(half_yaw)
        q_flu[:, 2] = np.cos(half_pitch) * np.sin(half_yaw)
        q_flu[:, 3] = np.cos(half_pitch) * np.cos(half_yaw)

        # Reference demo converts FLU tail orientation with yaw(pi) * q_flu.
        q_root = np.empty_like(q_flu)
        q_root[:, 0] = -q_flu[:, 1]
        q_root[:, 1] = q_flu[:, 0]
        q_root[:, 2] = q_flu[:, 3]
        q_root[:, 3] = -q_flu[:, 2]
        q_root /= np.maximum(np.linalg.norm(q_root, axis=1, keepdims=True), _EPS)
        return pos, q_root

    def _arc_at_phase(self, phases):
        phase_mod = np.mod(phases, 1.0)
        return np.interp(phase_mod, self._lookup_phase, self._lookup_arc)

    def _phase_at_arc(self, arc_positions):
        arc_mod = np.mod(arc_positions, self._total_arc_length)
        return np.interp(arc_mod, self._lookup_arc, self._lookup_phase)

    def _target_positions(self, phases):
        current_arc = self._arc_at_phase(phases)
        offsets = self.link_length * np.arange(1, self.LINK_NUM, dtype=float)
        target_arc = current_arc[:, None] - offsets[None, :]
        target_phase = self._phase_at_arc(target_arc.reshape(-1)).reshape(len(phases), -1)
        target_pos, _ = self._tail_pose_at_phase(target_phase.reshape(-1))
        return target_pos.reshape(len(phases), self.LINK_NUM - 1, 3)

    def _joint_angles(self, target_positions, tail_position, root_rotation):
        joints = np.zeros(len(DEFAULT_JOINT_NAMES), dtype=float)
        current_rotation = root_rotation
        current_tail = tail_position.copy()
        for i, target in enumerate(target_positions):
            segment = target - current_tail
            segment_norm = np.linalg.norm(segment)
            if segment_norm < _EPS:
                continue

            local_direction = current_rotation.T.dot(segment / segment_norm)
            local_norm = np.linalg.norm(local_direction)
            if local_norm < _EPS:
                current_tail = target
                continue

            local_direction /= local_norm
            pitch_index = 2 * i
            yaw_index = pitch_index + 1
            yaw = math.asin(float(np.clip(local_direction[1], -1.0, 1.0)))
            xz_norm = math.hypot(local_direction[0], local_direction[2])
            pitch = 0.0 if xz_norm < _EPS else math.atan2(-local_direction[2], local_direction[0])
            joints[pitch_index] = pitch
            joints[yaw_index] = yaw
            current_rotation = current_rotation.dot(_rot_y(pitch)).dot(_rot_z(yaw))
            current_tail = target
        return joints

    def _fc_pose_from_tail(self, tail_position, root_rotation, joints):
        pitch = joints[0]
        yaw = joints[1]
        root_to_inter = _rot_y(pitch)
        inter_to_link2 = _rot_z(yaw)
        link2_rotation = root_rotation.dot(root_to_inter).dot(inter_to_link2)
        tail_to_fc = root_to_inter.dot(
            np.array([2.0 * self.inter_joint_x_offset, 0.0, 0.0]) +
            inter_to_link2.dot(self.fc_offset_in_link2)
        )
        fc_position = tail_position + root_rotation.dot(tail_to_fc)
        return fc_position, link2_rotation


GEOMETRY_TRAJECTORIES = {
    'dragon_dance': DragonDanceTrajectory,
}


def make_geometry_trajectory(spec):
    if not isinstance(spec, dict):
        raise ValueError("geometry spec must be a mapping")

    geometry_type = spec.get('type')
    if geometry_type not in GEOMETRY_TRAJECTORIES:
        known = ', '.join(sorted(GEOMETRY_TRAJECTORIES))
        raise ValueError("Unknown geometry type '%s' (known: %s)" % (geometry_type, known))
    return GEOMETRY_TRAJECTORIES[geometry_type](spec)
