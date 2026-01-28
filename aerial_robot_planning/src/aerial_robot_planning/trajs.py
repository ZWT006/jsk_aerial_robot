"""
Created by li-jinjie on 24-1-5.
"""

import numpy as np
from typing import Tuple
import tf_conversions as tf
import rospy
from std_msgs.msg import Float32


class BaseTraj:
    def __init__(self, loop_num: int = np.inf) -> None:
        self.T = float()
        self.loop_num = loop_num
        self.use_constant_ref = False

        self.frame_id = "world"
        self.child_frame_id = "cog"

        self.t_total = None

    def check_finished(self, t: float) -> bool:
        if self.t_total is None:
            self.t_total = self.T * self.loop_num
        return t > self.t_total

    def get_frame_id(self) -> str:
        return self.frame_id

    def get_child_frame_id(self) -> str:
        return self.child_frame_id

    def get_3d_pt(self, t: float) -> Tuple[float, float, float, float, float, float, float, float, float]:
        x, y, z, vx, vy, vz, ax, ay, az = 0.0, 0.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        return x, y, z, vx, vy, vz, ax, ay, az

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        roll_rate, pitch_rate, yaw_rate = 0.0, 0.0, 0.0
        roll_acc, pitch_acc, yaw_acc = 0.0, 0.0, 0.0
        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc


class BaseTrajwFixedRotor(BaseTraj):
    def __init__(self, loop_num: int = np.inf) -> None:
        super().__init__(loop_num)
        self.hover_thrust = 7.5
        self.use_fix_rotor_flag = False

    def get_fixed_rotor(self, t: float):
        rotor_id = 0
        ft_fixed = 7.0
        alpha_fixed = 0.0
        return rotor_id, ft_fixed, alpha_fixed


class BaseTrajwSound(BaseTrajwFixedRotor):
    def __init__(self, loop_num: int = np.inf):
        super().__init__(loop_num)

        # thrust of each musical note
        self.note2thrust = {
            "g4": 7.75,
            "g4sharp": 8.83,
            "a4": 9.96,
            "a4sharp": 11.20,
            "b4": 12.54,
            "c5": 13.97,
            "c5sharp": 15.56,
            "d5": 17.22,
            "d5sharp": 18.87,
            "e5": 21.05,
        }

        self.freq_pub = rospy.Publisher("sound/fixed_rotor_frequency", Float32, queue_size=10)

    def thrust_to_freq(self, f):
        a = 0.0000161
        b = 0.0327
        c = -7.54 - f
        disc = b**2 - 4 * a * c
        if disc < 0:
            rospy.logwarn(f"Invalid thrust value f={f}, cannot compute frequency")
            return 0.0
        return (-b + np.sqrt(disc)) / (2 * a)

    def get_fixed_rotor(self, t: float):
        rotor_id = 0

        ft_fixed = self.compute_thrust_at_time(t)

        freq = self.thrust_to_freq(ft_fixed)
        self.freq_pub.publish(freq)

        alpha_fixed = np.arccos(self.hover_thrust / ft_fixed)
        self.use_fix_rotor_flag = True

        return rotor_id, ft_fixed, alpha_fixed

    def compute_thrust_at_time(self, t: float) -> float:
        if self.sequence is None:
            return self.hover_thrust

        t_mod = t % self.period
        idx = np.searchsorted(self.beat_times, t_mod, side="right") - 1

        if idx < 0 or idx >= len(self.sequence):
            return self.hover_thrust

        note, _ = self.sequence[idx]
        return self.note2thrust[note]


class CircleTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.r = 1  # radius in meters
        self.T = 10  # period in seconds
        self.omega = 2 * np.pi / self.T  # angular velocity

    def get_2d_pt(self, t: float) -> Tuple[float, float, float, float, float, float]:
        x = self.r * np.cos(self.omega * t) - 1.0
        y = self.r * np.sin(self.omega * t)
        vx = -self.r * self.omega * np.sin(self.omega * t)
        vy = self.r * self.omega * np.cos(self.omega * t)
        ax = -self.r * self.omega**2 * np.cos(self.omega * t)
        ay = -self.r * self.omega**2 * np.sin(self.omega * t)
        return x, y, vx, vy, ax, ay

    def get_3d_pt(self, t: float) -> Tuple[float, float, float, float, float, float, float, float, float]:
        x = self.r * np.cos(self.omega * t) - 1.0
        y = self.r * np.sin(self.omega * t)
        z = 0.5
        vx = -self.r * self.omega * np.sin(self.omega * t)
        vy = self.r * self.omega * np.cos(self.omega * t)
        vz = 0.0
        ax = -self.r * self.omega**2 * np.cos(self.omega * t)
        ay = -self.r * self.omega**2 * np.sin(self.omega * t)
        az = 0.0
        return x, y, z, vx, vy, vz, ax, ay, az


class LemniscateTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.a = 1.0  # parameter determining the size of the Lemniscate
        self.z_range = 0.3  # range of z
        self.T = 20  # period in seconds
        self.omega = 2 * np.pi / self.T  # angular velocity

    def get_2d_pt(self, t: float) -> Tuple[float, float, float, float, float, float]:
        t = t + self.T / 4  # shift the phase to make the trajectory start at the origin

        x = self.a * np.cos(self.omega * t)
        y = self.a * np.sin(2 * self.omega * t)

        vx = -self.a * self.omega * np.sin(self.omega * t)
        vy = 2 * self.a * self.omega * np.cos(2 * self.omega * t)

        ax = -self.a * self.omega**2 * np.cos(self.omega * t)
        ay = -4 * self.a * self.omega**2 * np.sin(2 * self.omega * t)

        return x, y, vx, vy, ax, ay

    def get_3d_pt(self, t: float) -> Tuple[float, float, float, float, float, float, float, float, float]:
        t = t + self.T / 4  # shift the phase to make the trajectory start at the origin

        x = self.a * np.cos(self.omega * t)
        y = self.a * np.sin(2 * self.omega * t) / 2
        z = self.z_range * np.sin(2 * self.omega * t + np.pi / 2) + 1.0

        vx = -self.a * self.omega * np.sin(self.omega * t)
        vy = 2 * self.a * self.omega * np.cos(2 * self.omega * t) / 2
        vz = 2 * self.z_range * self.omega * np.cos(2 * self.omega * t + np.pi)

        ax = -self.a * self.omega**2 * np.cos(self.omega * t)
        ay = -4 * self.a * self.omega**2 * np.sin(2 * self.omega * t) / 2
        az = -4 * self.z_range * self.omega**2 * np.sin(2 * self.omega * t + np.pi)

        return x, y, z, vx, vy, vz, ax, ay, az


class LemniscateTrajOmni(LemniscateTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.a_orientation = 0.5

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        t = t + self.T * 1 / 4

        roll = -2 * self.a_orientation * np.sin(2 * self.omega * t) / 2
        pitch = self.a_orientation * np.cos(self.omega * t)
        yaw = np.pi / 2 * np.sin(self.omega * t + np.pi) + np.pi / 2
        (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

        roll_rate = -2 * 2 * self.a_orientation * self.omega * np.cos(2 * self.omega * t) / 2
        pitch_rate = -self.a_orientation * self.omega * np.sin(self.omega * t)
        yaw_rate = np.pi / 2 * self.omega * np.cos(self.omega * t + np.pi / 2)

        roll_acc = -2 * -4 * self.a_orientation * self.omega**2 * np.sin(2 * self.omega * t) / 2
        pitch_acc = -self.a_orientation * self.omega**2 * np.cos(self.omega * t)
        yaw_acc = -np.pi / 2 * self.omega**2 * np.sin(self.omega * t + np.pi / 2)

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc


class SetPointTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.pos = np.array([0.0, 0.0, 0.7])
        self.vel = np.array([0.0, 0.0, 0.0])
        self.acc = np.array([0.0, 0.0, 0.0])

        self.att = np.array([0.0, 0.0, 0.0])
        self.att_rate = np.array([0.0, 0.0, 0.0])
        self.att_acc = np.array([0.0, 0.0, 0.0])

        self.t_converge = 8.0
        self.T = 4 * self.t_converge

    def get_3d_pt(self, t: float) -> Tuple[float, float, float, float, float, float, float, float, float]:
        x, y, z = self.pos
        vx, vy, vz = self.vel
        ax, ay, az = self.acc

        if 3 * self.t_converge > t > self.t_converge:
            x = 0.3
            y = 0.2
            z = 1.2

        if 3 * self.t_converge > t > 2 * self.t_converge:
            x = -0.3
            y = 0.0
            z = 1.0

        return x, y, z, vx, vy, vz, ax, ay, az

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        roll, pitch, yaw = self.att
        roll_rate, pitch_rate, yaw_rate = self.att_rate
        roll_acc, pitch_acc, yaw_acc = self.att_acc

        if 3 * self.t_converge > t > self.t_converge:
            roll = 0.5
            yaw = 0.3

        if 3 * self.t_converge > t > 2 * self.t_converge:
            pitch = 0.5
            yaw = -0.3

        (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc

class SetWaypointsTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.waypoints = [
        # want to show large tilt (back)
        #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],    # Hover at 1m
        #     [1.0, 0.0, 1.5, 45.0, 0.0, 0.0],    # Move X+
        #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        #     [0.0, 1.0, 1.5, 0.0, -25.0, 0.0],    # Move X+
        #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        #     [-1.0, 0.0, 1.5, -25.0, 0.0, 0.0],   # Move Y+, Up, Yaw 90
        #     [0.0, 0.0, 1.0, 0.0, 0.0, 90.0],
        #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Move X-, Yaw 180
        # ]

        # # want to show large tilt
        #     [0.0, 0.0, 0.8, 0.0, 0.0, 0.0],    # Hover at 1m
        #     [1.0, 0.0, 1.0, 45.0, 0.0, 0.0],    # Move X+
        #     [0.0, 1.0, 1.0, 0.0, -25.0, 0.0],    # Move X+
        #     [-1.0, 0.0, 1.0, -25.0, 0.0, 0.0],   # Move Y+, Up, Yaw 90
        #     [0.0, 0.0, 0.8, 0.0, 0.0, 90.0],
        #     # [0.0, 0.0, 0.8, 0.0, 0.0, 0.0],  # Move X-, Yaw 180
        # ]

        # # want to show large tilt
            [0.0, 0.0, 0.8, 0.0, 0.0, 0.0],    # Hover at 1m
            [0.0, 0.0, 0.8, 25.0, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.0, 0.0, 0.0],
            # [0.0, 0.0, 0.8, 0.0, 0.0, 0.0],  # Move X-, Yaw 180
        ]

        # # Challenge Waypoints
        #     [0.0, 0.0, 0.7, 0.0, 0.0, 0.0],    # Hover at 1m
        #     [1.0, 0.0, 1.0, 0.0, 45.0, 90.0],    # Move X+
        #     [0.0, 1.0, 1.0, 0.0, 90.0, 135.0],   # Move Y+, Up, Yaw 90
        #     [-1.0, 0.0, 1.0, 180.0, 0.0, 0.0],  # Move X-, Yaw 180
        #     [0.0, -1.0, 1.0, 0.0, 45.0, 90.0],
        #     # [0.0, -1.0, 1.0, 180.0, 0.0, 90.0],  # Move X-, Yaw 180
        #     # [1.0, 0.0, 1.0, 0.0, 45.0, 90.0],  # Move X-, Roll 180
        #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Move X-, Pitch 180
        #     [0.0, 0.0, 0.7, 0.0, 0.0, 0.0],    # Return origin
        # ]
        self.pos = np.array([0.0, 0.0, 0.7])
        self.vel = np.array([0.0, 0.0, 0.0])
        self.acc = np.array([0.0, 0.0, 0.0])

        self.att = np.array([0.0, 0.0, 0.0])
        self.att_rate = np.array([0.0, 0.0, 0.0])
        self.att_acc = np.array([0.0, 0.0, 0.0])
    
        self.t_converge = 8.0
        self.n_wp = len(self.waypoints)
        self.T = len(self.waypoints) * self.t_converge

    def _idx(self, t: float) -> int:
        if t < 0.0:
            return 0
        idx = int(t // self.t_converge)
        if idx >= self.n_wp:
            idx = self.n_wp - 1
        return idx

    def get_3d_pt(self, t: float) -> Tuple[float, float, float, float, float, float, float, float, float]:
        x, y, z = self.pos
        vx, vy, vz = self.vel
        ax, ay, az = self.acc

        if self.loop_num != np.inf:
            total_T = self.T * max(1, int(self.loop_num))
            if total_T > 0 and t >= total_T:
                # return final waypoint when finished
                idx = self.n_wp - 1
            else:
                if self.loop_num > 1 and self.T > 0:
                    t = t % self.T
                idx = self._idx(t)
        else:
            idx = self._idx(t)
        wp = self.waypoints[idx]
        x, y, z = float(wp[0]), float(wp[1]), float(wp[2])

        return x, y, z, vx, vy, vz, ax, ay, az

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        roll, pitch, yaw = self.att
        roll_rate, pitch_rate, yaw_rate = self.att_rate
        roll_acc, pitch_acc, yaw_acc = self.att_acc

        if self.loop_num != np.inf:
            total_T = self.T * max(1, int(self.loop_num))
            if total_T > 0 and t >= total_T:
                idx = self.n_wp - 1
            else:
                if self.loop_num > 1 and self.T > 0:
                    t = t % self.T
                idx = self._idx(t)
        else:
            idx = self._idx(t)
        wp = self.waypoints[idx]
        roll_deg, pitch_deg, yaw_deg = float(wp[3]), float(wp[4]), float(wp[5])
        # convert degrees to radians and get quaternion (qx,qy,qz,qw)
        qx, qy, qz, qw = tf.transformations.quaternion_from_euler(
            np.deg2rad(roll_deg), np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)
        )

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc

class SetRandomPointsTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.waypoints = [
        # want to show large tilt (back)
            [1.4,0.24,1.56,175.66,47.23,101.83],
            [0.24,1.42,0.78,-118.64,-52.17,69.76],
            [1.72,-0.61,1.61,-87.19,178.92,-176.47],
            [0.79,-0.22,0.75,-37.15,-99.3,123.56],
            [0.33,-1.78,0.76,-153.36,54.88,152.04],
            [1.26,-1.29,1.5,66.27,37.8,97.54],
            [1.52,0.65,1.06,-35.14,-40.59,-164.64],
            [1.96,-0.68,1.52,173.82,-128.81,-43.85],
            [-2,1.59,1.65,-35.21,-170.95,73.56],
            [1.46,-1.53,1.42,43.44,-28.4,82.62],
            [0.45,1.95,1.64,-124.43,-113.72,-99.26],
            [1.96,0.16,0.93,-42.72,81.28,-83.14],
            [0.11,0.83,1.63,-121.99,-46.67,62.29],
            [-0.08,2,1.96,92.92,122.96,-8.1],
            [1.21,-0.85,1.81,133.6,84.32,44.54],
            [-1.09,-0.34,0.72,-53.72,25.57,-94.88],
            [-0.01,-0.14,1.11,66.79,-116.33,-116.24],
            [1.6,1.06,1.12,-74.11,164.66,118.67],
            [0.3,1.27,1.56,11.03,-84.48,96.09],
            [1.38,-1.6,1.44,119.67,152.85,156.41],
            [0.95,-1.29,1.71,35.1,-99.44,-141.16],
            [0.34,-0.56,1.11,-59.29,-45.52,-114.4],
            [-1.01,-1.77,0.89,-72.28,-148.5,-144.33],
            [0.67,0.09,0.72,-17.07,50.44,-3.69],
            [-1.67,-0.66,1.68,-27.85,-114.98,-110.43],
            [0.5,-1.3,0.89,-50.54,-163.78,142.52],
            [0.64,-1.16,1.14,20.99,80.34,-144.33],
            [0.92,1.62,1.37,87.32,-54.92,-164.1],
            [1.56,0.7,0.92,-27.24,57.82,20.63],
            [1.93,-0.13,1.5,-25.43,-41.81,98.1],
            [1.08,1.65,1.28,-135.05,45.84,-67.7],
            [0.33,-1.58,0.81,-171.2,-172.21,-115.57],
            [1.71,0.98,1.69,-75.53,147.81,-57.98],
            [0.32,0.95,0.74,-65.69,108.2,-104.35],
            [-1.93,0.25,1.01,55.33,88.51,3.65],
            [-1.52,-1.26,0.93,164.5,112.72,146.29],
            [1.45,0.39,1.34,156.86,-42.01,46.41],
            [-0.06,-0.8,0.73,-15.16,42.22,-143.45],
            [1.38,-1.46,1.17,-93.43,27.18,-39.29],
            [-1.16,-1.15,0.75,95,10.82,-160.34],
            [0.21,1.58,0.76,93.36,-80.97,0.46],
            [0.52,-1.71,1.7,86.63,-90.49,-24.58],
            [-1.87,-1.03,1.01,87.73,-17.41,179.12],
            [0.46,-1.78,1.44,-141.87,-98.02,112.18],
            [-0.55,-0.23,1.95,65.36,109.6,-5.17],
            [-1.8,-1.95,1.21,-13.23,175,142],
            [-0.04,1.59,1.57,-103.62,-169.2,-130.48],
            [-1.23,-1.21,1.66,-144.53,12.84,-39.6],
            [-1.51,-1.63,1.21,116.49,-148.65,153.85],
            [-1.18,-0.77,1.52,-117,108.75,150.3],
            [-1.41,-0.18,0.75,-121.11,176.09,76.89],
            [-1.24,-1.59,1.91,59.76,-155.9,42.6],
            [-1.83,1.98,0.86,141.98,158.18,-56.42],
            [0.54,-0.67,0.97,5.96,-173.46,156.97],
            [-0.87,-0.81,1.72,72.97,66.18,-135.08],
            [0.15,-1.75,1.28,-124.71,102.15,83.01],
            [0.78,-0.81,1.68,163.24,12.29,52.73],
            [-0,-1.81,1.15,14.72,138.73,119.93],
            [0.14,0.02,0.98,64.7,143.64,-36.62],
            [-0.22,1.05,0.65,-166.84,45.34,89.94],
            [-1.5,0.52,1.54,111.31,-130.37,120.68],
            [-0.04,-1.64,1.2,89.5,-101.59,-63.91],
            [1.41,-1.68,1.23,-136.73,-114.43,18.81],
            [1.5,1.11,1.45,9.02,-164.94,172.49],
            [-0.92,1.62,0.68,-62.7,-141.5,17.75],
            [-1.17,0.14,1.04,16.72,41.92,-61.05],
            [0.26,-1.56,1.68,-36.4,158.28,43.01],
            [0.56,1.3,1.58,-30.57,-52.4,-50.17],
            [-0.33,-0.65,0.78,-114.93,-32.17,92.34],
            [-1.18,-0.82,0.78,-88.06,174.37,-31],
            [1.79,0.99,0.73,-172.61,160.41,-2.76],
            [-1.67,-1.96,0.61,152.52,63.59,70.11],
            [-1.58,-1.81,1.19,55.33,175.79,170.18],
            [-1.43,0.67,1.52,155.74,96.06,-62.01],
            [-1.33,0.41,1.61,-121.14,-58.79,121.61],
            [0.48,0.1,1.34,151.6,58.46,86.07],
            [0.29,0.92,0.75,106.08,-92.1,163.5],
            [-1.79,0.83,1.48,27.86,-73.62,-168.51],
            [1.72,1.13,0.78,-21.59,64.86,-51.53],
            [0.91,-0.85,0.79,-87.26,10.02,58.56],
            [0.95,0.77,0.74,90.7,-31.83,-78.66],
            [-1.75,0.23,0.8,-97.68,36.95,-97.06],
            [1.44,-0.41,0.84,-156.89,90.19,76.01],
            [1.74,-1.75,0.87,96.24,30.07,44.85],
            [1.94,1.12,1.04,61.63,18.65,32.62],
            [1.44,-0.65,1.04,77.48,30.09,57.76],
            [1.14,0.43,0.9,51.14,4.26,-162.88],
            [0.05,0.97,0.95,-29.14,-150.27,-54.44],
            [-1.29,-1.58,1.85,-39.33,79.05,-17.52],
            [-0.41,-1.49,1.58,113.81,178.62,-93.27],
            [-1.46,0.2,1.38,-65.73,-52.37,77.42],
            [-1.88,-0.06,0.86,113.23,169.65,128.23],
            [1.76,1.56,0.9,104.07,-55.28,-78.66],
            [-0.79,1.2,0.71,126.82,139.16,83.18],
            [-0.82,0.94,1.88,2.03,-16.31,-130.41],
            [-0.67,-1.79,1.59,48.84,-31.17,121.22],
            [-0.13,-1.71,1.38,162.32,-101.62,-130.1],
            [0.59,-1.65,1.04,-20.17,-134.76,31.76],
            [-1.9,1.19,0.83,-158.39,-68.79,-48.18],
            [1.37,1.77,1.47,132.03,81.4,110.43],
        ]

        self.pos = np.array([0.0, 0.0, 0.7])
        self.vel = np.array([0.0, 0.0, 0.0])
        self.acc = np.array([0.0, 0.0, 0.0])

        self.att = np.array([0.0, 0.0, 0.0])
        self.att_rate = np.array([0.0, 0.0, 0.0])
        self.att_acc = np.array([0.0, 0.0, 0.0])
    
        self.t_converge = 8.0
        self.n_wp = len(self.waypoints)
        self.T = len(self.waypoints) * self.t_converge

    def _idx(self, t: float) -> int:
        if t < 0.0:
            return 0
        idx = int(t // self.t_converge)
        if idx >= self.n_wp:
            idx = self.n_wp - 1
        return idx

    def get_3d_pt(self, t: float) -> Tuple[float, float, float, float, float, float, float, float, float]:
        x, y, z = self.pos
        vx, vy, vz = self.vel
        ax, ay, az = self.acc

        if self.loop_num != np.inf:
            total_T = self.T * max(1, int(self.loop_num))
            if total_T > 0 and t >= total_T:
                # return final waypoint when finished
                idx = self.n_wp - 1
            else:
                if self.loop_num > 1 and self.T > 0:
                    t = t % self.T
                idx = self._idx(t)
        else:
            idx = self._idx(t)
        wp = self.waypoints[idx]
        x, y, z = float(wp[0]), float(wp[1]), float(wp[2])

        return x, y, z, vx, vy, vz, ax, ay, az

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        roll, pitch, yaw = self.att
        roll_rate, pitch_rate, yaw_rate = self.att_rate
        roll_acc, pitch_acc, yaw_acc = self.att_acc

        if self.loop_num != np.inf:
            total_T = self.T * max(1, int(self.loop_num))
            if total_T > 0 and t >= total_T:
                idx = self.n_wp - 1
            else:
                if self.loop_num > 1 and self.T > 0:
                    t = t % self.T
                idx = self._idx(t)
        else:
            idx = self._idx(t)
        wp = self.waypoints[idx]
        roll_deg, pitch_deg, yaw_deg = float(wp[3]), float(wp[4]), float(wp[5])
        # convert degrees to radians and get quaternion (qx,qy,qz,qw)
        qx, qy, qz, qw = tf.transformations.quaternion_from_euler(
            np.deg2rad(roll_deg), np.deg2rad(pitch_deg), np.deg2rad(yaw_deg)
        )

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc

class PitchRotationTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.child_frame_id = "ee"

        self.T = 10  # total time for one full rotation cycle (0 to -2.5 and back to 0)
        self.omega = 2 * np.pi / self.T  # angular velocity

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        # Calculate the pitch angle based on time
        max_pitch = 3.2

        if 0 < t <= self.T / 2:
            pitch = max_pitch * (2 * t / self.T)  # from 0 to max_pitch rad
        elif self.T / 2 < t <= self.T:
            pitch = max_pitch * (2 - 2 * t / self.T)  # from max_pitch to 0 rad
        else:
            pitch = 0.0

        roll = 0.0
        yaw = 0.0

        (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

        if t <= self.T / 2:
            pitch_rate = max_pitch * 2 / self.T
        else:
            pitch_rate = -max_pitch * 2 / self.T

        roll_rate = 0.0
        yaw_rate = 0.0

        roll_acc = 0.0
        pitch_acc = 0.0
        yaw_acc = 0.0

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc


class PitchRotationFullTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.child_frame_id = "ee"

        self.T = 10  # total time for one full rotation cycle (0 to -2.5 and back to 0)
        self.omega = 2 * np.pi / self.T  # angular velocity

        self.converge_t = 3
        self.t_total = self.T * 2 + 3 * self.converge_t

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        # Calculate the pitch angle based on time
        max_pitch = np.pi
        cnvg_t = self.converge_t

        if 0 < t <= self.T / 2:
            pitch = max_pitch * (2 * t / self.T)
            pitch_rate = max_pitch * 2 / self.T

        elif self.T / 2 < t <= self.T / 2 + cnvg_t:
            pitch = max_pitch
            pitch_rate = 0.0

        elif self.T / 2 + cnvg_t < t <= self.T * 3 / 2 + cnvg_t:
            tau = t - self.T / 2 - cnvg_t
            pitch = max_pitch * (1 - 2 * tau / self.T)
            pitch_rate = -max_pitch * 2 / self.T

        elif self.T * 3 / 2 + cnvg_t < t <= self.T * 3 / 2 + 2 * cnvg_t:
            pitch = -max_pitch
            pitch_rate = 0.0

        elif self.T * 3 / 2 + 2 * cnvg_t < t <= self.T * 2 + 2 * cnvg_t:
            tau = t - self.T * 3 / 2 - 2 * cnvg_t
            pitch = max_pitch * (2 * tau / self.T - 1)
            pitch_rate = max_pitch * 2 / self.T

        else:
            pitch = 0.0
            pitch_rate = 0.0

        roll = 0.0
        yaw = 0.0

        (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

        roll_rate = 0.0
        yaw_rate = 0.0

        roll_acc = 0.0
        pitch_acc = 0.0
        yaw_acc = 0.0

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc


class PitchContinuousRotationTraj(BaseTrajwFixedRotor):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.child_frame_id = "ee"

        self.T = 20  # total time for one full rotation cycle (0 to -2.5 and back to 0)

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        # Calculate the pitch angle based on time
        t = t - np.floor(t / self.T) * self.T  # make t in the range of [0, T]

        max_pitch = np.pi * 2
        pitch = max_pitch * (t / self.T)  # from 0 to max_pitch rad
        roll = 0.0
        yaw = 0.0

        (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

        pitch_rate = max_pitch / self.T
        roll_rate = 0.0
        yaw_rate = 0.0

        roll_acc = 0.0
        pitch_acc = 0.0
        yaw_acc = 0.0

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc

    def get_fixed_rotor(self, t: float):
        rotor_id = 0
        ft_fixed = 7.0
        alpha_fixed = 0.0

        t_servo_change = 1.0
        min_ft = 0.5

        if self.T / 2 - 2 * t_servo_change >= t:
            self.use_fix_rotor_flag = False

        if self.T / 2 - 1 * t_servo_change >= t > self.T / 2 - 2 * t_servo_change:
            rotor_id = 1
            ft_fixed = min_ft
            alpha_fixed = np.pi
            self.use_fix_rotor_flag = True

        if self.T / 2 >= t > self.T / 2 - 1 * t_servo_change:
            rotor_id = 3
            ft_fixed = min_ft
            alpha_fixed = -np.pi
            self.use_fix_rotor_flag = True

        if self.T / 2 + 1 * t_servo_change >= t > self.T / 2:
            rotor_id = 0
            ft_fixed = min_ft
            alpha_fixed = np.pi
            self.use_fix_rotor_flag = True

        if self.T / 2 + 2 * t_servo_change >= t > self.T / 2 + 1 * t_servo_change:
            rotor_id = 2
            ft_fixed = min_ft
            alpha_fixed = -np.pi
            self.use_fix_rotor_flag = True

        if t > self.T / 2 + 2 * t_servo_change:
            self.use_fix_rotor_flag = False

        return rotor_id, ft_fixed, alpha_fixed


class PitchRotationTrajOpposite(PitchRotationTraj):
    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc = super().get_3d_orientation(t)
        return qw, qx, -qy, qz, roll_rate, -pitch_rate, yaw_rate, roll_acc, -pitch_acc, yaw_acc


class Continuous45DegRotationTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.child_frame_id = "ee"

        self.T = 10  # total time for one full rotation cycle (0 to -2.5 and back to 0)
        self.omega = 2 * np.pi / self.T  # angular velocity

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        roll = np.arctan2(np.sin(self.omega * t) / np.sqrt(2), np.cos(self.omega * t))
        pitch = np.arcsin(np.sin(self.omega * t) / np.sqrt(2))
        yaw = np.arctan(np.tan(self.omega * t / 2) ** 2)

        (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = 0.0

        roll_acc = 0.0
        pitch_acc = 0.0
        yaw_acc = 0.0

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc


class RollRotationTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.child_frame_id = "ee"

        self.T = 10  # total time for one full rotation cycle (0 to -2.5 and back to 0)
        self.omega = 2 * np.pi / self.T  # angular velocity

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        # Calculate the pitch angle based on time
        max_roll = 3.2

        if 0 < t <= self.T / 2:
            roll = max_roll * (2 * t / self.T)  # from 0 to max_roll rad
        elif self.T / 2 < t <= self.T:
            roll = max_roll * (2 - 2 * t / self.T)  # from max_roll to 0 rad
        else:
            roll = 0.0

        pitch = 0.0
        yaw = 0.0

        (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

        if t <= self.T / 2:
            roll_rate = max_roll * 2 / self.T
        else:
            roll_rate = -max_roll * 2 / self.T

        pitch_rate = 0.0
        yaw_rate = 0.0

        roll_acc = 0.0
        pitch_acc = 0.0
        yaw_acc = 0.0

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc


class RollRotationFullTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.child_frame_id = "ee"

        self.T = 10  # total time for one full rotation cycle (0 to -2.5 and back to 0)
        self.omega = 2 * np.pi / self.T  # angular velocity

        self.converge_t = 3
        self.t_total = self.T * 2 + 3 * self.converge_t

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        # Calculate the pitch angle based on time
        max_roll = np.pi
        cnvg_t = self.converge_t

        if 0 < t <= self.T / 2:
            roll = max_roll * (2 * t / self.T)
            roll_rate = 2 * max_roll / self.T

        elif self.T / 2 < t <= self.T / 2 + cnvg_t:
            roll = max_roll
            roll_rate = 0.0

        elif self.T / 2 + cnvg_t < t <= self.T * 3 / 2 + cnvg_t:
            tau = t - self.T / 2 - cnvg_t
            roll = max_roll * (1 - 2 * tau / self.T)
            roll_rate = -2 * max_roll / self.T

        elif self.T * 3 / 2 + cnvg_t < t <= self.T * 3 / 2 + 2 * cnvg_t:
            roll = -max_roll
            roll_rate = 0.0

        elif self.T * 3 / 2 + 2 * cnvg_t < t <= self.T * 2 + 2 * cnvg_t:
            tau = t - self.T * 3 / 2 - 2 * cnvg_t
            roll = max_roll * (2 * tau / self.T - 1)
            roll_rate = 2 * max_roll / self.T

        else:
            roll = 0.0
            roll_rate = 0.0

        pitch = 0.0
        yaw = 0.0

        (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

        pitch_rate = 0.0
        yaw_rate = 0.0

        roll_acc = 0.0
        pitch_acc = 0.0
        yaw_acc = 0.0

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc


class RollRotationTrajOpposite(RollRotationTraj):
    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc = super().get_3d_orientation(t)
        return qw, -qx, qy, qz, roll_rate, -pitch_rate, yaw_rate, roll_acc, -pitch_acc, yaw_acc


class RollRotationYaw045dTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.child_frame_id = "ee"

        self.T = 10  # total time for one full rotation cycle (0 to -2.5 and back to 0)
        self.omega = 2 * np.pi / self.T  # angular velocity

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        # Calculate the pitch angle based on time
        t = t - np.floor(t / self.T) * self.T  # make t in the range of [0, T]

        max_roll = np.pi

        if t <= self.T / 2:
            roll = max_roll * (2 * t / self.T)  # from 0 to max_roll rad
        else:
            roll = max_roll * (2 - 2 * t / self.T)  # from max_roll to 0 rad

        pitch = 0.0
        yaw = np.pi / 4.0

        (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(roll, pitch, yaw, axes="rxyz")

        if t <= self.T / 2:
            roll_rate = max_roll * 2 / self.T
        else:
            roll_rate = -max_roll * 2 / self.T

        pitch_rate = 0.0
        yaw_rate = 0.0

        roll_acc = 0.0
        pitch_acc = 0.0
        yaw_acc = 0.0

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc


class PitchSetPtTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.child_frame_id = "ee"

        self.t_converge = 8.0
        # fmt: off
        self.pitch_values = [ 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.0, 1.5, 1.0, 0.5,
            0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, ]
        # fmt: on
        self.T = len(self.pitch_values) * self.t_converge

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        roll, yaw = 0.0, 0.0
        roll_rate, yaw_rate = 0.0, 0.0
        roll_acc, yaw_acc = 0.0, 0.0

        index = min(int(t // self.t_converge), len(self.pitch_values) - 1)
        pitch = self.pitch_values[index]

        (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

        pitch_rate = 0.0
        pitch_acc = 0.0

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc


class YawRotationRoll090dTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.child_frame_id = "ee"

        self.T = 30  # total time for one full rotation cycle
        self.omega = 2 * np.pi / self.T  # angular velocity

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        # Calculate the yaw angle based on time

        yaw = self.omega * t

        roll = np.pi / 2.0
        pitch = 0.0

        (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(roll, pitch, yaw, axes="rxyz")

        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = self.omega

        roll_acc = 0.0
        pitch_acc = 0.0
        yaw_acc = 0.0

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc


class YawRotationRoll045dTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.child_frame_id = "ee"

        self.T = 30  # total time for one full rotation cycle
        self.omega = 2 * np.pi / self.T  # angular velocity

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        # Calculate the yaw angle based on time

        yaw = self.omega * t

        roll = np.pi / 4.0
        pitch = 0.0

        (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(roll, pitch, yaw, axes="rxyz")

        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = self.omega

        roll_acc = 0.0
        pitch_acc = 0.0
        yaw_acc = 0.0

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc


class YawRotationRoll135dTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.child_frame_id = "ee"

        self.T = 30  # total time for one full rotation cycle
        self.omega = 2 * np.pi / self.T  # angular velocity

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        # Calculate the yaw angle based on time

        yaw = self.omega * t

        roll = np.pi * 3.0 / 4.0
        pitch = 0.0

        (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(roll, pitch, yaw, axes="rxyz")

        roll_rate = 0.0
        pitch_rate = 0.0
        yaw_rate = self.omega

        roll_acc = 0.0
        pitch_acc = 0.0
        yaw_acc = 0.0

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc


class SingularityPointTraj(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.child_frame_id = "ee"

        self.att = np.array([0.0, 0.0, 0.0])
        self.att_rate = np.array([0.0, 0.0, 0.0])
        self.att_acc = np.array([0.0, 0.0, 0.0])

        self.t_converge = 8.0
        self.T = 8 * self.t_converge

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        yaw = np.pi / 4.0
        roll = np.pi / 2.0
        pitch = 0.0

        if 2 * self.t_converge >= t > self.t_converge:
            yaw = np.pi * 3.0 / 4.0

        if 3 * self.t_converge >= t > 2 * self.t_converge:
            yaw = np.pi * 5.0 / 4.0

        if 4 * self.t_converge >= t > 3 * self.t_converge:
            yaw = np.pi * 7.0 / 4.0

        if 5 * self.t_converge >= t > 4 * self.t_converge:
            yaw = np.pi * 1.0 / 4.0

        if 6 * self.t_converge >= t > 5 * self.t_converge:
            yaw = np.pi * 1.0 / 4.0 + 0.01

        if 7 * self.t_converge >= t > 6 * self.t_converge:
            yaw = np.pi * 1.0 / 4.0 - 0.01

        if 8 * self.t_converge >= t > 7 * self.t_converge:
            yaw = np.pi * 1.0 / 4.0 + 0.1

        (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(roll, pitch, yaw, axes="rxyz")

        roll_rate, pitch_rate, yaw_rate = self.att_rate
        roll_acc, pitch_acc, yaw_acc = self.att_acc

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc


class TestFixedRotorTraj(BaseTrajwFixedRotor):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.t_converge = 4.0
        self.T = 7 * self.t_converge

    def get_fixed_rotor(self, t: float):
        rotor_id = 0
        ft_fixed = self.hover_thrust
        alpha_fixed = 0.0

        if 0.0 >= t:
            self.use_fix_rotor_flag = False

        if self.t_converge >= t > 0.0:
            ft_fixed += 1.0
            self.use_fix_rotor_flag = True

        if 2 * self.t_converge >= t > self.t_converge:
            ft_fixed += 2.0
            self.use_fix_rotor_flag = True

        if 3 * self.t_converge >= t > 2 * self.t_converge:
            ft_fixed += 3.0
            self.use_fix_rotor_flag = True

        if 4 * self.t_converge >= t > 3 * self.t_converge:
            ft_fixed += 4.0
            self.use_fix_rotor_flag = True

        if 5 * self.t_converge >= t > 4 * self.t_converge:
            ft_fixed += 5.0
            self.use_fix_rotor_flag = True

        if 6 * self.t_converge >= t > 5 * self.t_converge:
            ft_fixed += 6.0
            self.use_fix_rotor_flag = True

        if 7 * self.t_converge >= t > 6 * self.t_converge:
            ft_fixed += 7.0
            self.use_fix_rotor_flag = True

        alpha_fixed = np.arccos(self.hover_thrust / ft_fixed)

        if t > 7 * self.t_converge:
            self.use_fix_rotor_flag = False

        return rotor_id, ft_fixed, alpha_fixed


class InfinitePitchNeg90deg(BaseTraj):
    def __init__(self, loop_num) -> None:
        super().__init__(loop_num)
        self.T = np.inf
        self.omega = 0.0

    def get_3d_pt(self, t: float) -> Tuple[float, float, float, float, float, float, float, float, float]:
        x, y, z, vx, vy, vz, ax, ay, az = 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        return x, y, z, vx, vy, vz, ax, ay, az

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        roll, pitch, yaw = 0.0, -np.pi / 2.0, 0.0
        (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

        roll_rate, pitch_rate, yaw_rate = 0.0, 0.0, 0.0
        roll_acc, pitch_acc, yaw_acc = 0.0, 0.0, 0.0

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc
