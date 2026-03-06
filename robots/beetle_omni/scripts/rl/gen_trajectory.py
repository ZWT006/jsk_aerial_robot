import numpy as np
import csv
import os
import rospkg
import tf.transformations as tf_trans
from typing import Tuple


class BaseTraj:
    def __init__(self, loop_num: int = 1):
        self.loop_num = loop_num


class LemniscateTraj(BaseTraj):
    def __init__(self, loop_num=1) -> None:
        super().__init__(loop_num)
        self.a = 1.0
        self.z_range = 0.3
        self.T = 20.0
        self.omega = 2 * np.pi / self.T

    def get_2d_pt(self, t: float) -> Tuple[float, float, float, float, float, float]:
        t = t + self.T / 4  # shift the phase to make the trajectory start at the origin

        x = self.a * np.cos(self.omega * t)
        y = self.a * np.sin(2 * self.omega * t)

        vx = -self.a * self.omega * np.sin(self.omega * t)
        vy = 2 * self.a * self.omega * np.cos(2 * self.omega * t)

        ax = -self.a * self.omega**2 * np.cos(self.omega * t)
        ay = -4 * self.a * self.omega**2 * np.sin(2 * self.omega * t)

        return x, y, vx, vy, ax, ay

    def get_3d_pt(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float]:
        t = t + self.T / 4

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
    def __init__(self, loop_num=1) -> None:
        super().__init__(loop_num)
        self.a_orientation = 2.5

    def get_3d_orientation(
        self, t: float
    ) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
        t = t + self.T * 1 / 4

        roll = -2 * self.a_orientation * np.sin(2 * self.omega * t) / 2
        pitch = self.a_orientation * np.cos(self.omega * t)
        yaw = np.pi / 2 * np.sin(self.omega * t + np.pi) + np.pi / 2
        (qx, qy, qz, qw) = tf_trans.quaternion_from_euler(roll, pitch, yaw)

        roll_rate = -2 * 2 * self.a_orientation * self.omega * np.cos(2 * self.omega * t) / 2
        pitch_rate = -self.a_orientation * self.omega * np.sin(self.omega * t)
        yaw_rate = np.pi / 2 * self.omega * np.cos(self.omega * t + np.pi / 2)

        roll_acc = -2 * -4 * self.a_orientation * self.omega**2 * np.sin(2 * self.omega * t) / 2
        pitch_acc = -self.a_orientation * self.omega**2 * np.cos(self.omega * t)
        yaw_acc = -np.pi / 2 * self.omega**2 * np.sin(self.omega * t + np.pi / 2)

        return qw, qx, qy, qz, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """
    roll = pitch = 0, yaw given
    """
    qw = np.cos(yaw / 2.0)
    qx = 0.0
    qy = 0.0
    qz = np.sin(yaw / 2.0)
    return qw, qx, qy, qz


def generate_trajectory_csv(
    T: float,
    dt: float,
    csv_path: str,
):
    traj = LemniscateTrajOmni()
    traj.T = T
    traj.omega = 2 * np.pi / traj.T

    time_seq = np.arange(0.0, T, dt)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Only save position and orientation quaternion in the order: x,y,z,qw,qx,qy,qz
        writer.writerow(["x", "y", "z", "qw", "qx", "qy", "qz"])

        for t in time_seq:
            x, y, z, vx, vy, vz, _, _, _ = traj.get_3d_pt(t)

            # compute full 3D orientation quaternion from roll/pitch/yaw schedule
            qw, qx, qy, qz, _, _, _, _, _, _ = traj.get_3d_orientation(t)

            # write only x,y,z,qw,qx,qy,qz
            writer.writerow([
                x,
                y,
                z,
                qw,
                qx,
                qy,
                qz,
            ])

    print(f"Trajectory saved to: {csv_path}")


if __name__ == "__main__":
    T = 20.0     # total duration [s]
    dt = 0.01    # time step [s]

    # Resolve path relative to beetle_omni package
    try:
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("beetle_omni")
        data_dir = os.path.join(pkg_path, "data")
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        full_path = os.path.join(data_dir, "lemniscate_traj.csv")
    except Exception as e:
        print(f"Error finding package path: {e}")
        # Fallback to current directory
        full_path = "lemniscate_traj.csv"

    generate_trajectory_csv(
        T=T,
        dt=dt,
        csv_path=full_path,
    )
