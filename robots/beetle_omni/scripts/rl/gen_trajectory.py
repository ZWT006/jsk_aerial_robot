import numpy as np
import csv
import os
import rospkg
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
    traj = LemniscateTraj()
    traj.T = T
    traj.omega = 2 * np.pi / traj.T

    time_seq = np.arange(0.0, T, dt)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z", "qw", "qx", "qy", "qz"])

        for t in time_seq:
            x, y, z, vx, vy, vz, _, _, _ = traj.get_3d_pt(t)

            yaw = np.arctan2(vy, vx)
            qw, qx, qy, qz = yaw_to_quaternion(yaw)

            writer.writerow([
                # round(t, 6),
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
    T = 10.0     # total duration [s]
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
