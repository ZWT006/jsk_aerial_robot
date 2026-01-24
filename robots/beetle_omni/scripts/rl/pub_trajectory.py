#!/usr/bin/env python3
import rospy
import rospkg
import csv
import sys
import os
import argparse
import time
from geometry_msgs.msg import PoseStamped, Quaternion

def get_package_path(package_name="beetle_omni"):
    """Get the path of the specified ROS package."""
    try:
        rospack = rospkg.RosPack()
        return rospack.get_path(package_name)
    except rospkg.common.ResourceNotFound:
        print(f"Error: Package '{package_name}' not found.")
        sys.exit(1)

def read_csv_trajectory(file_path):
    """
    Reads trajectory from CSV.
    Expected format: x, y, z, qw, qx, qy, qz
    Returns a list of dictionaries.
    """
    trajectory = []
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    try:
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip header
            
            # Standard order assumption: x, y, z, qw, qx, qy, qz
            for row in reader:
                if len(row) < 7:
                    continue
                
                # Parse float values
                point = {
                    'x': float(row[0]),
                    'y': float(row[1]),
                    'z': float(row[2]),
                    'qw': float(row[3]),
                    'qx': float(row[4]),
                    'qy': float(row[5]),
                    'qz': float(row[6])
                }
                trajectory.append(point)
                
        print(f"Loaded {len(trajectory)} points from {file_path}")
        return trajectory
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Publish trajectory from CSV relative to package.")
    # Default looks in 'config' folder of the package
    parser.add_argument("--file", type=str, default="data/lemniscate_traj_3d.csv", 
                        help="Relative path to CSV file from package root (e.g., config/traj.csv)")
    parser.add_argument("--package", type=str, default="beetle_omni", 
                        help="ROS package name to locate file in")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step between points (seconds)")
    parser.add_argument("--loop", type=int, default=1, help="Number of loops to run (default: 1)")
    args = parser.parse_args()

    rospy.init_node("trajectory_publisher", anonymous=True)
    pub = rospy.Publisher("/desired_3D_pose", PoseStamped, queue_size=1)

    # Resolve file path
    pkg_path = get_package_path(args.package)
    full_path = os.path.join(pkg_path, args.file)

    # Read Trajectory
    trajectory = read_csv_trajectory(full_path)
    
    if not trajectory:
        print("Empty trajectory.")
        return

    rate = rospy.Rate(1.0 / args.dt)
    
    print("------------------------------------------------")
    print(f"Starting publication.")
    print(f"Package: {args.package}")
    print(f"File: {full_path}")
    print(f"DT: {args.dt}s")
    print("Press Ctrl+C to exit.")
    print("------------------------------------------------")

    # Wait a bit for connections
    rospy.sleep(1.0)

    loop_count = 0
    try:
        while not rospy.is_shutdown() and loop_count < args.loop:
            for i, point in enumerate(trajectory):
                if rospy.is_shutdown():
                    break

                msg = PoseStamped()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = "world"
                
                msg.pose.position.x = point['x']
                msg.pose.position.y = point['y']
                msg.pose.position.z = point['z']
                
                msg.pose.orientation.w = point['qw']
                msg.pose.orientation.x = point['qx']
                msg.pose.orientation.y = point['qy']
                msg.pose.orientation.z = point['qz']

                pub.publish(msg)
                
                # Optional: Print status every 100 points
                if i % 100 == 0:
                    print(f"Published point {i}/{len(trajectory)}")

                rate.sleep()

            loop_count += 1
            if loop_count < args.loop:
                print(f"Looping trajectory... ({loop_count}/{args.loop})")
            else:
                print("Trajectory finished.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()