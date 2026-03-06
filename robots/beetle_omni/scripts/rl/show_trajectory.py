#!/usr/bin/env python3
"""
Reads a trajectory from a CSV file and publishes it as a visualization_msgs/Marker LINE_STRIP.

Usage examples:
    rosrun beetle_omni show_trajectory.py --package beetle_omni --file data/lemniscate_traj_3d_0.5.csv --color "1.0,1.0,0.0"

Arguments:
  --file     Relative path to CSV file.
  --package  ROS package name.
  --frame    Marker frame_id. Default: world
  --color    Marker color r,g,b. Default: 1.0,1.0,0.0 (Yellow)
  --width    Line width. Default: 0.03
"""

import rospy
import argparse
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import time
import rospkg
import csv
import sys
import os

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
                p = Point()
                p.x = float(row[0])
                p.y = float(row[1])
                p.z = float(row[2])
                trajectory.append(p)
                
        print(f"Loaded {len(trajectory)} points from {file_path}")
        return trajectory
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

def parse_color(s: str):
    parts = s.split(',')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be r,g,b")
    return tuple(float(x) for x in parts)


def main():
    parser = argparse.ArgumentParser(description='Read trajectory from CSV and publish as Marker')
    parser.add_argument('--frame', type=str, default='world', help='Marker frame_id')
    parser.add_argument('--rate', type=float, default=1.0, help='Publish rate (Hz)')
    parser.add_argument('--color', type=parse_color, default=(0.0, 0.0, 0.0), help='RGB color as "r,g,b" (0..1)')
    parser.add_argument('--alpha', type=float, default=0.8, help='Alpha (0..1)')
    parser.add_argument('--width', type=float, default=0.03, help='Line width (scale.x)')
    parser.add_argument('--ns', type=str, default='trajectory_ref', help='Marker namespace')
    parser.add_argument("--file", type=str, default="data/lemniscate_traj_3d_2.5.csv", 
                        help="Relative path to CSV file from package root")
    parser.add_argument("--package", type=str, default="beetle_omni", 
                        help="ROS package name to locate file in")
    
    args, _ = parser.parse_known_args()

    rospy.init_node('show_trajectory_marker', anonymous=True)

    # Resolve file path
    pkg_path = get_package_path(args.package)
    full_path = os.path.join(pkg_path, args.file)

    # Read Trajectory
    trajectory = read_csv_trajectory(full_path)
    
    if not trajectory:
        print("Empty trajectory.")
        return

    frame_id = args.frame
    rate_hz = args.rate
    color = args.color
    alpha = args.alpha
    width = args.width
    ns = args.ns

    pub = rospy.Publisher('/trajectory_marker', Marker, queue_size=1, latch=True)
    rate = rospy.Rate(rate_hz)

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = ns
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = width
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = alpha
    marker.lifetime = rospy.Duration(0)
    
    # Assign points once
    marker.points = trajectory

    rospy.loginfo("Started show_trajectory_marker: file=%s frame=%s points=%d", args.file, frame_id, len(trajectory))

    try:
        while not rospy.is_shutdown():
            marker.header.stamp = rospy.Time.now()
            # Publish static marker
            pub.publish(marker)
            rate.sleep()
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
