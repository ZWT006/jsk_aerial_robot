#!/usr/bin/env python3
import rospy
import tf.transformations as tf_trans
import math
import sys
import select
import termios
import tty
import argparse
from geometry_msgs.msg import PoseStamped, Quaternion

# ==========================================
# 1. Define Waypoint List
# Format: [x, y, z, roll(deg), pitch(deg), yaw(deg)]
# ==========================================

# Flip Test Waypoints
# WAYPOINTS = [
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],    # Hover at 1m
#     [0.0, 0.0, 1.5, 180.0, 0.0, 0.0],    # Move X+
#     [0.0, 0.0, 1.5, 0.0, 0.0, 0.0],  # Move X-, Pitch 180
#     [0.0, 0.0, 0.55, 0.0, 0.0, 0.0],    # Return origin, Down
# ]

# Vertical Flip Test Waypoints
# WAYPOINTS = [
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],    # Hover at 1m
#     [0.0, 0.0, 1.2, 90.0, 0.0, 0.0],    # Move X+
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Move X-, Pitch 180
#     [0.0, 0.0, 0.55, 0.0, 0.0, 0.0],    # Return origin, Down
# ]
######################################################################
# want to show large tilt (back)
# WAYPOINTS = [
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],    # Hover at 1m
#     [1.0, 0.0, 1.5, 45.0, 0.0, 0.0],    # Move X+
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#     [0.0, 1.0, 1.5, 0.0, -25.0, 0.0],    # Move X+
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#     [-1.0, 0.0, 1.5, -25.0, 0.0, 0.0],   # Move Y+, Up, Yaw 90
#     [0.0, 0.0, 1.0, 0.0, 0.0, 90.0],
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Move X-, Yaw 180
# ]

# want to show large tilt
# WAYPOINTS = [
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],    # Hover at 1m
#     [1.0, 0.0, 1.5, 45.0, 0.0, 0.0],    # Move X+
#     [0.0, 1.0, 1.5, 0.0, -25.0, 0.0],    # Move X+
#     [-1.0, 0.0, 1.5, -25.0, 0.0, 0.0],   # Move Y+, Up, Yaw 90
#     [0.0, 0.0, 1.0, 0.0, 0.0, 90.0],
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Move X-, Yaw 180
# ]

# Challenge Waypoints
# WAYPOINTS = [
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
######################################################################

# Normal Waypoints
# WAYPOINTS = [
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],    # Hover at 1m
#     [1.0, 1.0, 1.5, 0.0, 25.0, 0.0],    # Move X+
#     [-1.0, 1.0, 1.5, 0.0, 45.0, 90.0],   # Move Y+, Up, Yaw 90
#     [-1.0, -1.0, 1.5, 0.0, 45.0, 90.0],  # Move X-, Yaw 180
#     [1.0, -1.0, 1.5, 25.0, 0.0, 315.0],  # Move X-, Roll 180
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Move X-, Pitch 180
#     # [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],    # Return origin, Down
# ]

# Extended
# WAYPOINTS = [
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],    # Hover at 1m
#     [1.0, 1.0, 1.5, 0.0, 45.0, 90.0],    # Move X+
#     [-1.0, 1.0, 1.5, 0.0, 90.0, 135.0],   # Move Y+, Up, Yaw 90
#     [-1.0, -1.0, 1.5, 0.0, 90.0, 225.0],  # Move X-, Yaw 180
#     [1.0, -1.0, 1.5, 45.0, 45.0, 315.0],  # Move X-, Roll 180
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Move X-, Pitch 180
# ]

# 100 Sample Waypoints
WAYPOINTS = [
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

# Time interval for auto mode (seconds)
WAYPOINT_INTER = 8.0 

# ==========================================
# Helper Functions
# ==========================================
def getKey(settings):
    """Read keyboard input, non-blocking"""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def get_pose_msg(wp):
    """Convert list data to PoseStamped message"""
    x, y, z, r_deg, p_deg, y_deg = wp
    
    # Euler (deg) to Quaternion
    q = tf_trans.quaternion_from_euler(
        math.radians(r_deg), 
        math.radians(p_deg), 
        math.radians(y_deg)
    )
    
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "world"
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation = Quaternion(*q)
    return msg

# ==========================================
# Main Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Publish waypoints sequentially.")
    parser.add_argument("--mode", type=str, default="manual", choices=["manual", "auto"],
                        help="Mode: 'manual' (press N) or 'auto' (timer)")
    parser.add_argument("--loop", type=int, default=1, help="Number of loops to run in auto mode (default: 1)")
    args = parser.parse_args()

    rospy.init_node("waypoint_sequencer", anonymous=True)
    pub = rospy.Publisher("/desired_3D_pose", PoseStamped, queue_size=1, latch=True)

    # Save terminal settings to restore later
    settings = termios.tcgetattr(sys.stdin)

    current_idx = 0
    total_wp = len(WAYPOINTS)
    last_pub_time = rospy.Time.now()
    loop_count = 0 # Initialize loop counter

    print("------------------------------------------------")
    print(f"Loaded {total_wp} waypoints.")
    if args.mode == "auto":
        print(f"Mode: AUTO (Interval: {WAYPOINT_INTER}s)")
        print("Press Ctrl+C to exit.")
    else:
        print("Mode: MANUAL")
        print("Press 'N' (Shift+n) to publish the next waypoint.")
        print("Press Ctrl+C to exit.")
    print("------------------------------------------------")

    try:
        while not rospy.is_shutdown():
            key = getKey(settings)
            
            # Handle Ctrl+C (ASCII 3)
            if key == '\x03':
                break

            should_publish = False

            # Logic 1: Auto interval
            if args.mode == "auto":
                if (rospy.Time.now() - last_pub_time).to_sec() > WAYPOINT_INTER:
                    if current_idx < total_wp:
                        should_publish = True
                    elif loop_count < args.loop - 1:
                        # Reset for next loop
                        current_idx = 0
                        loop_count += 1
                        print(f"--- Starting loop {loop_count + 1}/{args.loop} ---")
                        should_publish = True
                    else:
                        # Finished all loops
                        pass 

            # Logic 2: Key 'N' trigger
            if key == 'N':
                if args.mode == "manual":
                    should_publish = True
                
                # If finished, print message on N press
                if current_idx >= total_wp:
                    print(f"All waypoints have been sent! ({current_idx}/{total_wp})")

            # Execute publish
            if should_publish and current_idx < total_wp:
                wp = WAYPOINTS[current_idx]
                msg = get_pose_msg(wp)
                pub.publish(msg)
                
                print(f"[{current_idx+1}/{total_wp}] Published: Pos({wp[0]}, {wp[1]}, {wp[2]}) RPY({wp[3]}, {wp[4]}, {wp[5]})")
                
                last_pub_time = rospy.Time.now()
                current_idx += 1
            
    except Exception as e:
        print(e)

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        print("\nExiting waypoint sequencer.")

if __name__ == "__main__":
    main()