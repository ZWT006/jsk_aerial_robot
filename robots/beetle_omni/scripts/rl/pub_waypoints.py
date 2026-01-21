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

# Normal Waypoints
WAYPOINTS = [
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],    # Hover at 1m
    [1.0, 1.0, 1.5, 0.0, 25.0, 45.0],    # Move X+
    [-1.0, 1.0, 1.5, 0.0, 45.0, 90.0],   # Move Y+, Up, Yaw 90
    [-1.0, -1.0, 1.5, 0.0, 45.0, 90.0],  # Move X-, Yaw 180
    [1.0, -1.0, 1.5, 45.0, 0.0, 315.0],  # Move X-, Roll 180
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Move X-, Pitch 180
    [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],    # Return origin, Down
]


# Extended
# WAYPOINTS = [
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],    # Hover at 1m
#     [1.0, 1.0, 1.5, 0.0, 45.0, 90.0],    # Move X+
#     [-1.0, 1.0, 1.5, 0.0, 90.0, 135.0],   # Move Y+, Up, Yaw 90
#     [-1.0, -1.0, 1.5, 0.0, 90.0, 225.0],  # Move X-, Yaw 180
#     [1.0, -1.0, 1.5, 45.0, 45.0, 315.0],  # Move X-, Roll 180
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Move X-, Pitch 180
#     [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],    # Return origin, Down
# ]


# Challenge Waypoints
# WAYPOINTS = [
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],    # Hover at 1m
#     [1.0, 1.0, 1.5, 0.0, 45.0, 90.0],    # Move X+
#     [-1.0, 1.0, 1.5, 0.0, 90.0, 135.0],   # Move Y+, Up, Yaw 90
#     # [0.0, 0.0, 1.5, 0.0, 0.0, 0.0],
#     [-1.0, -1.0, 1.5, 180.0, 0.0, 0.0],  # Move X-, Yaw 180
#     [1.0, -1.0, 1.5, 180.0, 45.0, 270.0],  # Move X-, Roll 180
#     # [1.0, -1.0, 1.5, 0.0, 180.0, 225.0], # Alternative
#     [0.0, 0.0, 1.5, 180.0, 0.0, 90.0],  # Move X-, Pitch 180
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],    # Return origin
#     [0.0, 0.0, 0.55, 0.0, 0.0, 0.0],    # Return origin, Down
# ]

# Time interval for auto mode (seconds)
WAYPOINT_INTER = 5.0 

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
    args = parser.parse_args()

    rospy.init_node("waypoint_sequencer", anonymous=True)
    pub = rospy.Publisher("/desired_3D_pose", PoseStamped, queue_size=1, latch=True)

    # Save terminal settings to restore later
    settings = termios.tcgetattr(sys.stdin)

    current_idx = 0
    total_wp = len(WAYPOINTS)
    last_pub_time = rospy.Time.now()

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
                if current_idx < total_wp:
                    if (rospy.Time.now() - last_pub_time).to_sec() > WAYPOINT_INTER:
                        should_publish = True
                elif current_idx >= total_wp:
                    # In auto mode, just pass after finishing to avoid spamming
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