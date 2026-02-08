#!/usr/bin/env python3
import rospy
import argparse
import time
from gazebo_msgs.srv import ApplyBodyWrench
from geometry_msgs.msg import Wrench, Point

# ================= Configuration =================
# 1. Define Root Link Name
# Format is usually "model_name::link_name"
# You can check the model name in Gazebo (left panel) or `rostopic echo /gazebo/model_states`
# MODEL_NAME = "beetle_omni" 
MODEL_NAME = "beetle1" 
LINK_NAME = "root" 
BODY_NAME = f"{MODEL_NAME}::{LINK_NAME}"

# 2. Define list of forces to apply [Fx, Fy, Fz] (Newtons)
# Applied in World Frame by default
# FORCE_SEQUENCE = [
#     [0.0, 0.0, 1.0],    # Force +X
#     [0.0, 0.0, 10.0],   # Force -X
#     [0.0, 0.0, 100.0],    # Force +Y
#     [0.0, 0.0, 500.0],   # Force -Y
#     [0.0, 0.0, 1000.0],    # Force XY
#     [0.0, 0.0, 2000.0],   # Force -Z (Down)
#     [0.0, 0.0, 5000.0],    # Force +Z (Up)
# ]

FORCE_SEQUENCE = [
    [0.0, 0.0, -1.0],    # Force +X
    [0.0, 0.0, -2.0],   # Force -X
    [0.0, 0.0, -5.0],   # Force -X
    [0.0, 0.0, -10.0],   # Force -X
    [0.0, 0.0, -20.0],    # Force +Y
    [0.0, 0.0, -50.0],   # Force -Y
    [0.0, 0.0, -100.0],    # Force XY
    # [0.0, 0.0, -200.0],   # Force -Z (Down)
    # [0.0, 0.0, -500.0],    # Force +Z (Up)
]
# =================================================

def apply_force_sequence(duration, interval):
    rospy.wait_for_service('/gazebo/apply_body_wrench')
    try:
        apply_wrench_srv = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        
        print(f"------------------------------------------------")
        print(f"Target Body: '{BODY_NAME}'")
        print(f"Force Duration: {duration} s")
        print(f"Interval: {interval} s")
        print(f"Sequence Length: {len(FORCE_SEQUENCE)}")
        print(f"------------------------------------------------")

        for i, force_vec in enumerate(FORCE_SEQUENCE):
            if rospy.is_shutdown():
                break

            fx, fy, fz = force_vec
            print(f"[{i+1}/{len(FORCE_SEQUENCE)}] Applying Force: [{fx:.1f}, {fy:.1f}, {fz:.1f}] N")

            # Create Wrench message
            wrench = Wrench()
            wrench.force.x = fx
            wrench.force.y = fy
            wrench.force.z = fz
            wrench.torque.x = 0.0
            wrench.torque.y = 0.0
            wrench.torque.z = 0.0

            # Call Gazebo service
            # reference_frame: "world" or "map" applies force in global axes
            # reference_point: point within the body where force is applied (0,0,0 is COM)
            resp = apply_wrench_srv(
                body_name=BODY_NAME,
                reference_frame="world", 
                reference_point=Point(0, 0, 0),
                wrench=wrench,
                start_time=rospy.Time(0), # Start immediately
                duration=rospy.Duration(duration)
            )

            if resp.success:
                print(f"   -> Started (will last {duration}s)")
            else:
                print(f"   -> Failed: {resp.status_message}")

            # Wait for the force duration + interval
            # We assume 'interval' is the rest time AFTER the force application finishes
            
            # 1. Wait for force application to finish
            rospy.sleep(duration)
            
            # 2. Wait for the interval (rest period)
            if i < len(FORCE_SEQUENCE) - 1: # Don't wait after the last one (optional)
                print(f"   -> Resting for {interval}s...")
                rospy.sleep(interval)
        
        print("------------------------------------------------")
        print("Force sequence finished.")

    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Apply external forces to a robot link in Gazebo.")
    parser.add_argument("--duration", type=float, default=2.0, 
                        help="Duration to apply each force (seconds). Default: 2.0")
    parser.add_argument("--interval", type=float, default=2.0, 
                        help="Rest interval between forces (seconds). Default: 2.0")
    
    args = parser.parse_args()

    rospy.init_node("gazebo_force_applicator", anonymous=True)
    
    # Give some time for user to see the start
    rospy.sleep(1.0)
    
    apply_force_sequence(args.duration, args.interval)

if __name__ == "__main__":
    main()
