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

# 2. Define list of torques to apply [Tx, Ty, Tz] (Nm)
# Applied in World Frame by default
TORQUE_SEQUENCE = [
    [1, 0.0, 0.0],    # Torque +X
    [2, 0.0, 0.0],   # Torque -X
    [5, 0.0, 0.0],    # Torque +Y
    [10, 0.0, 0.0],   # Torque -Y
    [15, 0.0, 0.0],    # Torque +Z 
    [20, 0.0, 0.0],   # Torque -Z
    [25, 0.0, 0.0],    # Torque +Z
    [50, 0.0, 0.0],   # Torque -Z
    [100, 0.0, 0.0],   # Torque +Z
    # [100, 0.0, 0.0],   # Torque -Z
    # [200, 0.0, 0.0],   # Torque -Z
    # [1000, 0.0, 0.0],   # Torque -Z
    # [2000, 0.0, 0.0],   # Torque -Z
    # [3000, 0.0, 0.0],   # Torque -Z
    # [4000, 0.0, 0.0],   # Torque -Z
    # [5000, 0.0, 0.0],   # Torque -Z
]
# =================================================

def apply_force_sequence(duration, interval):
    rospy.wait_for_service('/gazebo/apply_body_wrench')
    try:
        apply_wrench_srv = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        
        print(f"------------------------------------------------")
        print(f"Target Body: '{BODY_NAME}'")
        print(f"Torque Duration: {duration} s")
        print(f"Interval: {interval} s")
        print(f"Sequence Length: {len(TORQUE_SEQUENCE)}")
        print(f"------------------------------------------------")

        for i, torque_vec in enumerate(TORQUE_SEQUENCE):
            if rospy.is_shutdown():
                break

            tx, ty, tz = torque_vec
            print(f"[{i+1}/{len(TORQUE_SEQUENCE)}] Applying Torque: [{tx:.2f}, {ty:.2f}, {tz:.2f}] Nm")

            # Create Wrench message
            wrench = Wrench()
            wrench.force.x = 0.0
            wrench.force.y = 0.0
            wrench.force.z = 0.0
            wrench.torque.x = tx
            wrench.torque.y = ty
            wrench.torque.z = tz

            # Call Gazebo service
            # reference_frame: "world" or "map" applies frame in global axes
            # reference_point: point within the body where wrench is applied (0,0,0 is COM)
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
            if i < len(TORQUE_SEQUENCE) - 1: # Don't wait after the last one (optional)
                print(f"   -> Resting for {interval}s...")
                rospy.sleep(interval)
        
        print("------------------------------------------------")
        print("Force sequence finished.")

    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Apply external forces to a robot link in Gazebo.")
    parser.add_argument("--duration", type=float, default=1.0, 
                        help="Duration to apply each force (seconds). Default: 1.0")
    parser.add_argument("--interval", type=float, default=5.0, 
                        help="Rest interval between forces (seconds). Default: 5.0")
    
    args = parser.parse_args()

    rospy.init_node("gazebo_force_applicator", anonymous=True)
    
    # Give some time for user to see the start
    rospy.sleep(1.0)
    
    apply_force_sequence(args.duration, args.interval)

if __name__ == "__main__":
    main()
