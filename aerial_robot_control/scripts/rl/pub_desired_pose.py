#!/usr/bin/env python3
import rospy
import tf.transformations as tf_trans
import argparse
import math
from geometry_msgs.msg import PoseStamped, Quaternion

def normalize_quat(q):
    x,y,z,w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-8:
        return (0.0, 0.0, 0.0, 1.0)
    inv = 1.0 / n
    return (x*inv, y*inv, z*inv, w*inv)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("z", type=float)
    parser.add_argument("roll", type=float)
    parser.add_argument("pitch", type=float)
    parser.add_argument("yaw", type=float)
    parser.add_argument("--degrees", action="store_true", help="interpret roll/pitch/yaw as degrees (default)")
    parser.add_argument("--radians", action="store_true", help="interpret roll/pitch/yaw as radians")
    parser.add_argument("--frame", default="world", help="frame_id for PoseStamped")
    args = parser.parse_args()

    rospy.init_node("publish_desired_pose", anonymous=True)
    pub = rospy.Publisher("/desired_3D_pose", PoseStamped, queue_size=1, latch=True)

    roll, pitch, yaw = args.roll, args.pitch, args.yaw
    if args.degrees:
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
    elif not args.radians:
        # default to degrees if neither flag is set
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)

    q = tf_trans.quaternion_from_euler(roll, pitch, yaw)  # returns [x,y,z,w]
    qx, qy, qz, qw = normalize_quat(q)

    ps = PoseStamped()
    ps.header.stamp = rospy.Time.now()
    if args.frame != "":
        ps.header.frame_id = args.frame
    else:
        ps.header.frame_id = "world"
    ps.pose.position.x = args.x
    ps.pose.position.y = args.y
    ps.pose.position.z = args.z
    ps.pose.orientation = Quaternion(qx, qy, qz, qw)

    # publish once (latched) so late subscribers still get it
    rospy.loginfo("Publishing desired pose to /desired_3D_pose")
    pub.publish(ps)
    rospy.sleep(0.5)  # give some time for the message to be sent
    rospy.loginfo("Published: pos=(%.4f, %.4f, %.4f) rpy=(%.4f, %.4f, %.4f) (deg=%s)",
                  args.x, args.y, args.z, args.roll, args.pitch, args.yaw, str(args.degrees))

if __name__ == "__main__":
    main()