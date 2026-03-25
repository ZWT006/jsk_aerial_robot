#!/usr/bin/env python3
"""
Publish a fake MoCap PoseStamped message for development/debugging
when real MoCap data is unavailable.

The MoCap sensor plugin (aerial_robot_estimation/sensor/mocap.cpp) subscribes
to geometry_msgs/PoseStamped at <robot_ns>/pose (or the topic configured via
the sensor_plugin/mocap/mocap_sub_name parameter).

Usage:
  rosrun beetle_omni pub_fake_mocap.py

Edit variables directly in the "configurable parameters" block:
  robot_ns    (str)    robot namespace          default: "beetle_omni"
  topic       (str)    topic within robot_ns   default: "pose"
  frame_id    (str)    header frame_id         default: "world"
  rate_hz     (float)  publish rate [Hz]       default: 100.0
  x/y/z       (float)  position [m]            default: 0.0 / 0.0 / 0.2
  roll/pitch/yaw (float) orientation [rad]     default: 0.0
"""

import math
import rospy
from geometry_msgs.msg import PoseStamped


def rpy_to_quaternion(roll, pitch, yaw):
    """Convert Roll-Pitch-Yaw (rad) to quaternion (x, y, z, w)."""
    cr = math.cos(roll  * 0.5)
    sr = math.sin(roll  * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw   * 0.5)
    sy = math.sin(yaw   * 0.5)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


def main():
    rospy.init_node("pub_fake_mocap", anonymous=False)

    # --- configurable parameters ---
    robot_ns = "beetle_omni"
    topic    = "mocap/pose"
    frame_id = "world"
    rate_hz  = 100.0

    x     = 0.0
    y     = 0.0
    z     = 0.2
    roll  = 0.0
    pitch = 0.0
    yaw   = 0.0

    # full topic:  /<robot_ns>/<topic>
    full_topic = "/{}/{}".format(robot_ns, topic)

    pub = rospy.Publisher(full_topic, PoseStamped, queue_size=1)
    rate = rospy.Rate(rate_hz)

    qx, qy, qz, qw = rpy_to_quaternion(roll, pitch, yaw)

    rospy.loginfo(
        "[pub_fake_mocap] Publishing to: %s  @ %.0f Hz\n"
        "  pos  = (%.3f, %.3f, %.3f)\n"
        "  RPY  = (%.3f, %.3f, %.3f) rad\n"
        "  quat = (%.4f, %.4f, %.4f, %.4f)",
        full_topic, rate_hz,
        x, y, z,
        roll, pitch, yaw,
        qx, qy, qz, qw,
    )

    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw

    while not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
