#!/usr/bin/env python3
"""
Subscribe to /desired_3D_pose (PoseStamped) and publish visualization markers 
representing the 3D axes (RGB = XYZ) for RViz.
"""

import rospy
import copy
import math
import argparse
import sys
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseStamped, PoseArray
from visualization_msgs.msg import Marker, MarkerArray

class PoseToMarker:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Convert PoseStamped or PoseArray to MarkerArray')
        parser.add_argument('--topic', type=str, default='/desired_3D_pose', help='Topic name')
        parser.add_argument('--type', type=str, default='PoseStamped', choices=['PoseStamped', 'PoseArray'], help='Message type')
        
        args, _ = parser.parse_known_args()
        
        rospy.init_node('pose_to_marker', anonymous=True)
        
        topic = args.topic
        msg_type = args.type
        
        rospy.loginfo(f"Subscribing to {topic} ({msg_type})")

        self.pub = rospy.Publisher("/3D_pose_markers", MarkerArray, queue_size=1, latch=True)

        # Marker settings
        self.axis_length = 0.5
        self.axis_radius = 0.03
        self.alpha = 1.0

        if msg_type == 'PoseStamped':
            self.sub = rospy.Subscriber(topic, PoseStamped, self.cb_single, queue_size=1)
        elif msg_type == 'PoseArray':
            self.sub = rospy.Subscriber(topic, PoseArray, self.cb_array, queue_size=1)

        rospy.spin()

    def cb_single(self, msg: PoseStamped):
        markers = MarkerArray()
        self.add_pose_markers(markers, msg.pose, msg.header, 0)
        self.pub.publish(markers)

    def cb_array(self, msg: PoseArray):
        markers = MarkerArray()
        # for i, pose in enumerate(msg.poses):
        #     self.add_pose_markers(markers, pose, msg.header, i * 3)
        pose = msg.poses[0]
        self.add_pose_markers(markers, pose, msg.header, 0)
        self.pub.publish(markers)

    def add_pose_markers(self, markers, pose, header, id_start):
        base_quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        
        # X Axis (Red)
        m_x = self.create_arrow_marker(id_start + 0, header, pose.position, base_quat, [1.0, 0.0, 0.0])
        markers.markers.append(m_x)

        # Y Axis (Green)
        q_rot_z90 = tf_trans.quaternion_from_euler(0, 0, math.pi/2)
        q_y = tf_trans.quaternion_multiply(base_quat, q_rot_z90)
        m_y = self.create_arrow_marker(id_start + 1, header, pose.position, q_y, [0.0, 1.0, 0.0])
        markers.markers.append(m_y)

        # Z Axis (Blue)
        q_rot_yn90 = tf_trans.quaternion_from_euler(0, -math.pi/2, 0)
        q_z = tf_trans.quaternion_multiply(base_quat, q_rot_yn90)
        m_z = self.create_arrow_marker(id_start + 2, header, pose.position, q_z, [0.0, 0.0, 1.0])
        markers.markers.append(m_z)


    def create_arrow_marker(self, id, header, position, quat, rgb):
        m = Marker()
        m.header = header
        m.ns = "pose_axes"
        m.id = id
        m.type = Marker.ARROW
        m.action = Marker.ADD
        
        # Position
        m.pose.position = position
        
        # Orientation
        m.pose.orientation.x = quat[0]
        m.pose.orientation.y = quat[1]
        m.pose.orientation.z = quat[2]
        m.pose.orientation.w = quat[3]

        # Arrow dimensions: Scale X is length, Y/Z are width/height of shaft/head
        m.scale.x = self.axis_length # Length
        m.scale.y = self.axis_radius # Shaft width
        m.scale.z = self.axis_radius # Head width

        m.color.r = rgb[0]
        m.color.g = rgb[1]
        m.color.b = rgb[2]
        m.color.a = self.alpha
        
        m.lifetime = rospy.Duration(0) # 0 means forever
        return m

if __name__ == '__main__':
    try:
        PoseToMarker()
    except rospy.ROSInterruptException:
        pass
