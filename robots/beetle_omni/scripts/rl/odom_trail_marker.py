#!/usr/bin/env python3
"""
Subscribe to a nav_msgs/Odometry topic and visualize the recent trajectory as a LINE_STRIP marker.

Usage examples:
    rosrun beetle_omni odom_trail_marker.py --topic /odom --frame world --duration 10.0 --rate 10

Arguments:
  --topic    Topic name to subscribe (nav_msgs/Odometry). Default: /odom
  --frame    Marker frame_id (tf frame in which positions are expressed). Default: world
  --duration Trail length in seconds (how far back to show). Default: 5.0
  --rate     Publish rate for the marker (Hz). Default: 10
  --color    Marker color as r,g,b (0..1). Default: 0,1,0 (green)
  --alpha    Marker alpha (0..1). Default: 1.0
  --width    Line width (scale.x). Default: 0.03
  --ns       Marker namespace. Default: odom_trail
  --topic_pose Optional: if odom uses a child_frame pose in different frame, provide frame transform handling (not implemented)

The node stores time-stamped positions in a deque and publishes only points whose timestamp is within now-duration..now.
"""

import rospy
import argparse
from collections import deque
from threading import Lock
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import time


def parse_color(s: str):
    parts = s.split(',')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be r,g,b")
    return tuple(float(x) for x in parts)


def main():
    parser = argparse.ArgumentParser(description='Visualize odom trajectory as a time-limited LINE_STRIP marker')
    # parser.add_argument('--topic', type=str, default='/beetle_omni/uav/baselink/odom', help='Odometry topic to subscribe')
    parser.add_argument('--topic', type=str, default='/beetle1/uav/cog/odom', help='Odometry topic to subscribe')
    parser.add_argument('--frame', type=str, default='world', help='Marker frame_id')
    parser.add_argument('--duration', type=float, default=2.0, help='Trail duration in seconds')
    parser.add_argument('--rate', type=float, default=10.0, help='Publish rate (Hz)')
    parser.add_argument('--color', type=parse_color, default=(1.0, 0.0, 0.0), help='RGB color as "r,g,b" (0..1)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha (0..1)')
    parser.add_argument('--width', type=float, default=0.03, help='Line width (scale.x)')
    parser.add_argument('--ns', type=str, default='odom_trail', help='Marker namespace')
    # Use parse_known_args to act gracefully when roslaunch passes __name and __log args
    args, _ = parser.parse_known_args()

    rospy.init_node('odom_trail_marker', anonymous=True)

    topic = args.topic
    frame_id = args.frame
    trail_duration = args.duration
    rate_hz = args.rate
    color = args.color
    alpha = args.alpha
    width = args.width
    ns = args.ns

    points = deque()  # each item: (stamp_secs, Point)
    lock = Lock()
    cb_count = [0]

    def odom_cb(msg: Odometry):
        cb_count[0] += 1
        if cb_count[0] % 100 == 0:
            rospy.loginfo("odom_cb active. Total received: %d. Current trail size: %d", cb_count[0], len(points))

        # store position and stamp
        p = msg.pose.pose.position
        pt = Point()
        pt.x = p.x
        pt.y = p.y
        pt.z = p.z
        stamp = msg.header.stamp.to_sec()
        with lock:
            points.append((stamp, pt))
            # optionally prune here to keep deque bounded
            # but main loop will prune against current time

    sub = rospy.Subscriber(topic, Odometry, odom_cb, queue_size=100)

    pub = rospy.Publisher('/trail_marker', Marker, queue_size=1)

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
    marker.lifetime = rospy.Duration(0)  # persistent until overwritten

    rate = rospy.Rate(rate_hz)
    rospy.loginfo("Started odom_trail_marker: topic=%s frame=%s duration=%.2fs rate=%.1fHz", topic, frame_id, trail_duration, rate_hz)

    while not rospy.is_shutdown():
        now = rospy.Time.now().to_sec()
        cutoff = now - trail_duration
        with lock:
            # drop old points
            while points and points[0][0] < cutoff:
                points.popleft()
            # convert remaining to Marker points
            pts = [pt for (_t, pt) in points]
        marker.header.stamp = rospy.Time.now()

        if len(pts) < 2:
            marker.action = Marker.DELETE
            pub.publish(marker)
        else:
            marker.action = Marker.ADD
            marker.points = pts
            pub.publish(marker)
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
