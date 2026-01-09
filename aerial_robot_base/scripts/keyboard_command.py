#!/usr/bin/env python
import rospy

from std_msgs.msg import Empty
from std_msgs.msg import Int8
from std_msgs.msg import UInt16
from std_msgs.msg import UInt8
import rosgraph

import sys, select, termios, tty

def getKey():
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

if __name__=="__main__":
        settings = termios.tcgetattr(sys.stdin)
        rospy.init_node("keyboard_command")
        robot_ns = rospy.get_param("~robot_ns", "")

        if not robot_ns:
                master = rosgraph.Master('/rostopic')
                try:
                        _, subs, _ = master.getSystemState()

                except socket.error:
                        raise ROSTopicIOException("Unable to communicate with master!")

                teleop_topics = [topic[0] for topic in subs if 'teleop_command/start' in topic[0]]
                if len(teleop_topics) == 1:
                        robot_ns = teleop_topics[0].split('/teleop')[0]

        ns = robot_ns + "/teleop_command"
        land_pub = rospy.Publisher(ns + '/land', Empty, queue_size=1)
        halt_pub = rospy.Publisher(ns + '/halt', Empty, queue_size=1)
        start_pub = rospy.Publisher(ns + '/start', Empty, queue_size=1)
        takeoff_pub = rospy.Publisher(ns + '/takeoff', Empty, queue_size=1)
        force_landing_pub = rospy.Publisher(ns + '/force_landing', Empty, queue_size=1)
        fault_pub = rospy.Publisher(ns + '/fault', Int8, queue_size=1)
        brake_pub = rospy.Publisher(ns + '/brake', Empty, queue_size=1)
        unbrake_pub = rospy.Publisher(ns + '/unbrake', Empty, queue_size=1)
        ctrl_mode_pub = rospy.Publisher(ns + '/ctrl_mode', Int8, queue_size=1)
        motion_start_pub = rospy.Publisher('task_start', Empty, queue_size=1)


        #the way to write publisher in python
        comm=Int8()
        gain=UInt16()
        fault=Int8()
        try:
                while(True):
                        key = getKey()
                        print("the key value is {}".format(ord(key)))
                        # takeoff and landing
                        if key == 'l':
                                land_pub.publish(Empty())
                                print("----- land command sent -----")
                                #for hydra joints
                        if key == 'r':
                                start_pub.publish(Empty())
                                print("----- start command sent -----")
                                #for hydra joints
                        if key == 'h':
                                halt_pub.publish(Empty())
                                print("----- halt command sent -----")
                                 #for hydra joints
                        if key == 'f':
                                force_landing_pub.publish(Empty())
                                print("----- force landing command sent -----")
                        if key == 't':
                                takeoff_pub.publish(Empty())
                                print("----- takeoff command sent -----")
                        if key == 'u':
                                stair_pub.publish(Empty())
                                print("----- stair command sent -----")
                        if key == 'x':
                                motion_start_pub.publish()
                        if key == 'v':
                                comm.data = 1
                                ctrl_mode_pub.publish(comm)
                        if key == 'p':
                                comm.data = 0
                                ctrl_mode_pub.publish(comm)
                        # if key == 'b':
                        #         brake_pub.publish(Empty())
                        #         print("----- brake command sent -----")
                        # if key == 'B':
                        #         unbrake_pub.publish(Empty())
                        #         print("----- unbrake command sent -----")
                        if key >= '0' and key <= '9':
                                fault.data = int(key)
                                fault_pub.publish(fault)
                                print("----- fault command {} sent -----".format(fault.data))
                        if key == '\x03':
                                break
                        rospy.sleep(0.001)

        except Exception as e:
                print(repr(e))
        finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


