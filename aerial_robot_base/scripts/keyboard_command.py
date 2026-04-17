#!/usr/bin/env python

from __future__ import print_function # for print function in python2
import sys, select, termios, tty
import threading

import rospy
from std_msgs.msg import Empty
from std_msgs.msg import UInt32, Int8
from std_msgs.msg import Float32
# from aerial_robot_msgs.msg import FlightNav
import rosgraph

msg = """
Instruction:

---------------------------

r:  arming motor (please do before takeoff)
t:  takeoff
l:  land
f:  force landing
h:  halt (force stop motor)
0:  reset fault
1-9: set different fault (for testing fault handling)

     q           w           e           [
(turn left)  (forward)  (turn right)  (move up)

     a           s           d           ]
(move left)  (backward) (move right) (move down)


Please don't have caps lock on.
CTRL+c to quit
---------------------------
"""

def getKey():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

status_lock = threading.Lock()
status_initialized = False
command_status_msg = ""
battery_status_msg = ""

def printMsg(msg, msg_len = 60, line = 0):
    global status_initialized, command_status_msg, battery_status_msg
    with status_lock:
        if line == 0:
            command_status_msg = msg
        else:
            battery_status_msg = msg

        if status_initialized:
            sys.stdout.write("\033[1F")
        else:
            sys.stdout.write("\033[?25l")
            status_initialized = True

        sys.stdout.write("\r" + command_status_msg.ljust(msg_len) + "\033[K\n")
        sys.stdout.write("\r" + battery_status_msg.ljust(msg_len) + "\033[K")
        sys.stdout.flush()

def faultMaskString(mask, rotor_num):
    mask_width = max(1, min(int(rotor_num), 32))
    valid_mask = (1 << mask_width) - 1
    return "{0:0{1}b}".format(mask & valid_mask, mask_width)

def faultValidMask(rotor_num):
    mask_width = max(1, min(int(rotor_num), 32))
    return (1 << mask_width) - 1

def faultRotorCount(mask, rotor_num):
    return bin(mask & faultValidMask(rotor_num)).count("1")

# Global variables for battery monitoring
low_voltage = 22.0  # Default value

def battery_voltage_callback(msg):
    global low_voltage
    if msg.data < 19.2:
        printMsg("\033[91m Alert!!! Unsafe Battery Voltage : {:.2f}V\033[0m".format(msg.data), line = 1)
    elif msg.data < low_voltage:
        printMsg("\033[93m Warrning! Low Battery Voltage : {:.2f}V\033[0m".format(msg.data), line = 1)
    else:
        printMsg("\033[92m ======= Battery Voltage: {:.2f}V =======\033[0m".format(msg.data), line = 1)

if __name__=="__main__":
        settings = termios.tcgetattr(sys.stdin)
        rospy.init_node("keyboard_command")
        robot_ns = rospy.get_param("~robot_ns", "")
        print(msg)

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
        agentFaultRotor_pub = rospy.Publisher(ns + '/agentFaultRotor', UInt32, queue_size=1)
        spinalFaultRotor_pub = rospy.Publisher(robot_ns + '/spinal_fault_rotor_mask', UInt32, queue_size=1)
        brake_pub = rospy.Publisher(ns + '/brake', Empty, queue_size=1)
        unbrake_pub = rospy.Publisher(ns + '/unbrake', Empty, queue_size=1)
        ctrl_mode_pub = rospy.Publisher(ns + '/ctrl_mode', Int8, queue_size=1)
        # nav_pub = rospy.Publisher(robot_ns + '/uav/nav', FlightNav, queue_size=1)

        xy_vel   = rospy.get_param("xy_vel", 0.2)
        z_vel    = rospy.get_param("z_vel", 0.2)
        yaw_vel  = rospy.get_param("yaw_vel", 0.2)
        agentFault = False
        spinalFault = False
        rotorNum = rospy.get_param("~rotor_num", 4)
        maxFaultNum = 2
        fault = UInt32()
        
        # Battery voltage monitoring
        low_voltage = rospy.get_param("~low_voltage", 22.0)
        battery_voltage_sub = rospy.Subscriber(robot_ns + '/battery_voltage_status', Float32, battery_voltage_callback)

        motion_start_pub = rospy.Publisher('task_start', Empty, queue_size=1)

        try:
            while(True):
                # nav_msg = FlightNav()
                # nav_msg.control_frame = FlightNav.WORLD_FRAME
                # nav_msg.target = FlightNav.COG

                key = getKey()

                msg = ""

                if key == 'l':
                        land_pub.publish(Empty())
                        msg = "===== send land command ====="
                if key == 'r':
                        start_pub.publish(Empty())
                        msg = "===== send motor-arming command ====="
                if key == 'h':
                        halt_pub.publish(Empty())
                        msg = "===== send motor-disarming (halt) command ====="
                if key == 'f':
                        force_landing_pub.publish(Empty())
                        msg = "===== send force landing command ====="
                if key == 't':
                        takeoff_pub.publish(Empty())
                        msg = "===== send takeoff command ====="
                # if key == 'b':
                #         brake_pub.publish(Empty())
                #         msg = "===== send brake command ====="
                # if key == 'B':
                #         unbrake_pub.publish(Empty())
                #         msg = "===== send unbrake command ====="
                if key == '0':
                        fault.data = 0
                        fault_mask_str = faultMaskString(fault.data, rotorNum)
                        if agentFault and not spinalFault:
                                agentFaultRotor_pub.publish(fault)
                                agentFault = False
                                msg = "=====\033[93m send agent reset fault command: \033[91m{}\033[0m =====".format(fault_mask_str)
                        if spinalFault and agentFault:
                                spinalFaultRotor_pub.publish(fault)
                                spinalFault = False
                                msg = "=====\033[93m send spinal reset fault command: \033[91m{}\033[0m =====".format(fault_mask_str)
                if key >= '1' and key <= '4':
                        new_fault_mask = 1 << (int(key) - 1)
                        next_fault_data = (fault.data | new_fault_mask) & faultValidMask(rotorNum)
                        if faultRotorCount(next_fault_data, rotorNum) > maxFaultNum:
                                msg = "=====\033[91m error: fault mask {} exceeds max fault num {}\033[0m =====".format(
                                        faultMaskString(next_fault_data, rotorNum), maxFaultNum)
                        else:
                                fault.data = next_fault_data
                                agentFaultRotor_pub.publish(fault)
                                agentFault = True
                                spinalFaultRotor_pub.publish(fault)
                                spinalFault = True
                                fault_mask_str = faultMaskString(fault.data, rotorNum)
                                msg = "=====\033[93m send fault command: \033[91m{}\033[93m (+{})\033[0m =====".format(
                                        fault_mask_str, faultMaskString(new_fault_mask, rotorNum))
                if key >= '5' and key <= '8':
                        combined_fault_masks = {
                                '5': 0b1100,
                                '6': 0b0110,
                                '7': 0b0011,
                                '8': 0b1001,
                        }
                        new_fault_mask = combined_fault_masks[key]
                        next_fault_data = (fault.data | new_fault_mask) & faultValidMask(rotorNum)
                        if faultRotorCount(next_fault_data, rotorNum) > maxFaultNum:
                                msg = "=====\033[91m error: fault mask {} exceeds max fault num {}\033[0m =====".format(
                                        faultMaskString(next_fault_data, rotorNum), maxFaultNum)
                        else:
                                fault.data = next_fault_data
                                agentFaultRotor_pub.publish(fault)
                                agentFault = True
                                spinalFaultRotor_pub.publish(fault)
                                spinalFault = True
                                fault_mask_str = faultMaskString(fault.data, rotorNum)
                                msg = "=====\033[93m send fault command: \033[91m{}\033[93m (+{})\033[0m =====".format(
                                        fault_mask_str, faultMaskString(new_fault_mask, rotorNum))
                # if key == 'x':
                #         motion_start_pub.publish(Empty())
                #         msg = "===== send task-start command ====="
                # if key == 'w':
                #         nav_msg.pos_xy_nav_mode = FlightNav.VEL_MODE
                #         nav_msg.target_vel_x = xy_vel
                #         nav_pub.publish(nav_msg)
                #         msg = "===== send +x vel command ====="
                # if key == 's':
                #         nav_msg.pos_xy_nav_mode = FlightNav.VEL_MODE
                #         nav_msg.target_vel_x = -xy_vel
                #         nav_pub.publish(nav_msg)
                #         msg = "===== send -x vel command ====="
                # if key == 'a':
                #         nav_msg.pos_xy_nav_mode = FlightNav.VEL_MODE
                #         nav_msg.target_vel_y = xy_vel
                #         nav_pub.publish(nav_msg)
                #         msg = "===== send +y vel command ====="
                # if key == 'd':
                #         nav_msg.pos_xy_nav_mode = FlightNav.VEL_MODE
                #         nav_msg.target_vel_y = -xy_vel
                #         nav_pub.publish(nav_msg)
                #         msg = "===== send -y vel command ====="
                # if key == 'q':
                #         nav_msg.yaw_nav_mode = FlightNav.VEL_MODE
                #         nav_msg.target_omega_z = yaw_vel
                #         nav_pub.publish(nav_msg)
                #         msg = "===== send +yaw vel command ====="
                # if key == 'e':
                #         nav_msg.yaw_nav_mode = FlightNav.VEL_MODE
                #         nav_msg.target_omega_z = -yaw_vel
                #         msg = "===== send -yaw vel command ====="
                #         nav_pub.publish(nav_msg)
                # if key == '[':
                #         nav_msg.pos_z_nav_mode = FlightNav.VEL_MODE
                #         nav_msg.target_vel_z = z_vel
                #         nav_pub.publish(nav_msg)
                #         msg = "===== send +z vel command ====="
                # if key == ']':
                #         nav_msg.pos_z_nav_mode = FlightNav.VEL_MODE
                #         nav_msg.target_vel_z = -z_vel
                #         nav_pub.publish(nav_msg)
                #         msg = "===== send -z vel command ====="
                if key == '\x03':
                        break

                printMsg(msg)
                
                rospy.sleep(0.001)

        except Exception as e:
                print(repr(e))
        finally:
                sys.stdout.write("\033[?25h")
                if status_initialized:
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
