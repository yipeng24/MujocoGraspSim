#!/usr/bin/env python3
"""
mavros_sim.py - Fake MAVROS interface for MuJoCo simulation.

Bridges px4ctrl (which expects real MAVROS) with mujoco_ros_bridge:
  - Publishes /mavros/state, /mavros/extended_state, /mavros/imu/data, /mavros/battery
  - Serves /mavros/set_mode, /mavros/cmd/arming, /mavros/cmd/command
  - Relays /mavros/setpoint_raw/attitude -> /attitude_cmd (for mujoco_ros_bridge)
  - Publishes fake /joint_states (all-zeros, for arm-free flight)
"""

import threading
import rospy
import numpy as np

from mavros_msgs.msg import State, ExtendedState, AttitudeTarget
from mavros_msgs.srv import (SetMode, SetModeResponse,
                              CommandBool, CommandBoolResponse,
                              CommandLong, CommandLongResponse)
from sensor_msgs.msg import Imu, BatteryState
from nav_msgs.msg import Odometry


class MAVROSSim:
    def __init__(self):
        rospy.init_node('mavros_sim', anonymous=False)
        self._lock = threading.Lock()

        # FCU state
        self._connected = True
        self._armed = False
        self._mode = "MANUAL"          # start in MANUAL; px4ctrl sets OFFBOARD
        self._mode_before_offboard = "MANUAL"

        # Drone state cached from /odom (for IMU simulation)
        self._q = [1.0, 0.0, 0.0, 0.0]   # w, x, y, z
        self._ang_vel = [0.0, 0.0, 0.0]   # rad/s body frame
        self._lin_acc = [0.0, 0.0, 9.81]  # m/s^2, gravity dominant

        # ---- Publishers ----
        self._state_pub = rospy.Publisher(
            '/mavros/state', State, queue_size=5, latch=True)
        self._ext_state_pub = rospy.Publisher(
            '/mavros/extended_state', ExtendedState, queue_size=5)
        self._imu_pub = rospy.Publisher(
            '/mavros/imu/data', Imu, queue_size=10)
        self._bat_pub = rospy.Publisher(
            '/mavros/battery', BatteryState, queue_size=5)
        self._att_cmd_pub = rospy.Publisher(
            '/attitude_cmd', AttitudeTarget, queue_size=1)

        # ---- Subscribers ----
        rospy.Subscriber('/mavros/setpoint_raw/attitude', AttitudeTarget,
                         self._attitude_relay_cb, queue_size=1)
        rospy.Subscriber('/odom', Odometry, self._odom_cb, queue_size=1)

        # ---- Service servers ----
        rospy.Service('/mavros/set_mode',    SetMode,     self._set_mode_cb)
        rospy.Service('/mavros/cmd/arming',  CommandBool, self._arming_cb)
        rospy.Service('/mavros/cmd/command', CommandLong, self._command_cb)

        rospy.loginfo("[mavros_sim] Fake MAVROS ready (mode=%s)", self._mode)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _odom_cb(self, msg):
        with self._lock:
            o = msg.pose.pose.orientation
            self._q = [o.w, o.x, o.y, o.z]
            av = msg.twist.twist.angular
            self._ang_vel = [av.x, av.y, av.z]

    def _attitude_relay_cb(self, msg):
        """Relay px4ctrl output to mujoco_ros_bridge."""
        self._att_cmd_pub.publish(msg)

    def _set_mode_cb(self, req):
        new_mode = req.custom_mode
        with self._lock:
            if new_mode == "OFFBOARD":
                self._mode_before_offboard = self._mode
            self._mode = new_mode
        rospy.loginfo("[mavros_sim] Mode -> %s", new_mode)
        resp = SetModeResponse()
        resp.mode_sent = True
        return resp

    def _arming_cb(self, req):
        with self._lock:
            self._armed = req.value
        rospy.loginfo("[mavros_sim] Armed -> %s", req.value)
        resp = CommandBoolResponse()
        resp.success = True
        resp.result = 0
        return resp

    def _command_cb(self, req):
        rospy.loginfo("[mavros_sim] CommandLong cmd=%d", req.command)
        resp = CommandLongResponse()
        resp.success = True
        resp.result = 0
        return resp

    # ------------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------------
    def _publish_state(self):
        msg = State()
        msg.header.stamp = rospy.Time.now()
        with self._lock:
            msg.connected = self._connected
            msg.armed = self._armed
            msg.guided = True
            msg.mode = self._mode
            msg.system_status = 4  # MAV_STATE_ACTIVE
        self._state_pub.publish(msg)

    def _publish_extended_state(self):
        msg = ExtendedState()
        msg.header.stamp = rospy.Time.now()
        msg.vtol_state = ExtendedState.VTOL_STATE_UNDEFINED
        msg.landed_state = ExtendedState.LANDED_STATE_ON_GROUND
        self._ext_state_pub.publish(msg)

    def _publish_imu(self):
        msg = Imu()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        with self._lock:
            q = self._q
            av = self._ang_vel
        msg.orientation.w = q[0]
        msg.orientation.x = q[1]
        msg.orientation.y = q[2]
        msg.orientation.z = q[3]
        msg.angular_velocity.x = av[0]
        msg.angular_velocity.y = av[1]
        msg.angular_velocity.z = av[2]
        msg.linear_acceleration.x = 0.0
        msg.linear_acceleration.y = 0.0
        msg.linear_acceleration.z = 9.81
        # Small non-zero covariance (all-zero means "unknown" in ROS)
        msg.orientation_covariance[0] = 1e-4
        msg.orientation_covariance[4] = 1e-4
        msg.orientation_covariance[8] = 1e-4
        msg.angular_velocity_covariance[0] = 1e-4
        msg.angular_velocity_covariance[4] = 1e-4
        msg.angular_velocity_covariance[8] = 1e-4
        msg.linear_acceleration_covariance[0] = 1e-2
        msg.linear_acceleration_covariance[4] = 1e-2
        msg.linear_acceleration_covariance[8] = 1e-2
        self._imu_pub.publish(msg)

    def _publish_battery(self):
        msg = BatteryState()
        msg.header.stamp = rospy.Time.now()
        msg.voltage = 16.8          # 4S fully charged
        msg.percentage = 1.0
        msg.cell_voltage = [4.2, 4.2, 4.2, 4.2]   # 4 cells @ 4.2V
        self._bat_pub.publish(msg)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        # Publish slow topics (10 Hz) in background thread
        def _slow_loop():
            rate = rospy.Rate(10)
            while not rospy.is_shutdown():
                self._publish_state()
                self._publish_extended_state()
                self._publish_battery()
                rate.sleep()

        t = threading.Thread(target=_slow_loop, daemon=True)
        t.start()

        # Publish IMU at 100 Hz in main thread
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            self._publish_imu()
            rate.sleep()


if __name__ == '__main__':
    try:
        node = MAVROSSim()
        node.run()
    except rospy.ROSInterruptException:
        pass
