"""
bridge/planning_gate.py
=======================
Planning-mode logic:
  - G key: send goal to planner (/move_base_simple/goal), reset fly approval
  - F key: approve trajectory → open gate → /position_cmd_planner forwarded
  - H key: hover in place (clears fly approval in planning mode)
  - Direct mode (planning_mode=False): G key → publish /position_cmd directly at sim rate
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from quadrotor_msgs.msg import PositionCommand

GLFW_KEY_F = 70
GLFW_KEY_G = 71
GLFW_KEY_H = 72


class PlanningGate:
    """
    Manages goal state and the command gate for planning mode.

    planning_mode=False (default):
      G key → set _goal_pos; publish_goal_if_set() sends /position_cmd each tick.

    planning_mode=True:
      G key → publish PoseStamped to /move_base_simple/goal; reset fly approval.
      F key → set _fly_approved=True; cmd_gate_cb() starts forwarding commands.
    """

    def __init__(self, lock, planning_mode: bool):
        self.lock          = lock
        self.planning_mode = planning_mode

        self._goal_pos     = None    # np.array([x,y,z]) or None
        self._goal_yaw     = 0.0
        self._fly_approved = False

        # Publisher: direct position command (non-planning mode)
        self._goal_pub = rospy.Publisher('/position_cmd', PositionCommand, queue_size=1)

        # Publisher: send goal to planner (planning mode)
        self._plan_goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

        # Publisher: gate output forwarded to /position_cmd (planning mode)
        self._cmd_gate_pub = rospy.Publisher('/position_cmd', PositionCommand, queue_size=1)

        if planning_mode:
            rospy.Subscriber('/position_cmd_planner', PositionCommand,
                             self.cmd_gate_cb, queue_size=1)

    # ── ROS callback ──────────────────────────────────────────────────────

    def cmd_gate_cb(self, msg):
        """Forward planner commands to /position_cmd only after F-key approval."""
        with self.lock:
            approved = self._fly_approved
        if approved:
            self._cmd_gate_pub.publish(msg)

    # ── Viewer key handler ────────────────────────────────────────────────

    def handle_key(self, keycode, drone_pos, drone_yaw, viewer):
        """
        Process G / H / F key events.
        drone_pos: np.array([x,y,z]) current drone position
        drone_yaw: float current drone yaw (rad)
        viewer:    mujoco passive viewer (for cam.lookat)
        """
        if keycode == GLFW_KEY_G:
            lookat = viewer.cam.lookat.copy()
            gpos   = np.array([lookat[0], lookat[1], max(0.2, drone_pos[2])])
            with self.lock:
                self._goal_pos     = gpos
                self._goal_yaw     = drone_yaw
                self._fly_approved = False   # new goal always needs re-approval

            rospy.loginfo(
                f"[bridge] Goal set: ({gpos[0]:.2f}, {gpos[1]:.2f}, {gpos[2]:.2f})")

            if self.planning_mode:
                ps                    = PoseStamped()
                ps.header.stamp       = rospy.Time.now()
                ps.header.frame_id    = 'world'
                ps.pose.position.x    = float(gpos[0])
                ps.pose.position.y    = float(gpos[1])
                ps.pose.position.z    = float(gpos[2])
                ps.pose.orientation.w = 1.0
                self._plan_goal_pub.publish(ps)
                rospy.loginfo("[bridge] Planning goal sent. Press F to approve and fly.")

        elif keycode == GLFW_KEY_H:
            with self.lock:
                self._goal_pos     = drone_pos.copy()
                self._goal_yaw     = drone_yaw
                self._fly_approved = False
            rospy.loginfo(
                f"[bridge] Hover at: ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f})")

        elif keycode == GLFW_KEY_F and self.planning_mode:
            with self.lock:
                self._fly_approved = True
            rospy.loginfo("[bridge] Flight APPROVED — drone will follow planned trajectory.")

    # ── Per-tick publishing (direct mode only) ────────────────────────────

    def publish_goal_if_set(self):
        """Publish /position_cmd at sim rate.  No-op in planning mode."""
        if self.planning_mode:
            return
        with self.lock:
            goal = self._goal_pos.copy() if self._goal_pos is not None else None
            yaw  = self._goal_yaw
        if goal is None:
            return
        cmd                   = PositionCommand()
        cmd.header.stamp      = rospy.Time.now()
        cmd.header.frame_id   = 'world'
        cmd.position.x        = float(goal[0])
        cmd.position.y        = float(goal[1])
        cmd.position.z        = float(goal[2])
        cmd.yaw               = float(yaw)
        cmd.trajectory_flag   = PositionCommand.TRAJECTORY_STATUS_READY
        self._goal_pub.publish(cmd)

    # ── Accessors ─────────────────────────────────────────────────────────

    @property
    def goal_pos(self):
        with self.lock:
            return self._goal_pos.copy() if self._goal_pos is not None else None
