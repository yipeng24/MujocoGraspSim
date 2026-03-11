"""
bridge/physics.py
=================
Drone physics: thrust + attitude/rate torque applied via xfrc_applied.
Arm joints driven via MuJoCo position servos (ctrl[4:7]).
"""

import numpy as np
import mujoco


class DronePhysics:
    """
    Holds control state (thrust, attitude quat, arm targets) and
    applies forces/torques to the MuJoCo drone body each step.
    """

    def __init__(self, model, data, drone_body_id, arm_qadr, arm_vadr, lock):
        self.model        = model
        self.data         = data
        self.drone_body_id = drone_body_id
        self.arm_qadr     = arm_qadr
        self.arm_vadr     = arm_vadr
        self.lock         = lock

        # Commanded values (written by ROS callbacks under lock)
        self.thrust         = 0.0
        self.attitude_quat  = np.array([1.0, 0.0, 0.0, 0.0])  # w,x,y,z desired
        self.body_rate_des  = np.zeros(3)                       # rad/s body frame
        self.use_rate_ctrl  = False
        self.received_cmd   = False
        self.arm_target     = np.zeros(3)  # [pitch, pitch2, roll]
        self.gripper_target = 0.0          # neutral (O=open 0.025, P=close -0.025)

    def apply_control(self):
        """
        Apply drone thrust + attitude/rate torques via xfrc_applied.
        Apply arm joint targets via data.ctrl[4:7] (position servos).

        Attitude PD gains (body frame, then rotated to world for xfrc_applied):
          Attitude mode: torque_body = 12 * ang_err_body - 1.8 * omega_body
          Rate    mode:  torque_body = -3 * (omega_body - omega_des_body)
        """
        # Arm servos (always safe: zero by default)
        self.data.ctrl[4] = float(self.arm_target[0])
        self.data.ctrl[5] = float(self.arm_target[1])
        self.data.ctrl[6] = float(self.arm_target[2])
        self.data.ctrl[7] = float(self.gripper_target)  # gripper (left mirrors via equality)

        # Clear external forces on drone body
        self.data.xfrc_applied[self.drone_body_id, :] = 0.0

        with self.lock:
            if not self.received_cmd:
                return
            thrust         = self.thrust
            q_des          = self.attitude_quat.copy()
            omega_des_body = self.body_rate_des.copy()
            use_rate_ctrl  = self.use_rate_ctrl

        # Rotation matrix body→world (row-major 3×3)
        rot    = self.data.xmat[self.drone_body_id].reshape(3, 3)
        body_z = rot[:, 2]   # body z-axis in world frame

        # Thrust along body z
        self.data.xfrc_applied[self.drone_body_id, 0] = thrust * body_z[0]
        self.data.xfrc_applied[self.drone_body_id, 1] = thrust * body_z[1]
        self.data.xfrc_applied[self.drone_body_id, 2] = thrust * body_z[2]

        # Angular velocity in body frame (freejoint qvel[3:6] is body frame)
        omega_body = self.data.qvel[3:6]

        if use_rate_ctrl:
            # Rate control: error in body frame, torque in body frame
            omega_err    = omega_body - omega_des_body
            Kd           = 3.0
            torque_body  = -Kd * omega_err
        else:
            # Attitude control: quaternion error q_err = q_c^{-1} * q_d (body frame)
            qc  = self.data.xquat[self.drone_body_id]   # current (w,x,y,z)
            qd  = q_des

            qew =  qc[0]*qd[0] + qc[1]*qd[1] + qc[2]*qd[2] + qc[3]*qd[3]
            qex =  qc[0]*qd[1] - qc[1]*qd[0] - qc[2]*qd[3] + qc[3]*qd[2]
            qey =  qc[0]*qd[2] + qc[1]*qd[3] - qc[2]*qd[0] - qc[3]*qd[1]
            qez =  qc[0]*qd[3] - qc[1]*qd[2] + qc[2]*qd[1] - qc[3]*qd[0]
            if qew < 0:
                qex, qey, qez = -qex, -qey, -qez

            # ang_err and omega_body are both in body frame
            ang_err     = np.array([2.0*qex, 2.0*qey, 2.0*qez])
            Kp, Kd      = 15.0, 2.2
            torque_body = Kp * ang_err - Kd * omega_body

        # xfrc_applied expects world-frame torque; rotate body → world
        torque = rot @ torque_body
        self.data.xfrc_applied[self.drone_body_id, 3] = torque[0]
        self.data.xfrc_applied[self.drone_body_id, 4] = torque[1]
        self.data.xfrc_applied[self.drone_body_id, 5] = torque[2]
