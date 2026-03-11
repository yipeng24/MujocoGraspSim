#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry
import tf.transformations as tf
import numpy as np

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


# w x y z
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def stampedMsg2Pos(msg):
    #! cam_odom -> np_arr
    # Formatting camera pose as a transformation matrix w.r.t world frame
    cam_p = msg.pose.position
    cam_q = msg.pose.orientation

    #TODO 待确定
    R_c2w = qvec2rotmat([cam_q.w,cam_q.x,cam_q.y,cam_q.z])
    T_rob = np.array([cam_p.x,cam_p.y,cam_p.z])

    # 创建一个4x4的单位矩阵
    cam_pos = np.eye(4)
    cam_pos[:3, :3] = R_c2w
    cam_pos[:3, 3] = T_rob

    return cam_pos

class GoalToOdometryPublisher:
    def __init__(self):
        rospy.init_node('simple_goal2odom_node', anonymous=True)
        # self.odom_sub = rospy.Subscriber('/drone0/odom', Odometry, self.goal_callback)
        self.odom_sub = rospy.Subscriber('/odom_gt', Odometry, self.goal_callback)
        self.free_cam_pub = rospy.Publisher('/free_cam_pos', Odometry, queue_size=10)
        print("Waiting for a new goal...")

        rospy.spin()

    def goal_callback(self, msg):
        drone_odom_cur = msg
        q_b2w = [drone_odom_cur.pose.pose.orientation.w,
                 drone_odom_cur.pose.pose.orientation.x,
                 drone_odom_cur.pose.pose.orientation.y,
                 drone_odom_cur.pose.pose.orientation.z]
        T_b2w = [drone_odom_cur.pose.pose.position.x,
                 drone_odom_cur.pose.pose.position.y,
                 drone_odom_cur.pose.pose.position.z]
        R_b2w = qvec2rotmat(q_b2w)

        body_pose = np.eye(4)
        body_pose[:3,:3] = R_b2w
        body_pose[:3,3] = T_b2w

        R_c2b = np.array([[0,0,1],
                          [-1,0,0],
                          [0,-1,0]],dtype=float)
        T_c2b = np.array([0.2,0,0],dtype=float)
        cam2body = np.eye(4)
        cam2body[:3,:3] = R_c2b
        cam2body[:3,3] = T_c2b

        cam2world = np.dot(body_pose,cam2body)
        T_c2w = cam2world[:3,3]
        R_c2w = cam2world[:3,:3]

        cam_pose_msg = Odometry()
        cam_pose_msg.header.stamp = rospy.Time.now()
        cam_pose_msg.header.frame_id = "world"

        cam_pose_msg.pose.pose.position.x = T_c2w[0]
        cam_pose_msg.pose.pose.position.y = T_c2w[1]
        cam_pose_msg.pose.pose.position.z = T_c2w[2]

        q_c2w = rotmat2qvec(R_c2w)
        cam_pose_msg.pose.pose.orientation = Quaternion(q_c2w[1],q_c2w[2],q_c2w[3],q_c2w[0])

        self.free_cam_pub.publish(cam_pose_msg)
        print("Published free_cam message!")

    def run(self):
        while not rospy.is_shutdown():
            
            if self.simpleGoal:
                msg = self.simpleGoal

                odom_msg = Odometry()
                odom_msg.header.stamp = rospy.Time.now()
                odom_msg.header.frame_id = "world"

                # 填入接收到的目标位姿消息到里程计消息中
                odom_msg.pose.pose.position.x = msg.pose.position.x
                odom_msg.pose.pose.position.y = msg.pose.position.y
                odom_msg.pose.pose.position.z = 1.0
                odom_msg.pose.pose.orientation = msg.pose.orientation

                self.odom_pub.publish(odom_msg)
                print("Published odom message!")

                if rospy.Time.now() - self.fist_rev_time > rospy.Duration(0.5):
                    free_cam_msg = Odometry()
                    free_cam_msg.header.stamp = rospy.Time.now()
                    free_cam_msg.header.frame_id = "world"

                    free_cam_msg.pose.pose.position.x = msg.pose.position.x
                    free_cam_msg.pose.pose.position.y = msg.pose.position.y
                    free_cam_msg.pose.pose.position.z = 1.0

                    R_cam2body = np.array([[0,0,1],
                                             [-1,0,0],
                                            [0,-1,0]],dtype=float)
                    R_cam = np.dot(self.goal_odom[:3,:3],R_cam2body)

                    q_cam = rotmat2qvec(R_cam)
                    free_cam_msg.pose.pose.orientation = Quaternion(q_cam[1],q_cam[2],q_cam[3],q_cam[0])
                    self.free_cam_pub.publish(free_cam_msg)
                    print("Published free_cam message!")

                    # body to free_cam






            self.rate.sleep()




if __name__ == '__main__':
    try:
        goal_to_odom_publisher = GoalToOdometryPublisher()
        # goal_to_odom_publisher.run()
    except rospy.ROSInterruptException:
        pass