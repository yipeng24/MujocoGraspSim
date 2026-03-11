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
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        self.odom_pub = rospy.Publisher('/drone0/odom', Odometry, queue_size=10)
        self.free_cam_pub = rospy.Publisher('/free_cam_pos', Odometry, queue_size=10)
        self.rate = rospy.Rate(1)  # 发布频率为1Hz
        self.simpleGoal = None
        self.fist_rev_time = None
        self.goal_odom = None
        print("Waiting for a new goal...")

    def goal_callback(self, msg):
        self.fist_rev_time = rospy.Time.now()
        self.goal_odom = stampedMsg2Pos(msg)
        self.simpleGoal = msg
        print("Received a new goal!")

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
        goal_to_odom_publisher.run()
    except rospy.ROSInterruptException:
        pass