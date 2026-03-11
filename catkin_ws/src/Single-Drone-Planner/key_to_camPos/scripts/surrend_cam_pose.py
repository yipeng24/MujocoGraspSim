#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Quaternion, Point
from tf.transformations import quaternion_from_euler
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


def publish_camera_pose():
    rospy.init_node('camera_pose_publisher', anonymous=True)
    pub = rospy.Publisher('/free_cam_pos', Odometry, queue_size=10)
    
    # supermarcket
    # radius = 20
    # height = 25
    # n_sample_yaw = 240
    # pitch = np.arctan(height/radius)
    # rate = rospy.Rate(0.5)  # 发布频率设置为1Hz

    # palworld
    # c_x = -1.5
    # c_y = 0
    # c_z = 0

    # radius = 10
    # height = 3.8
    # n_sample_yaw = 240
    # pitch = 5/180*3.1415926  # 仿真时pitch为-30度

    # patrick
    c_x = 0
    c_y = 0
    c_z = 0

    radius = 15
    height = 5
    n_sample_yaw = 240
    pitch = 0/180*3.1415926  # 仿真时pitch为-30度

    rate = rospy.Rate(4)  # 发布频率设置为1Hz


    time_init = rospy.Time.now()

    cnt = 0
    while not rospy.is_shutdown():
        camera_pose = Odometry()
        camera_pose.header.stamp = rospy.Time.now()
        camera_pose.header.frame_id = "world"

        d_t = (rospy.Time.now() - time_init).to_sec()

        p_theta = 2 * 3.1415926 * cnt / n_sample_yaw + 3*3.1415926/2  # 角度增量
        p_x = radius * np.cos(p_theta)
        p_y = radius * np.sin(p_theta)

        camera_pose.pose.pose.position = Point(p_x+c_x, p_y+c_y, height+c_z)
        orientation = quaternion_from_euler(0, pitch, np.pi + p_theta)  # 使用欧拉角(0, 0, 0)创建四元数

        orientation = [orientation[3], orientation[0], orientation[1], orientation[2]]
        R_b2w = qvec2rotmat(orientation)
        R_c2b = np.array([[0,0,1],
                          [-1,0,0],
                          [0,-1,0]],dtype=float)
        R_c2w = np.dot(R_b2w,R_c2b)
        q_c2w = rotmat2qvec(R_c2w)

        camera_pose.pose.pose.orientation = Quaternion(q_c2w[1],q_c2w[2],q_c2w[3],q_c2w[0])

        pub.publish(camera_pose)
        cnt += 1
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_camera_pose()
    except rospy.ROSInterruptException:
        pass
