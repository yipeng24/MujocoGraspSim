#!/usr/bin/env python3
import rospy
from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import Quaternion, Vector3

def main():
    rospy.init_node("attitude_cmd_pub", anonymous=True)
    pub = rospy.Publisher("/attitude_cmd", AttitudeTarget, queue_size=1)
    rate = rospy.Rate(50)

    msg = AttitudeTarget()
    msg.header.frame_id = "world"

    # 模式A：只用 body_rate + thrust（更稳，不依赖姿态四元数解释）
    msg.type_mask = 64  # ignore orientation, use body_rate + thrust
    msg.body_rate = Vector3(0.0, 0.0, 0.8)  # yaw rate
    msg.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)  # ignored
    msg.thrust = 0.90

    rospy.loginfo("Publishing /attitude_cmd at 50Hz (type_mask=%d, thrust=%.2f)", msg.type_mask, msg.thrust)

    while not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()  # 关键：每次更新stamp
        pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    main()
