rosbag record --tcpnodelay \
/drone0/odom \
/position_cmd \
/joint_state_est \
/joint_state_cmd \
/vicon/hand_grasp/hand_grasp \
/vicon/hand/hand \
/debugPx4ctrl \
/px4ctrl/end_pose_est \
/px4ctrl/end_pose_ref \
/mavros/setpoint_raw/attitude \
/mavros/imu/data \
/px4ctrl/robot \
/px4ctrl/uam_state_vis_marker \
/camera/compressed \
/vicon/hand_stick/hand_stick
# /d435/color/image_raw \
# /d435/aligned_depth_to_color/image_raw \
# /d435/aligned_depth_to_color/camera_info