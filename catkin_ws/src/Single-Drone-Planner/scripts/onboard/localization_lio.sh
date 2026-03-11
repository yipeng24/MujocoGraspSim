# sudo chmod 777 /dev/ttyACM0;
# roslaunch mavros px4.launch & sleep 5;
# rosrun mavros mavcmd long 511 31 5000 0 0 0 0 0 & sleep 3;
roslaunch livox_ros_driver2 msg_MID360.launch & sleep 1;
roslaunch fast_lio mapping_mid360.launch & sleep 1;
# roslaunch ekf_quat ekf_quat_lidar.launch & sleep 1;
wait;