roslaunch bridge_node pcl_sync.launch & sleep 1;
roslaunch airgrasp_urdf display.launch & sleep 1;

roslaunch px4ctrl run_ctrl_check_calibration.launch & sleep 1;
roslaunch ekf_quat ekf_quat_vicon_marvos.launch & sleep 1;
wait;
