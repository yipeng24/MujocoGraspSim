sudo chmod 777 /dev/ttyACM0;
roslaunch mavros px4.launch & sleep 5;
rosrun mavros mavcmd long 511 31 5000 0 0 0 0 0 & sleep 3;
roslaunch vicon_bridge vicon.launch & sleep 3;
roslaunch ekf_quat ekf_quat_vicon_marvos.launch & sleep 1;
wait;