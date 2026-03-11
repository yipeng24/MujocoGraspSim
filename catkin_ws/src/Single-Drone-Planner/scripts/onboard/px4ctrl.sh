# tool
roslaunch key_to_camPos key2UAM.launch & sleep 0.5;
roslaunch bridge_node pcl_sync_world.launch & sleep 1;
roslaunch airgrasp_urdf display_real.launch & sleep 0.5;

roslaunch px4ctrl run_ctrl_real.launch & sleep 0.5;

wait;
