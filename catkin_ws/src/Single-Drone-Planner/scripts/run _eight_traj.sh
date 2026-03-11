# tool
roslaunch key_to_camPos key2UAM.launch & sleep 0.5;
roslaunch bridge_node pcl_sync.launch & sleep 1;
roslaunch airgrasp_urdf display.launch & sleep 0.5;

# sim
# roslaunch pcd_to_pcl pub_pcl_from_pcd.launch & sleep 1;
roslaunch fake_mod simulator.launch & sleep 1;

# sensing
# roslaunch local_sensing_node local_sensing.launch & sleep 1;

# planning
# roslaunch planning simulation_rviz_sensing.launch & sleep 1;

wait;
