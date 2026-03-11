# tool
roslaunch bridge_node pcl_sync_world_sim.launch & sleep 1;
roslaunch airgrasp_urdf display.launch & sleep 0.5;

# sim
roslaunch pcd_to_pcl pub_pcl_from_pcd.launch & sleep 1;
roslaunch fake_mod simulator.launch & sleep 1;

# sensing
# roslaunch local_sensing_node local_sensing.launch & sleep 1;

roslaunch px4ctrl run_ctrl_sim.launch;

wait;
