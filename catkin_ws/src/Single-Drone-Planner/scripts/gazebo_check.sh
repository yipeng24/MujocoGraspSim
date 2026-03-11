roslaunch rotors_gazebo simulator_test_mpc.launch & sleep 3;
roslaunch px4ctrl run_ctrl.launch & sleep 2;
roslaunch poly_planner eight.launch & sleep 1;
roslaunch poly_traj_server traj_server.launch & sleep 1;
wait;
