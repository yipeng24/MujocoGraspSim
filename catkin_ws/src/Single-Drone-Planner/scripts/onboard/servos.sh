sudo chmod 777 /dev/ttyUSB0; 
sleep 0.5;

sudo modprobe mttcan;
sleep 1.5;
sudo ip link set can0 type can bitrate 1000000;
sleep 0.5;
sudo ip link set can0 up & sleep 0.5;

roslaunch grasp_chassis grasp_chassis.launch > tmp 2>&1;
wait;
