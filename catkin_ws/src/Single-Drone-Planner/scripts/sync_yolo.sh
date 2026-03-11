roslaunch bridge_node img_sync.launch & sleep 1.0;
python src/Single-Drone-Planner/yolo_cup_seg/scripts/cup_segentation_pc.py & sleep 1.0;
wait;