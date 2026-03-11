#include "ch_rc_sdf/ch_rc_sdf.h"

int main(int argc, char** argv){
    ros::init(argc, argv, "test_node");
    ros::NodeHandle nh("~");
   
    std::shared_ptr<clutter_hand::CH_RC_SDF> rc_sdf_ptr_;
    rc_sdf_ptr_ = std::make_shared<clutter_hand::CH_RC_SDF>();
    rc_sdf_ptr_->initMap(nh, true, true);

    std::cout << "CH_RC_SDF initialized" << std::endl;

    // ROS spin
    ros::Rate loop_rate(10);

    while (ros::ok()) {
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
