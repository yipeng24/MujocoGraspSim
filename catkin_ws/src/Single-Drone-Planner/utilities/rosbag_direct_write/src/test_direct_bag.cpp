#include "rosbag_direct_write/rosbag_tool.hpp"

std::shared_ptr<rosbag_tool::Bag> bag_ptr_;

int main(int argc, char** argv){

    ros::init(argc, argv, "test_bag");
    ros::NodeHandle nh;

    bag_ptr_ = std::make_shared<rosbag_tool::Bag>("kk");
    for (int i = 0; i < 100; i++){
        Eigen::Vector3d p(i,0,0);
        Eigen::Quaterniond q(1, 0, 0, 0);
        bag_ptr_->write_posestamp("/test_p", p, q, ros::Time::now());
    }
    std::cout << " pub done " <<std::endl;
    
    return 0;
}