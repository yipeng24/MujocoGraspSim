/*
 * @FilePath: /OA_ros/src/dynamic_planner/include/dynamic_planner.h
 * @Brief: 
 * @Version: 1.0
 * @Date: 2020-10-25 14:51:54
 * @Author: your name
 * @Copyright: your copyright description
 * @LastEditors: your name
 * @LastEditTime: 2020-11-01 19:12:41
 */
#ifndef _DYNAMIC_CONTROL_H
#define _DYNAMIC_CONTROL_H

#include <chrono>

#include <ros/ros.h>
#include <nav_msgs/Path.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Joy.h>
#include <sensor_msgs/PointCloud2.h>

#include <tf/transform_broadcaster.h>
#include <tf/tf.h>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>

#include <quadrotor_planner_interface/quadrotor_planner_interface.h>
#include <opencv4/opencv2/opencv.hpp> // For Unbuntu 20.04
// #include <opencv/cv.hpp> // For Unbuntu 18.04
#include <Eigen/Core>

#include <pcl/point_cloud.h> 
#include <pcl_conversions/pcl_conversions.h> 

using namespace Eigen;
using namespace std;
using namespace cv;
namespace dynamic_planner{

//Y means forward and backward
//X means left and right
//Z means up and down
//yaw means yaw
struct joyState{
    double Y= 0.0;
    double X = 0.0;
    double Z = 0.0;
    double yaw = 0.0;
};

class DynamicPlanner: public quadrotor_planner_interface::QuadrotorPlannerInterface{
private:
    joyState _joyState;
    ros::Subscriber _joy_sub;
    ros::Subscriber _odom_sub;
    void joy_callback(const sensor_msgs::Joy::ConstPtr& msg);

    vector<double> startPos;
    vector<double> endPos;
    double flightTime;
    bool isAuto;
    bool isGoEnd;

protected:
    nav_msgs::Odometry _odom_msg;
    ros::Time _odom_time;
    // call back functions 
    void odom_callback(const nav_msgs::OdometryConstPtr& msg);
    virtual void plan_callback(const ros::TimerEvent& event);
    virtual void ctrl_callback(const ros::TimerEvent& event);
public:

    DynamicPlanner(ros::NodeHandle& nh);
    virtual bool getCmd(double t, Vector3d& pos, Vector3d& vel, Vector3d& acc);
    virtual bool getCmd(double t, Vector3d& pos, Vector3d& vel, Vector3d& acc, double& yaw, double& yaw_dot);
};

}

#endif