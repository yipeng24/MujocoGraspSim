/*
 * @FilePath: /OA_ros/src/dynamic_planner/src/dynamic_planner.cc
 * @Brief: 
 * @Version: 1.0
 * @Date: 2020-10-25 14:52:05
 * @Author: your name
 * @Copyright: your copyright description
 * @LastEditors: your name
 * @LastEditTime: 2020-11-01 19:27:08
 */
#include "dynamic_planner.h"
using namespace Eigen;
namespace dynamic_planner{

DynamicPlanner::DynamicPlanner(ros::NodeHandle& nh):quadrotor_planner_interface::QuadrotorPlannerInterface(nh), isGoEnd(true){
    _odom_sub = nh.subscribe<nav_msgs::Odometry>("dynamic_odom", 1, &DynamicPlanner::odom_callback, this);
    _joy_sub = nh.subscribe<sensor_msgs::Joy>("joy", 10, &DynamicPlanner::joy_callback, this);

    nh.getParam("/StartPos", startPos);
    nh.getParam("/EndPos", endPos);
    nh.getParam("/FlightTime", flightTime);

    nh.getParam("planning/isAuto", isAuto);

}

void DynamicPlanner::joy_callback(const sensor_msgs::Joy::ConstPtr& msg){
    _joyState.X = msg->axes.at(3);
    _joyState.Y = msg->axes.at(4);
    _joyState.Z = msg->axes.at(1);
    _joyState.yaw = msg->axes.at(0);

    // cout << "joy callback: " << _joyState.X << " | " << _joyState.Y << " | " << _joyState.Z << " | " << _joyState.yaw << endl;
}

void DynamicPlanner::odom_callback(const nav_msgs::OdometryConstPtr& msg) {
    _odom_msg = *msg;
    _odom_time = ros::Time::now();
}

void DynamicPlanner::plan_callback(const ros::TimerEvent& event){

}
void DynamicPlanner::ctrl_callback(const ros::TimerEvent& event){
    double t = 1.0;
    Vector3d pos, vel, acc;
    getCmd(t, pos, vel, acc);
    pubCmd(pos, vel, acc);
}

bool DynamicPlanner::getCmd(double t, Vector3d& pos, Vector3d& vel, Vector3d& acc){
    ros::Time cmdTime = ros::Time::now();
    double cmdT = (cmdTime - _odom_time).toSec();

    if(!isAuto){
        pos = Vector3d(_odom_msg.pose.pose.position.x + cmdT*_joyState.X, 
                    _odom_msg.pose.pose.position.y + cmdT*_joyState.Y,
                    _odom_msg.pose.pose.position.z + cmdT*_joyState.Z);
        vel = Vector3d(_joyState.X, _joyState.Y, _joyState.Z);
        acc = Vector3d(0, 0, 0);
        return true;
    }

    Vector3d nowPose = Vector3d(_odom_msg.pose.pose.position.x, _odom_msg.pose.pose.position.y, _odom_msg.pose.pose.position.z);
    Vector3d startPose = Vector3d(startPos[0], startPos[1], startPos[2]);
    Vector3d endPose = Vector3d(endPos[0], endPos[1], endPos[2]);

    if((nowPose - startPose).norm() < 1.0)
        isGoEnd = true;
    if((nowPose - endPose).norm() < 1.0)
        isGoEnd = false;

    if(!isGoEnd){
        Vector3d temp = startPose;
        startPose = endPose;
        endPose = temp;
    }

    double needTime = flightTime * (nowPose - endPose).norm() / (startPose - endPose).norm();
    vel = (endPose - nowPose) / needTime;
    pos = nowPose + cmdT * vel;
    acc = Vector3d(0,0,0);

    return true;
}
bool DynamicPlanner::getCmd(double t, Vector3d& pos, Vector3d& vel, Vector3d& acc, double& yaw, double& yaw_dot){

    return true;
}


}