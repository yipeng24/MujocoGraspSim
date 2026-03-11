#include <iostream>
#include <math.h>
#include <random>
#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/ArmAnglesState.h>
#include "quadrotor_msgs/PositionCommand.h"
#include "tf/transform_broadcaster.h"
#include <string.h>
#include "rotation_util/rotation_util.hpp"

using namespace std;
using rot_util = rotation_util::RotUtil;


ros::Subscriber _cmd_sub;
ros::Publisher  _odom_pub;

ros::Subscriber _arm_cmd_sub;
ros::Publisher  _arm_angles_pub;

quadrotor_msgs::PositionCommand _cmd;
double _init_x, _init_y, _init_z;

tf::TransformBroadcaster* broadcaster;

bool rcv_cmd = false;

Eigen::VectorXd _arm_angles_cmd, _arm_angles_cur, _last_arm_angles_cur;
bool _rcv_arm_angles_cmd = false;

void rcvPosCmdCallBack(const quadrotor_msgs::PositionCommand cmd)
{	
	rcv_cmd = true;
	_cmd    = cmd;

	//! 排除yaw
	// _cmd.yaw = 0.0;
}

void pubOdom()
{	
	nav_msgs::Odometry odom;
	odom.header.stamp    = ros::Time::now();
	odom.header.frame_id = "world";

	if(rcv_cmd)
	{
	    odom.pose.pose.position.x = _cmd.position.x;
	    odom.pose.pose.position.y = _cmd.position.y;
	    odom.pose.pose.position.z = _cmd.position.z;

		// 视频
		// _cmd.acceleration.x = _cmd.acceleration.y = 0.0;

		Eigen::Vector3d alpha = Eigen::Vector3d(_cmd.acceleration.x, _cmd.acceleration.y, _cmd.acceleration.z) + 9.8*Eigen::Vector3d(0,0,1);
		Eigen::Vector3d xC(cos(_cmd.yaw), sin(_cmd.yaw), 0);
		Eigen::Vector3d yC(-sin(_cmd.yaw), cos(_cmd.yaw), 0);
		Eigen::Vector3d xB = (yC.cross(alpha)).normalized();
		Eigen::Vector3d yB = (alpha.cross(xB)).normalized();
		Eigen::Vector3d zB = xB.cross(yB);
		Eigen::Matrix3d R;
		R.col(0) = xB;
		R.col(1) = yB;
		R.col(2) = zB;
		Eigen::Quaterniond q(R);
		const double atteny_coef = 0.5;
		Eigen::Vector3d rpy = rot_util::quaternion2euler(q);
		rpy.x() = rpy.x() * atteny_coef;
		rpy.y() = rpy.y() * atteny_coef;
		Eigen::Quaterniond q_fix = rot_util::euler2quaternion(rpy);
		q = q_fix;

	    odom.pose.pose.orientation.w = q.w();
	    odom.pose.pose.orientation.x = q.x();
	    odom.pose.pose.orientation.y = q.y();
	    odom.pose.pose.orientation.z = q.z();

	    odom.twist.twist.linear.x = _cmd.velocity.x;
	    odom.twist.twist.linear.y = _cmd.velocity.y;
	    odom.twist.twist.linear.z = _cmd.velocity.z;

	    odom.twist.twist.angular.x = _cmd.acceleration.x;
	    odom.twist.twist.angular.y = _cmd.acceleration.y;
	    odom.twist.twist.angular.z = _cmd.acceleration.z;

		// TF for raw sensor visualization
		if (1) {
			tf::Transform transform;
			transform.setOrigin(tf::Vector3(odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z));
			transform.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
			// cout << "q: " << q(0) << ", " << q(1) << ", " << q(2)<< ", " << q(3) << endl;  
			broadcaster->sendTransform(tf::StampedTransform(transform, ros::Time::now(), string("world"), string("base")));
		}

	}
	else
	{
		odom.pose.pose.position.x = _init_x;
	    odom.pose.pose.position.y = _init_y;
	    odom.pose.pose.position.z = _init_z;

	    odom.pose.pose.orientation.w = 1;
	    odom.pose.pose.orientation.x = 0;
	    odom.pose.pose.orientation.y = 0;
	    odom.pose.pose.orientation.z = 0;

	    odom.twist.twist.linear.x = 0.0;
	    odom.twist.twist.linear.y = 0.0;
	    odom.twist.twist.linear.z = 0.0;

	    odom.twist.twist.angular.x = 0.0;
	    odom.twist.twist.angular.y = 0.0;
	    odom.twist.twist.angular.z = 0.0;
	}

    _odom_pub.publish(odom);
}

void rcvArmAnglesCmdCallBack(const quadrotor_msgs::ArmAnglesState::ConstPtr& msg)
{
	// std::cout << "rcvArmAnglesCmdCallBack" << std::endl;
    _arm_angles_cmd.resize(msg->theta.size());
	_arm_angles_cmd.setZero();

    for (size_t i = 0; i < msg->theta.size(); ++i) {
        _arm_angles_cmd(i) = msg->theta[i];
    }

	if(!_rcv_arm_angles_cmd)
	{
		_arm_angles_cur = _last_arm_angles_cur = _arm_angles_cmd;
		_rcv_arm_angles_cmd = true;
	}

	// std::cout << "rcvArmAnglesCmdCallBack end" << std::endl;
}

void pubArmAnglesState()
{

    if (!_rcv_arm_angles_cmd) return;
	// std::cout << "pubArmAnglesState" << std::endl;

	double k = 0.1;
	_arm_angles_cur = (1-k)*_last_arm_angles_cur + k * _arm_angles_cmd;
	_last_arm_angles_cur = _arm_angles_cur;

	quadrotor_msgs::ArmAnglesState msg;
	msg.theta.resize(_arm_angles_cur.size());
	msg.dtheta.resize(_arm_angles_cur.size());
	for (size_t i = 0; i < _arm_angles_cur.size(); ++i) {
		msg.theta[i] = _arm_angles_cur(i);
		msg.dtheta[i] = 0.0;
	}

	_arm_angles_pub.publish(msg);
	// std::cout << "pubArmAnglesState end" << std::endl;
}

int main (int argc, char** argv) 
{        
    ros::init (argc, argv, "odom_generator");
    ros::NodeHandle nh( "~" );

    nh.param("init_x", _init_x,  0.0);
    nh.param("init_y", _init_y,  0.0);
    nh.param("init_z", _init_z,  0.0);

    _cmd_sub  = nh.subscribe( "command", 1, rcvPosCmdCallBack, ros::TransportHints().tcpNoDelay());
    _odom_pub = nh.advertise<nav_msgs::Odometry>("odometry", 1);    

	// ! pos2cmd not used anymore
	_arm_cmd_sub  = nh.subscribe( "/arm_angles_cmd", 1, rcvArmAnglesCmdCallBack, ros::TransportHints().tcpNoDelay());       
    _arm_angles_pub = nh.advertise<quadrotor_msgs::ArmAnglesState>("/arm_angles_cur", 1);

	tf::TransformBroadcaster b;
	broadcaster = &b;

	std::cout << "poscmd_2_odom started" << std::endl;

    ros::Rate rate(100);
    bool status = ros::ok();
    while(status) 
    {
		pubOdom();   
		pubArmAnglesState();                
        ros::spinOnce();
        status = ros::ok();
        rate.sleep();
    }

    return 0;
}