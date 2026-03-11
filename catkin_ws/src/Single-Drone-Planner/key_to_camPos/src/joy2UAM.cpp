#include <Eigen/Eigen>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Joy.h>
#include "tf/tf.h"
#include <quadrotor_msgs/UAMFullState.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/TransformStamped.h>
#include <quadrotor_msgs/GetBasePose.h>

#define CH_FORWARD 0
#define CH_LEFT 1
#define CH_UP 2

#define CH_YAW 4
#define CH_PITCH 5
#define CH_PITCH2 6
#define CH_ROLL 7

#define BUTTON_CANCEL 0
#define BUTTON_CONFIRM 1
#define BUTTON_LOCK 6

ros::Publisher uam_goal_vis_pub_, uam_goal_cmd_pub_, uam_goal_waypoint_pub_;
ros::Subscriber joy_sub_, vicon_sub_;
ros::ServiceClient get_body_pose_client_;
ros::Timer uam_vis_timer_;

int uam_vis_rate_ = 50.0;
double uam_vis_cyc_;
int run_mode_ = 0; 

quadrotor_msgs::UAMFullState uam_state_msg_;
sensor_msgs::Joy rcv_joy_msg_;
geometry_msgs::Pose vicon_target_pose_;
geometry_msgs::Pose locked_end_pose_; // 锁定后的目标位姿

bool have_joy_ = false;
bool have_vicon_ = false;
bool is_locked_ = false;        // 锁定标志
bool last_confirm_state_ = false; 
bool last_cancel_state_ = false;
bool last_lock_state_ = false;

double yaw_ = 0;
Eigen::Vector3d thetas_ = Eigen::Vector3d::Zero();

// 角度限幅
void thetasLimit(Eigen::Vector3d& thetas) {
    Eigen::Vector3d thetas_low, thetas_high;
    thetas_low << 0, -M_PI/2, -0.8*M_PI;
    thetas_high << M_PI/2, M_PI/2, 0.8*M_PI;
    for (int i = 0; i < 3; ++i) {
        if (thetas[i] < thetas_low[i])  thetas[i] = thetas_low[i];
        if (thetas[i] > thetas_high[i]) thetas[i] = thetas_high[i];
    }
}

void rcvJoyCallbck(const sensor_msgs::Joy &joy_msg) {
    have_joy_ = true;
    rcv_joy_msg_ = joy_msg;

    // --- 按钮逻辑：锁定与发布 ---
    bool current_lock = joy_msg.buttons.at(BUTTON_LOCK) > 0.5;
    if (current_lock && !last_lock_state_) {
        if (run_mode_ == 1 || run_mode_ == 2) {
            if (have_vicon_) {
                locked_end_pose_ = vicon_target_pose_; // 捕获当前 Vicon 位姿
                is_locked_ = true;
                ROS_INFO("\033[32m[Mode %d] End Pose LOCKED. Adjust thetas then Confirm to publish.\033[0m", run_mode_);
            } else {
                 ROS_WARN("[Mode %d] Cannot lock: No Vicon data.", run_mode_);
            }
        }
    }
    last_lock_state_ = current_lock;

    bool current_confirm = joy_msg.buttons.at(BUTTON_CONFIRM) > 0.5;
    if (current_confirm && !last_confirm_state_) { 
        if (run_mode_ == 2) {
            uam_goal_waypoint_pub_.publish(uam_state_msg_);
        } else {
            uam_goal_cmd_pub_.publish(uam_state_msg_);
        }
        ROS_INFO("\033[36m[Mode %d] MISSION SENT!\033[0m", run_mode_);
    }
    last_confirm_state_ = current_confirm;

    // --- Cancel 逻辑 ---
    bool current_cancel = joy_msg.buttons.at(BUTTON_CANCEL) > 0.5;
    if (current_cancel && !last_cancel_state_) {
        if (is_locked_) {
            is_locked_ = false;
            ROS_INFO("\033[33m[Mode 1] UNLOCKED. Follow Vicon again.\033[0m");
        }
    }
    last_cancel_state_ = current_cancel;
}

void viconCallbck(const geometry_msgs::TransformStampedConstPtr& msg) {
    have_vicon_ = true;
    vicon_target_pose_.position.x = msg->transform.translation.x;
    vicon_target_pose_.position.y = msg->transform.translation.y;
    vicon_target_pose_.position.z = msg->transform.translation.z;
    vicon_target_pose_.orientation = msg->transform.rotation;
}

void UAMStatePubCallbck(const ros::TimerEvent& event) {
    uam_state_msg_.header.stamp = ros::Time::now();

    if (run_mode_ == 0) {
        // ... (保持原有的手动增量逻辑)
        if(!have_joy_) return;
        Eigen::Vector3d T_b, T_w;
        T_b << rcv_joy_msg_.axes.at(CH_FORWARD), rcv_joy_msg_.axes.at(CH_LEFT), rcv_joy_msg_.axes.at(CH_UP);
        T_b *= uam_vis_cyc_;
        Eigen::Matrix2d R_b2w_2d;
        R_b2w_2d << cos(yaw_), -sin(yaw_), sin(yaw_), cos(yaw_);
        T_w.head(2) = R_b2w_2d * T_b.head(2);
        T_w.z() = T_b.z();
        uam_state_msg_.pose.position.x += T_w.x();
        uam_state_msg_.pose.position.y += T_w.y();
        uam_state_msg_.pose.position.z += T_w.z();
        yaw_ += uam_vis_cyc_ * rcv_joy_msg_.axes.at(CH_YAW);
        Eigen::Quaterniond q_b2w(Eigen::AngleAxisd(yaw_, Eigen::Vector3d::UnitZ()));
        uam_state_msg_.pose.orientation.w = q_b2w.w();
        uam_state_msg_.pose.orientation.x = q_b2w.x();
        uam_state_msg_.pose.orientation.y = q_b2w.y();
        uam_state_msg_.pose.orientation.z = q_b2w.z();
        Eigen::Vector3d thetas_add;
        thetas_add << rcv_joy_msg_.axes.at(CH_PITCH), rcv_joy_msg_.axes.at(CH_PITCH2), rcv_joy_msg_.axes.at(CH_ROLL);
        thetas_ += thetas_add * uam_vis_cyc_;
        thetasLimit(thetas_);
    } else if (run_mode_ == 1 || run_mode_ == 2) {
        if (!have_joy_) return;

        geometry_msgs::Pose current_target = is_locked_ ? locked_end_pose_ : vicon_target_pose_;
        if (!is_locked_ && !have_vicon_) return;

        // 1. 获取旋转矩阵
        Eigen::Quaterniond q_v(current_target.orientation.w, current_target.orientation.x,
                                current_target.orientation.y, current_target.orientation.z);
        Eigen::Matrix3d R = q_v.normalized().toRotationMatrix();

        // 2. 手动提取欧拉角 (基于 Z-Y-X 顺序)
        // Pitch (theta): asin(-R20), 范围 [-pi/2, pi/2]
        double raw_pitch = std::asin(-R(2, 0));
        
        // Yaw (psi): atan2(R10, R00), 范围 [-pi, pi]
        double target_yaw = std::atan2(R(1, 0), R(0, 0));
        
        // Roll (phi): atan2(R21, R22), 范围 [-pi, pi]
        double raw_roll = std::atan2(R(2, 1), R(2, 2));

        // 3. 强制执行你要求的范围限制
        // Pitch 限制在 [0, 0.5 * PI]
        double target_pitch = std::max(0.0, std::min(M_PI / 2.0, raw_pitch));
        
        // Roll 限制在 [-0.5 * PI, 0.5 * PI]
        double target_roll = std::max(-M_PI , std::min(M_PI, raw_roll));

        // 4. 实时增量调整 theta1 (thetas_[0])
        double theta1_speed = 1.0; 
        thetas_[0] += -rcv_joy_msg_.axes.at(CH_PITCH) * theta1_speed * uam_vis_cyc_;
        
        // 自动补偿 theta2 以满足约束: theta1 + theta2 = target_pitch
        thetas_[1] = target_pitch - thetas_[0];
        thetas_[2] = target_roll;

        // 物理限幅处理
        thetasLimit(thetas_);
        // 修正 theta2 确保末端 Pitch 姿态准确
        thetas_[1] = target_pitch - thetas_[0]; 

        // 5. 调用服务
        quadrotor_msgs::GetBasePose srv;
        srv.request.thetas = {thetas_[0], thetas_[1], thetas_[2]};
        srv.request.end_pose = current_target;

        if (get_body_pose_client_.call(srv) && srv.response.success) {
            uam_state_msg_.pose = srv.response.body_pose;
            uam_state_msg_.ee_pose = current_target;
        } else {
            ROS_WARN_THROTTLE(1.0, "IK Service failed. Target may be out of reach.");
        }
    }

    // 更新 FullState 消息并发布可视化 (用于 Rviz 预览)
    uam_state_msg_.theta[0] = thetas_.x();
    uam_state_msg_.theta[1] = thetas_.y();
    uam_state_msg_.theta[2] = thetas_.z();
    uam_goal_vis_pub_.publish(uam_state_msg_);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "joy2UAM");
    ros::NodeHandle nh("~");
    nh.param("run_mode", run_mode_, 0);
    
    // 初始化数组
    thetas_.setZero();
    for(int i=0; i<4; ++i) {
        uam_state_msg_.theta.push_back(0);
        uam_state_msg_.dtheta.push_back(0);
    }
    uam_state_msg_.header.frame_id = "world";

    joy_sub_ = nh.subscribe("/joy", 1, rcvJoyCallbck);
    uam_goal_vis_pub_ = nh.advertise<quadrotor_msgs::UAMFullState>("uam_state_goal_vis", 1);
    uam_goal_cmd_pub_ = nh.advertise<quadrotor_msgs::UAMFullState>("uam_state_goal_cmd", 1);
    uam_goal_waypoint_pub_ = nh.advertise<quadrotor_msgs::UAMFullState>("uam_state_waypoint", 1);

    if (run_mode_ == 1 || run_mode_ == 2) {
        vicon_sub_ = nh.subscribe("target_pose", 1, viconCallbck);
        get_body_pose_client_ = nh.serviceClient<quadrotor_msgs::GetBasePose>("get_base_pose");
        get_body_pose_client_.waitForExistence();
    }

    uam_vis_cyc_ = 1.0 / (double)uam_vis_rate_;
    uam_vis_timer_ = nh.createTimer(ros::Duration(uam_vis_cyc_), &UAMStatePubCallbck);

    ros::spin();
    return 0;
}