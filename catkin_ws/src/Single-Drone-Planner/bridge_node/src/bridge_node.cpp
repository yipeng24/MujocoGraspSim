// src/arm_bridge_node.cpp
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <ros/ros.h>
#include <iostream>

#include <std_msgs/Float64.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Bool.h>

#include <grasp_chassis_comm/GraspPIDCmdMsg.h> 
#include <grasp_chassis_comm/FTServoAllFeedbackRawMsg.h> 
#include <grasp_chassis_comm/LZMotorFeedbackMsg.h> 
#include <grasp_chassis_comm/GripperOperateMsg.h> 
#include <grasp_chassis_comm/SpeedLevelMsg.h> 

#include <quadrotor_msgs/SyncFrame.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/JointState.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

// --- 辅助函数：Eigen 与 Msg 转换 ---
Eigen::Isometry3d odom2isometry(const nav_msgs::Odometry& msg) {
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.translation() << msg.pose.pose.position.x,
                       msg.pose.pose.position.y,
                       msg.pose.pose.position.z;
    Eigen::Quaterniond q(msg.pose.pose.orientation.w,
                         msg.pose.pose.orientation.x,
                         msg.pose.pose.orientation.y,
                         msg.pose.pose.orientation.z);
    T.linear() = q.toRotationMatrix();
    return T;
}

nav_msgs::Odometry isometry2odom(const Eigen::Isometry3d& T, const nav_msgs::Odometry& ref_msg) {
    nav_msgs::Odometry odom = ref_msg;
    Eigen::Vector3d t = T.translation();
    Eigen::Quaterniond q(T.linear());
    
    odom.pose.pose.position.x = t.x();
    odom.pose.pose.position.y = t.y();
    odom.pose.pose.position.z = t.z();
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    return odom;
}

nav_msgs::Odometry transformOdom(
    const nav_msgs::Odometry& odom_in, 
    const Eigen::Isometry3d& T_ext, 
    const std::string& target_child_frame)
{
    nav_msgs::Odometry odom_out = odom_in;

    // 1. Pose 变换
    Eigen::Isometry3d T_W_B = Eigen::Isometry3d::Identity();
    T_W_B.translation() << odom_in.pose.pose.position.x,
                           odom_in.pose.pose.position.y,
                           odom_in.pose.pose.position.z;
    Eigen::Quaterniond q_in(odom_in.pose.pose.orientation.w,
                            odom_in.pose.pose.orientation.x,
                            odom_in.pose.pose.orientation.y,
                            odom_in.pose.pose.orientation.z);
    T_W_B.linear() = q_in.toRotationMatrix();

    Eigen::Isometry3d T_W_S = T_W_B * T_ext;

    Eigen::Vector3d t_out = T_W_S.translation();
    Eigen::Quaterniond q_out(T_W_S.linear());

    odom_out.pose.pose.position.x = t_out.x();
    odom_out.pose.pose.position.y = t_out.y();
    odom_out.pose.pose.position.z = t_out.z();
    odom_out.pose.pose.orientation.w = q_out.w();
    odom_out.pose.pose.orientation.x = q_out.x();
    odom_out.pose.pose.orientation.y = q_out.y();
    odom_out.pose.pose.orientation.z = q_out.z();

    // 2. Twist 变换
    Eigen::Vector3d v_b(odom_in.twist.twist.linear.x, odom_in.twist.twist.linear.y, odom_in.twist.twist.linear.z);
    Eigen::Vector3d w_b(odom_in.twist.twist.angular.x, odom_in.twist.twist.angular.y, odom_in.twist.twist.angular.z);
    
    Eigen::Matrix3d R = T_ext.rotation();
    Eigen::Vector3d t = T_ext.translation();

    Eigen::Vector3d w_s = R.transpose() * w_b;
    Eigen::Vector3d v_s = R.transpose() * (v_b + w_b.cross(t));

    odom_out.twist.twist.linear.x  = v_s.x();
    odom_out.twist.twist.linear.y  = v_s.y();
    odom_out.twist.twist.linear.z  = v_s.z();
    odom_out.twist.twist.angular.x = w_s.x();
    odom_out.twist.twist.angular.y = w_s.y();
    odom_out.twist.twist.angular.z = w_s.z();

    // 3. 修改 Frame ID
    odom_out.child_frame_id = target_child_frame;

    return odom_out;
}

class ArmBridgeNode
{
    public:
        ArmBridgeNode(ros::NodeHandle& nh)
        : nh_(nh)
        {
            // ----- 参数 -----
            nh_.param<std::string>("topic_arm_angles_cmd", topic_arm_angles_cmd_, "joint_state_cmd");
            nh_.param<std::string>("topic_arm_grab_cmd",   topic_arm_grab_cmd_,   "/arm_grab_cmd");
            nh_.param<std::string>("topic_grasp_cmd_out",  topic_grasp_cmd_out_,  "/grasp_chassis/grasp_pid_cmd");
            nh_.param<std::string>("topic_lz_motor_est",   topic_lz_motor_est_,   "/grasp_chassis/lz_motor_feedback");
            nh_.param<std::string>("topic_ft_servo_est",   topic_ft_servo_est_,   "/grasp_chassis/ft_servo_feedback");
            nh_.param<std::string>("topic_joint_state_est", topic_joint_state_est_, "/joint_state_est");
            nh_.param<std::string>("topic_operate_cmd",    topic_operate_cmd_,    "/grasp_chassis/gripper_operate_cmd");

            if (!nh_.getParam("joint_names", joint_names_)) {
                joint_names_ = {"arm_joint_pitch", "arm_joint_pitch2", "arm_joint_roll", "grab"};
            }
            nh_.param<double>("stuck_duration_threshold", stuck_duration_threshold_, 0.5);
            nh_.param<double>("protect_angle_offset", protect_angle_offset_, 0.2);
            
            pub_arm_thetas_cmd_ = nh_.advertise<grasp_chassis_comm::GraspPIDCmdMsg>(topic_grasp_cmd_out_, 100);
            pub_operate_cmd_    = nh_.advertise<grasp_chassis_comm::GripperOperateMsg>(topic_operate_cmd_, 100);
            pub_joint_state_est_ = nh_.advertise<sensor_msgs::JointState>(topic_joint_state_est_, 100);

            // ----- 订阅者 -----
            sub_arm_thetas_cmd_ = nh_.subscribe(topic_arm_angles_cmd_, 10,
                                                &ArmBridgeNode::armThetasCmdCb, this,
                                                ros::TransportHints().tcpNoDelay());
            sub_arm_grab_cmd_   = nh_.subscribe(topic_arm_grab_cmd_, 10,
                                                &ArmBridgeNode::armGrabCmdCb, this,
                                                ros::TransportHints().tcpNoDelay());

            sub_lz_motor_est_ = nh_.subscribe(topic_lz_motor_est_, 10,
                                              &ArmBridgeNode::lzMotorEstCb, this);
            sub_ft_servo_est_ = nh_.subscribe(topic_ft_servo_est_, 10,
                                              &ArmBridgeNode::ftServoEstCb, this);

            // 初始化 JointState 缓存
            est_joint_state_.name = joint_names_; // 使用参数中定义的名称
            est_joint_state_.position.resize(joint_names_.size(), 0.0);
            est_joint_state_.velocity.resize(joint_names_.size(), 0.0);

            ROS_INFO("[arm_bridge] Node initialized with feedback subscribers");

            ROS_INFO("[arm_bridge] Node initialized");
        }

    private:
        void armThetasCmdCb(const sensor_msgs::JointState::ConstPtr& msg)
        {
            const size_t joint_num = msg->position.size();
            if (joint_num == 0) return;

            if (msg->position.size() >= 1) {
                grasp_chassis_comm::GraspPIDCmdMsg motor_cmd;
                motor_cmd.header.stamp = ros::Time::now();
                motor_cmd.id = 127;
                motor_cmd.manufacturer.name = grasp_chassis_comm::ManufacturerMsg::LZMOTOR;
                motor_cmd.speed     = msg->velocity[0]; 
                motor_cmd.position  = msg->position[0];
                pub_arm_thetas_cmd_.publish(motor_cmd);
            }

            if (msg->position.size() >= 2) {
                grasp_chassis_comm::GraspPIDCmdMsg s1_cmd;
                s1_cmd.header.stamp = ros::Time::now();
                s1_cmd.id = 1;
                s1_cmd.manufacturer.name = grasp_chassis_comm::ManufacturerMsg::FTSERVO;
                s1_cmd.speed     = msg->velocity[1];
                s1_cmd.position  = msg->position[1];
                pub_arm_thetas_cmd_.publish(s1_cmd);
            }

            if (msg->position.size() >= 3) {
                grasp_chassis_comm::GraspPIDCmdMsg s2_cmd;
                s2_cmd.header.stamp = ros::Time::now();
                s2_cmd.id = 2;
                s2_cmd.manufacturer.name = grasp_chassis_comm::ManufacturerMsg::FTSERVO;
                s2_cmd.speed     = msg->velocity[2];
                s2_cmd.position  = msg->position[2];
                pub_arm_thetas_cmd_.publish(s2_cmd);
            }
        }

        void armGrabCmdCb(const std_msgs::Bool::ConstPtr& msg)
        {
            grab_requested_ = msg->data;
            is_stuck_protected_ = false; // 每次新指令重置保护状态
            
            grasp_chassis_comm::GripperOperateMsg operate_msg;
            operate_msg.header.stamp = ros::Time::now();
            operate_msg.id = 3;
            operate_msg.speed.level = grasp_chassis_comm::SpeedLevelMsg::FAST;
            
            // 初始尝试：如果是抓取发 0.0，释放发 1.0
            operate_msg.angle = grab_requested_ ? 0.0 : 1.6;
            pub_operate_cmd_.publish(operate_msg);
            
            last_grab_pos_ = -1.0; // 重置位置记录

            std::cout << "[arm_bridge] Gripper command: " 
                      << (grab_requested_ ? "GRAB" : "RELEASE") << std::endl;
        }

        // LZMotor 反馈回调 (对应 id 127)
        void lzMotorEstCb(const grasp_chassis_comm::LZMotorFeedbackMsg::ConstPtr& msg)
        {
            // 更新第 0 个关节 (arm_joint_pitch)
            est_joint_state_.header.stamp = ros::Time::now();
            if(est_joint_state_.position.size() >= 1) {
                est_joint_state_.position[0] = msg->position;
                est_joint_state_.velocity[0] = msg->speed;
            }
            pub_joint_state_est_.publish(est_joint_state_);
        }

        // FTServo 反馈回调 (包含多个 servo)
        void ftServoEstCb(const grasp_chassis_comm::FTServoAllFeedbackRawMsg::ConstPtr& msg)
        {
            est_joint_state_.header.stamp = ros::Time::now();
            for (const auto& s : msg->servos) {
                if (s.id == 1 && est_joint_state_.position.size() >= 2) {
                    est_joint_state_.position[1] = s.position;
                    est_joint_state_.velocity[1] = s.speed;
                }
                else if (s.id == 2 && est_joint_state_.position.size() >= 3) {
                    est_joint_state_.position[2] = s.position;
                    est_joint_state_.velocity[2] = s.speed;
                }
                // 如果有第 4 个关节 (grab)
                else if (s.id == 3 && est_joint_state_.position.size() >= 4) {

                    double current_pos = s.position;
                    est_joint_state_.position[3] = current_pos;


                    // --- 过流保护逻辑 ---
                    if (grab_requested_) {
                        // 1. 判断是否“还没到达目标”且“位置几乎不动”
                        // 假设目标是 0.0，且位置变动极小 (小于 0.005)
                        bool is_not_moving = (std::abs(current_pos - last_grab_pos_) < 0.005);
                        bool not_at_target = (current_pos > 0.05); // 还没完全闭合

                        if (not_at_target && is_not_moving) {
                            if (last_grab_pos_ < 0) { // 第一次记录
                                grab_stuck_start_time_ = ros::Time::now();
                            } else {
                                // 2. 检查停滞持续时间是否超过阈值
                                double stuck_duration = (ros::Time::now() - grab_stuck_start_time_).toSec();
                                if (stuck_duration > stuck_duration_threshold_ && !is_stuck_protected_) {
                                    ROS_WARN("[arm_bridge] Gripper stuck detected for %.2fs, triggering protection.", stuck_duration);
                                    
                                    // 3. 触发保护：发布当前位置略微深入一点点的值 (维持压力但不堵转)
                                    grasp_chassis_comm::GripperOperateMsg protect_msg;
                                    protect_msg.header.stamp = ros::Time::now();
                                    protect_msg.id = 3;
                                    protect_msg.speed.level = grasp_chassis_comm::SpeedLevelMsg::SLOW;
                                    
                                    // 目标设为当前实际位置再往里压一定偏移量 (根据实际调整)
                                    protect_msg.angle = std::max(0.0, current_pos - protect_angle_offset_);
                                    pub_operate_cmd_.publish(protect_msg);
                                    
                                    is_stuck_protected_ = true; // 标记已进入保护，避免频繁发包
                                }
                            }
                        } else {
                            // 如果位置还在动或者已经松开，重置计时器
                            grab_stuck_start_time_ = ros::Time::now();
                            is_stuck_protected_ = false;
                        }
                        last_grab_pos_ = current_pos;
                    }
                }

            }
            pub_joint_state_est_.publish(est_joint_state_);
        }

    private:
        ros::NodeHandle nh_;
        std::string topic_arm_angles_cmd_;
        std::string topic_arm_grab_cmd_;
        std::string topic_grasp_cmd_out_;
        std::string topic_lz_motor_est_;
        std::string topic_ft_servo_est_;
        std::string topic_joint_state_est_;
        std::string topic_operate_cmd_;
        ros::Subscriber sub_arm_thetas_cmd_;
        ros::Subscriber sub_arm_grab_cmd_;
        ros::Publisher  pub_arm_thetas_cmd_;
        ros::Publisher  pub_operate_cmd_;
        std::vector<std::string> joint_names_;

        ros::Subscriber sub_lz_motor_est_;
        ros::Subscriber sub_ft_servo_est_;
        ros::Publisher  pub_joint_state_est_;

        // [新增] 状态缓存
        sensor_msgs::JointState est_joint_state_;

        bool grab_requested_ = false;       // 当前是否处于抓取请求状态
        double last_grab_pos_ = -1.0;       // 记录上一次的实际位置
        ros::Time grab_stuck_start_time_;   // 开始卡住的时间点
        bool is_stuck_protected_ = false;   // 是否已经进入过流保护模式
        double stuck_duration_threshold_;   // 卡住时长阈值 (秒)
        double protect_angle_offset_;       // 保护模式角度偏移量

};

class ImgSyncNode
{
    public:
        ImgSyncNode(ros::NodeHandle& nh)
        : nh_(nh)
        {
            nh_.param<std::string>("depth_topic", depth_topic_, "/depth_image");
            nh_.param<std::string>("rgb_topic",   rgb_topic_,   "/rgb_image");
            nh_.param<std::string>("end_pose_topic",  end_pose_topic_,  "/px4ctrl/end_pose_est");
            nh_.param<std::string>("out_topic",   out_topic_,   "/sync_frame");
            nh_.param<int>("queue_size",          queue_size_,  20);
            nh_.param<double>("slop",             slop_sec_,    0.008); 
            nh_.param<std::string>("cam_pose_topic",  cam_pose_topic_,  std::string("/sync_cam_pose"));
            nh_.param<bool>("en_vis", en_vis_, false);

            // 相机内参 (如果启用可视化需要)
            nh_.param<double>("fx", fx_, 615.0);
            nh_.param<double>("fy", fy_, 615.0);
            nh_.param<double>("cx", cx_, 320.0);
            nh_.param<double>("cy", cy_, 240.0);

            std::vector<double> T_E_C_list;
            T_E_C_ = Eigen::Isometry3d::Identity();
            if (nh_.getParam("T_E_C", T_E_C_list) && T_E_C_list.size() == 16) {
                T_E_C_.matrix() = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(T_E_C_list.data());
                ROS_INFO_STREAM("[img_sync_node] T_E_C loaded:\n" << T_E_C_.matrix());
            } else {
                ROS_WARN("[img_sync_node] Param 'T_E_C' not found or invalid. Using Identity.");
            }

            pub_ = nh_.advertise<quadrotor_msgs::SyncFrame>(out_topic_, 10);
            pub_cam_pose_ = nh_.advertise<nav_msgs::Odometry>(cam_pose_topic_, 10);
            
            if (en_vis_) {
                pub_depth_pcl_ = nh_.advertise<sensor_msgs::PointCloud2>("/depth_point_cloud_world", 10);
                ROS_INFO("[img_sync_node] Depth visualization enabled, publishing to /depth_point_cloud_world");
            }

            depth_sub_.subscribe(nh_, depth_topic_, queue_size_);
            rgb_sub_.subscribe(nh_,   rgb_topic_,   queue_size_);
            odom_sub_.subscribe(nh_,  end_pose_topic_,  queue_size_);

            typedef message_filters::sync_policies::ApproximateTime<
                sensor_msgs::Image, sensor_msgs::Image, nav_msgs::Odometry> SyncPolicy;

            sync_.reset(new message_filters::Synchronizer<SyncPolicy>(
                SyncPolicy(queue_size_), depth_sub_, rgb_sub_, odom_sub_));
            sync_->setInterMessageLowerBound(0, ros::Duration(0.0)); 
            sync_->registerCallback(boost::bind(&ImgSyncNode::cb, this, _1, _2, _3));

            ROS_INFO_STREAM("[img_sync_node] Subscribing:\n  depth: " << depth_topic_
                            << "\n  rgb:   " << rgb_topic_
                            << "\n  end_pose:  " << end_pose_topic_
                            << "\nPublishing: " << out_topic_);
        }

    private:
        void cb(const sensor_msgs::ImageConstPtr& depth,
                const sensor_msgs::ImageConstPtr& rgb,
                const nav_msgs::OdometryConstPtr& odom)
        {
            std::cout << "ImgSyncNode::cb called" << std::endl;
            quadrotor_msgs::SyncFrame out;
            
            const ros::Time& t_depth = depth->header.stamp;
            const ros::Time& t_rgb   = rgb->header.stamp;
            const ros::Time& t_odom  = odom->header.stamp;
            ros::Time t_max = std::max({t_depth, t_rgb, t_odom});
            
            out.header.stamp = t_max;
            if (!odom->header.frame_id.empty())      out.header.frame_id = odom->header.frame_id;
            else if (!rgb->header.frame_id.empty())  out.header.frame_id = rgb->header.frame_id;
            else                                     out.header.frame_id = depth->header.frame_id;

            out.depth = *depth;
            out.rgb   = *rgb;
            out.body_odom = *odom;

            out.cam_pose = transformOdom(*odom, T_E_C_, "camera_link");

            pub_.publish(out);
            pub_cam_pose_.publish(out.cam_pose);
            
            // 可视化：发布深度点云投影到世界坐标系
            if (en_vis_) {
                publishDepthPointCloud(depth, out.cam_pose);
            }
        }

    private:
        void publishDepthPointCloud(const sensor_msgs::ImageConstPtr& depth_msg,
                                   const nav_msgs::Odometry& cam_pose)
        {
            try {
                // 转换深度图像
                cv_bridge::CvImageConstPtr depth_ptr = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
                const cv::Mat& depth_image = depth_ptr->image;
                
                // 创建点云消息
                sensor_msgs::PointCloud2 pcl_msg;
                pcl_msg.header.stamp = depth_msg->header.stamp;
                pcl_msg.header.frame_id = "world";
                pcl_msg.height = 1;
                pcl_msg.is_dense = false;
                pcl_msg.is_bigendian = false;
                
                // 设置点云字段
                sensor_msgs::PointCloud2Modifier modifier(pcl_msg);
                modifier.setPointCloud2Fields(3,
                    "x", 1, sensor_msgs::PointField::FLOAT32,
                    "y", 1, sensor_msgs::PointField::FLOAT32,
                    "z", 1, sensor_msgs::PointField::FLOAT32);
                
                // 获取相机位姿
                Eigen::Isometry3d T_W_C = odom2isometry(cam_pose);
                
                // 预估点云大小并预分配
                std::vector<Eigen::Vector3d> points;
                points.reserve(depth_image.rows * depth_image.cols / 10); // 大约10%的点有效
                
                // 深度图转3D点云
                for (int v = 0; v < depth_image.rows; v++) {
                    for (int u = 0; u < depth_image.cols; u++) {
                        uint16_t depth_value = depth_image.at<uint16_t>(v, u);
                        if (depth_value == 0 || depth_value > 10000) continue; // 过滤无效深度
                        
                        // 深度值从毫米转换为米
                        double depth = depth_value / 1000.0;
                        
                        // 像素坐标转相机坐标系
                        Eigen::Vector3d p_C;
                        p_C.x() = (u - cx_) * depth / fx_;
                        p_C.y() = (v - cy_) * depth / fy_;
                        p_C.z() = depth;
                        
                        // 转换到世界坐标系
                        Eigen::Vector3d p_W = T_W_C * p_C;
                        points.push_back(p_W);
                    }
                }
                
                // 设置点云数据
                modifier.resize(points.size());
                sensor_msgs::PointCloud2Iterator<float> iter_x(pcl_msg, "x");
                sensor_msgs::PointCloud2Iterator<float> iter_y(pcl_msg, "y");
                sensor_msgs::PointCloud2Iterator<float> iter_z(pcl_msg, "z");
                
                for (const auto& p : points) {
                    *iter_x = p.x();
                    *iter_y = p.y();
                    *iter_z = p.z();
                    ++iter_x; ++iter_y; ++iter_z;
                }
                
                pcl_msg.width = points.size();
                pcl_msg.row_step = pcl_msg.point_step * pcl_msg.width;
                
                pub_depth_pcl_.publish(pcl_msg);
            }
            catch (const cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception: %s", e.what());
            }
        }

        ros::NodeHandle nh_;
        std::string depth_topic_, rgb_topic_, end_pose_topic_, out_topic_;
        std::string cam_pose_topic_;

        int queue_size_;
        double slop_sec_;

        Eigen::Isometry3d T_E_C_; 

        ros::Publisher pub_;
        ros::Publisher pub_cam_pose_;
        ros::Publisher pub_depth_pcl_;
        
        bool en_vis_;
        double fx_, fy_, cx_, cy_;

        message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
        message_filters::Subscriber<sensor_msgs::Image> rgb_sub_;
        message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
        boost::shared_ptr< message_filters::Synchronizer<
            message_filters::sync_policies::ApproximateTime<
                sensor_msgs::Image, sensor_msgs::Image, nav_msgs::Odometry> > > sync_;
};


class PclSyncNode
{
public:
    explicit PclSyncNode(ros::NodeHandle& nh)
    : nh_(nh)
    {
        nh_.param<std::string>("tgt_topic",       tgt_topic_,       std::string("/tgt_pcl"));
        nh_.param<std::string>("env_topic",       env_topic_,       std::string("/env_pcl"));
        nh_.param<std::string>("odom_topic",      odom_topic_,      std::string("/body_odom"));
        nh_.param<std::string>("out_topic",       out_topic_,       std::string("/sync_frame"));
        nh_.param<std::string>("cam_pose_topic",  cam_pose_topic_,  std::string("/sync_cam_pose"));
        
        // 是否使用世界系点云（不需要转换，也不需要订阅 odom）
        nh_.param<bool>("use_world_frame_pcl", use_world_frame_pcl_, false);
        
        nh_.param<int>        ("queue_size",      queue_size_,      50);
        nh_.param<double>     ("slop",            slop_sec_,        0.01);

        std::vector<double> T_B_C_list;
        T_B_C_ = Eigen::Isometry3d::Identity();
        if (nh_.getParam("T_B_C", T_B_C_list) && T_B_C_list.size() == 16) {
            T_B_C_.matrix() = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(T_B_C_list.data());
            ROS_INFO_STREAM("[pcl_sync_node] T_B_C loaded:\n" << T_B_C_.matrix());
        } else {
            if (!use_world_frame_pcl_) {
                ROS_WARN("[pcl_sync_node] Param 'T_B_C' not found or invalid. Using Identity.");
            }
        }

        pub_ = nh_.advertise<quadrotor_msgs::SyncFrame>(out_topic_, 10);

        if (use_world_frame_pcl_) {
            // 世界系点云模式：分开订阅，env_pcl 作为缓存，tgt_pcl 触发发布
            env_pcl_init_ = false;
            
            sub_env_world_ = nh_.subscribe(env_topic_, queue_size_, &PclSyncNode::cb_env_world, this);
            sub_tgt_world_ = nh_.subscribe(tgt_topic_, queue_size_, &PclSyncNode::cb_tgt_world, this);

            ROS_INFO_STREAM("[pcl_sync_node] Init done (World Frame Mode).\n"
                            << "  Sub tgt: " << tgt_topic_ << "\n"
                            << "  Sub env: " << env_topic_ << "\n"
                            << "  Pub Sync: " << out_topic_);
        } else {
            // 相机系点云模式：订阅点云 + odom，使用同步器
            pub_cam_pose_ = nh_.advertise<nav_msgs::Odometry>(cam_pose_topic_, 10);

            tgt_sub_.subscribe(nh_, tgt_topic_, queue_size_);
            env_sub_.subscribe(nh_, env_topic_, queue_size_);
            odom_sub_.subscribe(nh_, odom_topic_, queue_size_);

            typedef message_filters::sync_policies::ApproximateTime<
                sensor_msgs::PointCloud2,
                sensor_msgs::PointCloud2,
                nav_msgs::Odometry> SyncPolicy3;

            sync3_.reset(new message_filters::Synchronizer<SyncPolicy3>(
                SyncPolicy3(queue_size_), tgt_sub_, env_sub_, odom_sub_));

            sync3_->setMaxIntervalDuration(ros::Duration(slop_sec_));
            sync3_->registerCallback(boost::bind(&PclSyncNode::cb_camera, this, _1, _2, _3));

            ROS_INFO_STREAM("[pcl_sync_node] Init done (Camera Frame Mode).\n"
                            << "  Sub tgt: " << tgt_topic_ << "\n"
                            << "  Sub env: " << env_topic_ << "\n"
                            << "  Sub odom: " << odom_topic_ << "\n"
                            << "  Pub Sync: " << out_topic_);
        }
    }

private:
    // 世界系 env_pcl 回调：持续更新缓存
    void cb_env_world(const sensor_msgs::PointCloud2::ConstPtr& env)
    {
        cached_env_pcl_ = *env;
        env_pcl_init_ = true;
    }

    // 世界系 tgt_pcl 回调：触发发布
    void cb_tgt_world(const sensor_msgs::PointCloud2::ConstPtr& tgt)
    {
        if (!env_pcl_init_) {
            ROS_WARN_THROTTLE(1.0, "[pcl_sync_node] env_pcl not initialized yet, skipping tgt_pcl");
            return;
        }

        quadrotor_msgs::SyncFrame out;
        out.header.stamp = tgt->header.stamp;
        out.header.frame_id = "world";

        // 使用当前 tgt_pcl 和缓存的 env_pcl
        out.tgt_pcl = *tgt;
        out.env_pcl = cached_env_pcl_;

        pub_.publish(out);
    }

    // 相机系点云回调（需要 odom 进行转换）
    void cb_camera(const sensor_msgs::PointCloud2ConstPtr& tgt,
                   const sensor_msgs::PointCloud2ConstPtr& env,
                   const nav_msgs::OdometryConstPtr& body_odom)
    {
        quadrotor_msgs::SyncFrame out;

        const ros::Time& t1 = tgt->header.stamp;
        const ros::Time& t2 = env->header.stamp;
        const ros::Time& t3 = body_odom->header.stamp;
        ros::Time t_max = std::max({t1, t2, t3});

        out.header.stamp = t_max;
        if (!body_odom->header.frame_id.empty()) out.header.frame_id = body_odom->header.frame_id;
        else if (!tgt->header.frame_id.empty())  out.header.frame_id = tgt->header.frame_id;
        else                                     out.header.frame_id = env->header.frame_id;

        out.tgt_pcl = *tgt;
        out.env_pcl = *env;
        out.body_odom = *body_odom;
        out.cam_pose = transformOdom(*body_odom, T_B_C_, "camera_link");

        pub_.publish(out);
        pub_cam_pose_.publish(out.cam_pose);
    }

    ros::NodeHandle nh_;
    std::string tgt_topic_, env_topic_, odom_topic_, out_topic_;
    std::string cam_pose_topic_;

    int queue_size_;
    double slop_sec_;
    bool use_world_frame_pcl_;

    Eigen::Isometry3d T_B_C_; 

    ros::Publisher pub_;
    ros::Publisher pub_cam_pose_;

    // 世界系模式：独立订阅
    ros::Subscriber sub_env_world_;
    ros::Subscriber sub_tgt_world_;
    bool env_pcl_init_;  // env_pcl 是否已初始化
    sensor_msgs::PointCloud2 cached_env_pcl_;  // 缓存的 env_pcl

    // 相机系模式：使用同步器
    message_filters::Subscriber<sensor_msgs::PointCloud2> tgt_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> env_sub_;
    message_filters::Subscriber<nav_msgs::Odometry>       odom_sub_;
    
    boost::shared_ptr< message_filters::Synchronizer<
        message_filters::sync_policies::ApproximateTime<
            sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, nav_msgs::Odometry> > > sync3_;  // 相机系模式
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "bridge_node");
    ros::NodeHandle nh("~");

    bool enable_img_sync;
    bool enable_pcl_sync;
    bool enable_arm_bridge;
    nh.param("enable_img_sync", enable_img_sync, false);
    nh.param("enable_pcl_sync", enable_pcl_sync, false);
    nh.param("enable_arm_bridge", enable_arm_bridge, false);

    //   TODO enable arm bridge node
    std::unique_ptr<ArmBridgeNode> node1;
    std::unique_ptr<ImgSyncNode> node2;
    std::unique_ptr<PclSyncNode> node3;

    if (enable_arm_bridge) {
        node1.reset(new ArmBridgeNode(nh));
    } else {
        ROS_INFO("[bridge_node] ArmBridgeNode disabled.");
    }


    if (enable_img_sync) {
    node2.reset(new ImgSyncNode(nh));
        ROS_INFO("[bridge_node] ImgSyncNode enabled.");
    } else {
        ROS_INFO("[bridge_node] ImgSyncNode disabled.");
    }

    if (enable_pcl_sync) {
        node3.reset(new PclSyncNode(nh));
        ROS_INFO("[bridge_node] PclSyncNode enabled.");
    } else {
        ROS_INFO("[bridge_node] PclSyncNode disabled.");
    }

    ros::spin();
    return 0;
}