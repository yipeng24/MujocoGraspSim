#ifndef _CH_RC_SDF_H_
#define _CH_RC_SDF_H_

#include <Eigen/Eigen>
#include <Eigen/StdVector>

#include <queue>
#include <ros/ros.h>
#include <tuple>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <random>  
#include <iostream>

#include <tf/tf.h>

#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>
#include <ch_rc_sdf/visualization.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <quadrotor_msgs/ArmAnglesState.h>
#include <quadrotor_msgs/ArmAngleState.h>
#include <quadrotor_msgs/UAMFullState.h>
#include <sensor_msgs/JointState.h>
#include "nav_msgs/Odometry.h"

// [NEW] 引入新的运动学和SDF地图头文件
#include "ch_rc_sdf/robot_kinematics.hpp"
#include "ch_rc_sdf/sdf_map.hpp"
#include <quadrotor_msgs/GetBasePose.h> // 假设包名为 ch_rc_sdf

namespace clutter_hand {

    struct FieldData;
    struct DiskParams;
    struct LineParams;
    using PublisherMap = std::unordered_map<std::string, ros::Publisher>;
    
    class CH_RC_SDF {
    public:
        CH_RC_SDF();
        ~CH_RC_SDF();

        void initMap(ros::NodeHandle &nh, bool init_field = true, bool vis_en = true);

        // [NEW] 获取运动学共享指针的接口
        RobotKinematics::Ptr getKinematicsPtr() { return kine_ptr_; }
        
        // [NEW] 设置运动学共享指针
        void setKinePtr(RobotKinematics::Ptr kine_ptr) { kine_ptr_ = kine_ptr; }
        
        // [NEW] 获取SDF地图共享指针的接口
        SDFMap::Ptr getSDFMapPtr() { return sdf_map_ptr_; }
        
        // [NEW] 设置SDF地图共享指针
        void setSDFMapPtr(SDFMap::Ptr sdf_map_ptr) { sdf_map_ptr_ = sdf_map_ptr; }

        // SDF Query Functions
        double getDistWithGradInBox(const Eigen::VectorXd& arm_angles, 
                                const Eigen::Vector3d& p_body, 
                                const int& box_id, 
                                Eigen::Vector3d& grad_body, 
                                Eigen::Vector3d& grad_box);
        double getDistWithGrad_body(const Eigen::VectorXd& arm_angles, const Eigen::Vector3d& pt, int& box_id, Eigen::Vector3d& grad);
        double getRoughDist_body(const Eigen::VectorXd& arm_angles, const Eigen::Vector3d& pt, int& box_id);
        double getRoughDistInFrameBox(const Eigen::Vector3d& pt, const int& box_id);  // Wrapper to sdf_map_ptr_
        double getDistance_box(const Eigen::Vector3i &id, const int& box_id);
        
        bool isInMap(const Eigen::Vector3d &pos);
        int getBoxNum() const { return box_num_; }
        int getDofNum() const { return box_num_-1; }
        Eigen::VectorXd get_thetas() const { return thetas_est_; }
        Eigen::VectorXd get_d_thetas() const { return dthetas_est_; }

        void getCurEndPose(const Eigen::Vector3d &drone_pos,
                            const Eigen::Quaterniond &drone_q,
                            Eigen::Vector3d &end_pos,
                            Eigen::Quaterniond &end_q);

        // Gradients
        Eigen::VectorXd get_grad_thetas_sdf(const Eigen::VectorXd& thetas, const Eigen::Vector3d& p_body, 
                                        const Eigen::Vector3d& gd_box, const int& box_id,
                                        const std::vector<Eigen::Matrix4d>& transforms);

        Eigen::VectorXd get_grad_thetas_wp(const Eigen::Vector3d& pos, 
                                            const Eigen::Quaterniond& q,
                                            const Eigen::VectorXd& thetas, 
                                            const Eigen::Matrix4d& T_e2b, 
                                            const Eigen::Vector3d& d_dis_d_p_e,
                                            const std::vector<Eigen::Matrix4d>& transforms);
        void get_grad_dis_wp_full(
            const Eigen::Vector3d &pos, const Eigen::Quaterniond &q,
            const Eigen::VectorXd &thetas, const Eigen::Vector3d &d_dis_d_p_e,
            Eigen::Vector3d &d_p, Eigen::Vector4d &d_q, Eigen::VectorXd &d_theta);

        //! pub
        void end_close_pub(){std_msgs::Bool msg; msg.data = true; end_mode_pub_.publish(msg);}
        void end_open_pub(){std_msgs::Bool msg; msg.data = false; end_mode_pub_.publish(msg);}

        //! vis
        void getRobotMarkerArray(const Eigen::Vector3d& pos_cur,
                                const Eigen::Quaterniond& quat_cur, 
                                const Eigen::VectorXd& arm_angles, 
                                visualization_msgs::MarkerArray& marker_array,
                                const visualization_rc_sdf::Color& color_mode = visualization_rc_sdf::Color::colorful,
                                const double& alpha = 0.4);

        void getRobotMarkerArray(const Eigen::VectorXd& state, 
                                visualization_msgs::MarkerArray& marker_array,
                                const visualization_rc_sdf::Color& color_mode = visualization_rc_sdf::Color::colorful,
                                const double& alpha = 0.4);

        void robotMarkersPub(const visualization_msgs::MarkerArray& marker_array,const std::string& topic);

        void visRobotSeq(const std::vector<Eigen::VectorXd>& pathXd, 
                        const std::string& topic,
                        const visualization_rc_sdf::Color& color_mode = visualization_rc_sdf::Color::colorful);

        bool isInRob(const Eigen::VectorXd &arm_angles, const Eigen::Vector3d& pt_body);

    private:


        void joint_state_est_cb(const sensor_msgs::JointState::ConstPtr& msg);

        //! vis
        void vis_box_sdf_box(const int& box_id); 
        void vis_box_sdf_body(const int& box_id);
        void vis_map_sdf();
        void callback_slice_coord(const geometry_msgs::PointStamped::ConstPtr& msg);
        void odom_callback(const nav_msgs::Odometry::ConstPtr& msg);
        void getBoxPclBoxFrame(const int& box_id, std::vector<std::pair<Eigen::Vector3d,double>>& pcl_i);
        void visBoxesPclWorldFrame(const Eigen::Vector3d& pos_cur, const Eigen::Quaterniond& quat_cur);
        void colorModeTo3D(Eigen::Vector3d& color, const visualization_rc_sdf::Color& color_mode);

        // 可视化手操状态
        void uam_state_vis_callback(const quadrotor_msgs::UAMFullState::ConstPtr& msg);

        //! util
        void indexToPos(const Eigen::Vector3i &id, const Eigen::Vector3d& map_origin, Eigen::Vector3d &pos);
        void posToIndex(const Eigen::Vector3d &pos, const Eigen::Vector3d& map_origin, Eigen::Vector3i &id);
        void indexToPos(const Eigen::Vector3i &id, const int& box_id, Eigen::Vector3d &pos);
        void posToIndex(const Eigen::Vector3d &pos, const int& box_id, Eigen::Vector3i &id);
        int toAddress(const int &x, const int &y, const int &z, const Eigen::Vector3i &map_voxel_num);
        int toAddress(const Eigen::Vector3i &id, const Eigen::Vector3i &map_voxel_num);
        int toAddress(const Eigen::Vector3i &id, const int& box_id);
        bool isInBox(const Eigen::Vector3d &pos, const int& box_id);
        bool isInBox(const Eigen::Vector3i &id, const int& box_id);

        bool getBasePoseCallback(quadrotor_msgs::GetBasePose::Request &req,
                                 quadrotor_msgs::GetBasePose::Response &res);

    private:
        ros::NodeHandle nh_;
        int box_num_;
        double resolution_, resolution_inv_;
        double default_dist_;
        std::vector<FieldData> box_data_list_;
        Eigen::Vector3d col_check_map_size_;

        ros::ServiceServer body_pose_srv_;

        //! vis
        bool vis_en_ = true;
        Eigen::Vector3d vis_map_sdf_silce_coord_ = Eigen::Vector3d::Zero();
        ros::Subscriber sub_slice_coord_;
        std::shared_ptr<visualization_rc_sdf::Visualization> vis_ptr_;
        PublisherMap publisher_map_;
        ros::Time last_vis_time_;
        
        ros::Publisher end_mode_pub_;
        ros::Subscriber sub_odom, sub_joint_state_est, sub_uam_state_vis_;
        ros::Subscriber sub_arm_pitch_ang_, sub_arm_pitch2_ang_, sub_arm_roll_ang_;
        
        Eigen::VectorXd thetas_est_, dthetas_est_;
        bool have_arm_angles_cur_ = false;

    public:
        // [NEW] Kinematics Pointer
        RobotKinematics::Ptr kine_ptr_;
        
        // [NEW] SDF Map Pointer
        SDFMap::Ptr sdf_map_ptr_;

        typedef std::shared_ptr<CH_RC_SDF> Ptr;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    struct FieldData {
        // 依然保留这些数据以便于 CH_RC_SDF 初始化，
        // 但实际计算时将使用 kine_ptr_ 中的数据
        Eigen::Vector3d T_cur2last;
        int ang_id;

        Eigen::Vector3d field_size;
        Eigen::Vector3d field_min_boundary, field_max_boundary, field_origin;
        Eigen::Vector3i field_voxel_num;
        std::vector<double> distance_buffer;
        double shell_thickness;

        std::vector<DiskParams> disk_params_list;
        std::vector<LineParams> line_params_list;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    // Note: DiskParams and LineParams are now defined in sdf_map.hpp

    // Inline Implementations for Indexing (Kept here as they relate to FieldData)
    inline void CH_RC_SDF::indexToPos(const Eigen::Vector3i &id, const Eigen::Vector3d& map_origin, Eigen::Vector3d &pos){
        for (size_t i = 0; i < 3; ++i)
            pos(i) = (id(i) + 0.5) * resolution_ + map_origin(i);
    }

    inline void CH_RC_SDF::posToIndex(const Eigen::Vector3d &pos, const Eigen::Vector3d& map_origin, Eigen::Vector3i &id){
        for (size_t i = 0; i < 3; ++i)
            id(i) = floor((pos(i) - map_origin(i)) * resolution_inv_);
    }

    inline void CH_RC_SDF::indexToPos(const Eigen::Vector3i &id, const int& box_id, Eigen::Vector3d &pos){
        indexToPos(id, box_data_list_[box_id].field_origin, pos);
    }

    inline void CH_RC_SDF::posToIndex(const Eigen::Vector3d &pos, const int& box_id, Eigen::Vector3i &id){
        posToIndex(pos, box_data_list_[box_id].field_origin, id);
    }

    inline bool CH_RC_SDF::isInBox(const Eigen::Vector3d &pos, const int& box_id) {
        Eigen::Vector3d min_boundary = box_data_list_[box_id].field_min_boundary;
        Eigen::Vector3d max_boundary = box_data_list_[box_id].field_max_boundary;

        if (pos(0) < min_boundary(0) + 1e-4 || pos(1) < min_boundary(1) + 1e-4 ||
            pos(2) < min_boundary(2) + 1e-4)
            return false;
        if (pos(0) > max_boundary(0) - 1e-4 || pos(1) > max_boundary(1) - 1e-4 ||
            pos(2) > max_boundary(2) - 1e-4)
            return false;
        return true;
    }

    inline bool CH_RC_SDF::isInBox(const Eigen::Vector3i &idx, const int& box_id) {
        Eigen::Vector3i map_voxel_num = box_data_list_[box_id].field_voxel_num;

        if (idx(0) < 0 || idx(1) < 0 || idx(2) < 0) return false;
        if (idx(0) > map_voxel_num(0) - 1 || idx(1) > map_voxel_num(1) - 1 ||
            idx(2) > map_voxel_num(2) - 1)
            return false;
        return true;
    }

    inline int CH_RC_SDF::toAddress(const int &x, const int &y, const int &z, const Eigen::Vector3i &map_voxel_num) {
        return x * map_voxel_num(1) * map_voxel_num(2) + y * map_voxel_num(2) + z;
    }

    inline int CH_RC_SDF::toAddress(const Eigen::Vector3i &id, const Eigen::Vector3i &map_voxel_num) {
        return toAddress(id[0], id[1], id[2], map_voxel_num);
    }

    inline int CH_RC_SDF::toAddress(const Eigen::Vector3i &id, const int& box_id) {
        Eigen::Vector3i map_voxel_num = box_data_list_[box_id].field_voxel_num;
        return toAddress(id[0], id[1], id[2], map_voxel_num);
    }

    inline double CH_RC_SDF::getDistance_box(const Eigen::Vector3i &id, const int& box_id) {
        if (!isInBox(id, box_id)) return default_dist_;
        return box_data_list_[box_id].distance_buffer[toAddress(id,box_id)];
    }

    inline bool CH_RC_SDF::isInMap(const Eigen::Vector3d &pos) {
       if (pos(0) < -0.5*col_check_map_size_(0) || pos(0) > 0.5*col_check_map_size_(0) ||
           pos(1) < -0.5*col_check_map_size_(1) || pos(1) > 0.5*col_check_map_size_(1) ||
           pos(2) < -0.5*col_check_map_size_(2) || pos(2) > 0.5*col_check_map_size_(2))
            return false;
        return true;
    }

}
#endif