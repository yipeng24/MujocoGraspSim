#ifndef _ROBOT_KINEMATICS_HPP_
#define _ROBOT_KINEMATICS_HPP_

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include <memory>
#include <cmath>
#include <ros/ros.h>

namespace clutter_hand {

struct KinematicLink {
    Eigen::Vector3d T_cur2last; // Translation relative to parent
    int ang_id;                 // Rotation axis: 0 for X, 1 for Y, 2 for Z
};

class RobotKinematics {
public:
    typedef std::shared_ptr<RobotKinematics> Ptr;

    /**
     * @brief Load kinematic link parameters from ROS parameter server
     * @param nh ROS NodeHandle
     * @param box_num Number of boxes/links in the kinematic chain
     * @return Vector of KinematicLink structures
     */
    static std::vector<KinematicLink> loadKinematicLinksFromROS(ros::NodeHandle& nh, int box_num) {
        std::vector<KinematicLink> links;
        links.reserve(box_num);
        
        for (int i = 0; i < box_num; i++) {
            KinematicLink link;
            
            // Read translation parameters
            Eigen::Vector3d T_2_last;
            nh.param("ch_rc_sdf/box" + std::to_string(i) + "/T_2_last/x", T_2_last(0), 0.0);
            nh.param("ch_rc_sdf/box" + std::to_string(i) + "/T_2_last/y", T_2_last(1), 0.0);
            nh.param("ch_rc_sdf/box" + std::to_string(i) + "/T_2_last/z", T_2_last(2), 0.0);
            link.T_cur2last = T_2_last;
            
            // Read rotation axis ID
            nh.param("ch_rc_sdf/box" + std::to_string(i) + "/Ang_2_last/ang_id", link.ang_id, 0);
            
            std::cout << "[RobotKinematics]: box_" << i 
                      << " T_2_last: " << link.T_cur2last.transpose() << std::endl;
            std::cout << "[RobotKinematics]: box_" << i 
                      << " ang_id: " << link.ang_id << std::endl;
            
            links.push_back(link);
        }
        
        return links;
    }

    RobotKinematics() = default;
    ~RobotKinematics() = default;

    // 初始化参数
    void setKinematicParams(const std::vector<KinematicLink>& links) {
        links_ = links;
        box_num_ = links.size();
    }

    int getBoxNum() const { return box_num_; }
    const std::vector<KinematicLink>& getLinks() const { return links_; }

    // ==========================================
    // 基础变换逻辑
    // ==========================================
    // 稳定的变换矩阵求逆函数
    inline Eigen::Matrix4d inverseTransform(const Eigen::Matrix4d& T) const {
        Eigen::Matrix4d inv = Eigen::Matrix4d::Identity();
        Eigen::Matrix3d RT = T.block<3,3>(0,0).transpose(); // 旋转部分的逆就是转置
        inv.block<3,3>(0,0) = RT;
        inv.block<3,1>(0,3) = -RT * T.block<3,1>(0,3); // 新的平移部分
        return inv;
    }

    inline void getRelativeTransform(const Eigen::VectorXd& arm_angles, const int& i, const int& j, Eigen::Matrix4d& relative_transform) const {
        relative_transform.setIdentity();
        if (i == j) return;

        int small_id = std::min(i, j);
        int large_id = std::max(i, j);

        for (int k = large_id; k > small_id; --k) {
            Eigen::Matrix4d transform_k_last;
            double angle = (k - 1 >= 0 && k - 1 < arm_angles.size()) ? arm_angles(k-1) : 0.0;
            getBoxTransform2last(angle, k, transform_k_last);
            relative_transform = transform_k_last * relative_transform;
        }

        if (small_id != j) {
            relative_transform = inverseTransform(relative_transform); // 使用稳定的求逆
        }
    }

    inline void getBoxTransform2last(const double& angle, const int& box_id, Eigen::Matrix4d& transform_cur2last) const {
        transform_cur2last.setIdentity();
        if (box_id < 0 || box_id >= links_.size()) return;

        // 确保索引安全
        int axis = std::max(0, std::min(2, links_[box_id].ang_id));
        Eigen::Vector3d ea = Eigen::Vector3d::Zero();
        ea(axis) = angle;

        // 使用 AngleAxis 乘法产生的旋转矩阵在数学上是完美的
        Eigen::Matrix3d R = Eigen::AngleAxisd(ea(0), Eigen::Vector3d::UnitX()) 
                        * Eigen::AngleAxisd(ea(1), Eigen::Vector3d::UnitY()) 
                        * Eigen::AngleAxisd(ea(2), Eigen::Vector3d::UnitZ()).toRotationMatrix();;
        
        transform_cur2last.block<3,3>(0,0) = R;
        transform_cur2last.block<3,1>(0,3) = links_[box_id].T_cur2last;
    }


    // T_{box_id}^{box_id-1} * p_i_homo
    inline void convertPosToFrame(const Eigen::VectorXd& arm_angles, const Eigen::Vector3d &p_i, const int& i, const int& j, Eigen::Vector3d &p_j) const {
        Eigen::Matrix4d relative_transform;
        getRelativeTransform(arm_angles, i, j, relative_transform);
        p_j = relative_transform.block<3,3>(0,0)*p_i + relative_transform.block<3,1>(0,3);
    }

    // ==========================================
    // 正运动学 / 逆运动学 (FK / IK)
    // ==========================================

    void getEndPose(const Eigen::Vector3d& drone_pos,
                    const Eigen::Quaterniond& drone_q,
                    const Eigen::VectorXd& arm_angles,
                    Eigen::Vector3d &end_pos,
                    Eigen::Quaterniond &end_q,
                    Eigen::Matrix4d& T_e2b) const {
        // 默认最后一个box为末端
        getRelativeTransform(arm_angles, box_num_-1, 0, T_e2b);

        Eigen::Matrix4d T_drone2w = Eigen::Matrix4d::Identity();
        T_drone2w.block<3,3>(0,0) = drone_q.toRotationMatrix();
        T_drone2w.block<3,1>(0,3) = drone_pos;
        Eigen::Matrix4d T_end2w = T_drone2w * T_e2b;

        end_pos = T_end2w.block<3,1>(0,3);
        end_q = Eigen::Quaterniond(T_end2w.block<3,3>(0,0));
    }

    void getCurEndPose(const Eigen::Vector3d &drone_pos,
                       const Eigen::Quaterniond &drone_q,
                       const Eigen::VectorXd &thetas_cur,
                       Eigen::Vector3d &end_pos,
                       Eigen::Quaterniond &end_q) const {
        Eigen::Matrix4d T_end2drone;
        getRelativeTransform(thetas_cur, box_num_-1, 0, T_end2drone);

        Eigen::Matrix4d T_drone2w = Eigen::Matrix4d::Identity();
        T_drone2w.block<3,3>(0,0) = drone_q.toRotationMatrix();
        T_drone2w.block<3,1>(0,3) = drone_pos;
        Eigen::Matrix4d T_end2w = T_drone2w * T_end2drone;

        end_pos = T_end2w.block<3,1>(0,3);
        end_q = Eigen::Quaterniond(T_end2w.block<3,3>(0,0));
    }

    // p_end2w + R_end2w -> odom_drone_end + thetas_end
    // 注意：这里的 IK 逻辑是针对特定机械臂构型的
    void endPose2Thetas(const Eigen::Vector3d &p_end2w,
                        const Eigen::Matrix3d &R_end2w,
                        const int dof_num, // 传入自由度数量用于resize
                        Eigen::VectorXd &thetas_end,
                        Eigen::Vector3d &pos_drone_out,
                        double &yaw_out) const {
        
        if (links_.size() < 3) {
            std::cerr << "[RobotKinematics] Link size too small for endPose2Thetas" << std::endl;
            return;
        }

        // 旋转矩阵 --> 欧拉角(Z-Y-X，即YPR)
        Eigen::Vector3d ea;
        // x
        ea(2) = std::atan2(R_end2w(2, 1), R_end2w(2, 2));
        // y
        ea(1) = std::atan2(-R_end2w(2, 0), std::sqrt(R_end2w(2, 1) * R_end2w(2, 1) + R_end2w(2, 2) * R_end2w(2, 2)));
        // z
        ea(0) = std::atan2(R_end2w(1, 0), R_end2w(0, 0));
        double p = ea(1); // pitch

        // 要求重心尽量不偏移
        double d1/*joint0-joint1*/, d2/*joint1-obj*/, d3/*uav-obj*/;
        
        // get d1: 对应原代码 box_data_list_[2]
        d1 = links_[2].T_cur2last.norm();
        
        // get d2
        Eigen::Vector3d pos_obj_f_grab, pos_obj_f_j2;
        pos_obj_f_grab << 0.0, 0, 0; // 夹爪坐标系下物体的位置

        // 这里的 thetas_temp 是全0，用于计算静态变换
        Eigen::VectorXd thetas_temp(dof_num); 
        thetas_temp.setZero();
        
        Eigen::Matrix4d T_end2box2;
        getRelativeTransform(thetas_temp, box_num_-1, 2, T_end2box2);
        pos_obj_f_j2 = T_end2box2.block<3,3>(0,0) * pos_obj_f_grab + T_end2box2.block<3,1>(0,3);
        
        d2 = pos_obj_f_j2.norm();
        
        // get d3
        d3 = d2*sin(p) + sqrt(d1*d1 - d2*d2*cos(p)*cos(p));

        // links_[1] 对应原代码 box_data_list_[1]
        pos_drone_out = p_end2w + d3*Eigen::Vector3d::UnitZ() /*box12end*/ - links_[1].T_cur2last /*drone2box1*/;
        yaw_out = ea(0);

        thetas_end.resize(dof_num);
        thetas_end(0) = asin(d2*cos(p)/d1) + 0.5*M_PI;
        thetas_end(1) = - thetas_end(0) + p;
        thetas_end(2) = ea(2);
    }


    /**
     * @brief 计算末端位置相对于 [p_d, q_d(w,x,y,z), theta] 的雅可比矩阵
     * @return Eigen::MatrixXd 矩阵大小为 3 x (3 + 4 + arm_angles.size())
     */
    Eigen::MatrixXd getEndPosJacobianFull(const Eigen::Vector3d& drone_pos,
                                        const Eigen::Quaterniond& drone_q,
                                        const Eigen::VectorXd& arm_angles) const {
        const int n = arm_angles.size();
        // 状态量：3 (pos) + 4 (quat: w,x,y,z) + n (angles)
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 7 + n);

        // 1. 获取机械臂末端相对于基座的偏移 t_e2b
        Eigen::Matrix4d T_e2b;
        getRelativeTransform(arm_angles, box_num_ - 1, 0, T_e2b);
        Eigen::Vector3d v = T_e2b.block<3, 1>(0, 3); // 即 t_e2b

        // 四元数分量 (注意 Eigen 内部存储顺序是 x,y,z,w，但这里显式读取)
        double w = drone_q.w();
        double x = drone_q.x();
        double y = drone_q.y();
        double z = drone_q.z();

        // ======================================================
        // A. 对 drone_pos 的导数 (3x3)
        // ======================================================
        J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

        // ======================================================
        // B. 对 drone_q (w, x, y, z) 的导数 (3x4)
        // ======================================================
        // Column w
        J.col(3) = 2.0 * Eigen::Vector3d(w*v.x() + y*v.z() - z*v.y(),
                                        w*v.y() + z*v.x() - x*v.z(),
                                        w*v.z() + x*v.y() - y*v.x());
        // Column x
        J.col(4) = 2.0 * Eigen::Vector3d(x*v.x() + y*v.y() + z*v.z(),
                                        y*v.x() - x*v.y() - w*v.z(),
                                        z*v.x() - x*v.z() + w*v.y());
        // Column y
        J.col(5) = 2.0 * Eigen::Vector3d(x*v.y() - y*v.x() + w*v.z(),
                                        x*v.x() + y*v.y() + z*v.z(),
                                        z*v.y() - y*v.z() - w*v.x());
        // Column z
        J.col(6) = 2.0 * Eigen::Vector3d(x*v.z() - z*v.x() - w*v.y(),
                                        y*v.z() - z*v.y() + w*v.x(),
                                        x*v.x() + y*v.y() + z*v.z());

        // ======================================================
        // C. 对 arm_angles 的导数 (3xN)
        // ======================================================
        Eigen::Matrix3d R_d = drone_q.toRotationMatrix();
        for (int i = 1; i < box_num_; ++i) {
            if (i - 1 >= n) break;

            Eigen::Matrix4d T_i_0, T_prev_0;
            getRelativeTransform(arm_angles, i, 0, T_i_0);
            getRelativeTransform(arm_angles, i-1, 0, T_prev_0);
            
            Eigen::Vector3d p_i_b = T_i_0.block<3, 1>(0, 3);
            Eigen::Vector3d axis_local = Eigen::Vector3d::Zero();
            axis_local(links_[i].ang_id) = 1.0;
            
            // 旋转轴在基座系下的表达
            Eigen::Vector3d z_i_b = T_prev_0.block<3, 3>(0, 0) * axis_local;

            // 基座系下的关节雅可比，然后投影到世界系
            Eigen::Vector3d Jp_i_b = z_i_b.cross(v - p_i_b);
            J.block<3, 1>(0, 7 + i - 1) = R_d * Jp_i_b;
        }

        return J;
    }

private:
    std::vector<KinematicLink> links_;
    int box_num_ = 0;
};

} // namespace clutter_hand

#endif // _ROBOT_KINEMATICS_HPP_