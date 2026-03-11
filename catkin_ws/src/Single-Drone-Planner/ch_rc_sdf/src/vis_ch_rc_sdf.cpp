#include "ch_rc_sdf/ch_rc_sdf.h"

#include <memory>

namespace clutter_hand {

    // goal vis
    void CH_RC_SDF::uam_state_vis_callback(const quadrotor_msgs::UAMFullState::ConstPtr& msg)
    {
        Eigen::Vector3d pos_cur;
        Eigen::Quaterniond q_b;
        pos_cur << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
        q_b.w() = msg->pose.orientation.w;
        q_b.x() = msg->pose.orientation.x;
        q_b.y() = msg->pose.orientation.y;
        q_b.z() = msg->pose.orientation.z;

        Eigen::VectorXd thetas(thetas_est_.size());
        thetas << msg->theta[0], msg->theta[1], msg->theta[2];

        //! vis marker
        if(!vis_en_) return;
        visualization_msgs::MarkerArray marker_array;
        getRobotMarkerArray(pos_cur, q_b, thetas, marker_array);
        robotMarkersPub(marker_array,"uam_state_vis_marker");
    }

    //! vis robot
    void CH_RC_SDF::odom_callback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        double dt = ros::Time::now().toSec() - last_vis_time_.toSec();
        if(dt < 0.05)
            return;

        Eigen::Vector3d pos_cur;
        Eigen::Quaterniond quat_cur;
        pos_cur << msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z;
        quat_cur.w() = msg->pose.pose.orientation.w;
        quat_cur.x() = msg->pose.pose.orientation.x;
        quat_cur.y() = msg->pose.pose.orientation.y;
        quat_cur.z() = msg->pose.pose.orientation.z;

        //! vis marker
        if(!vis_en_) return;
        visualization_msgs::MarkerArray marker_array;
        getRobotMarkerArray(pos_cur, quat_cur, thetas_est_, marker_array);
        robotMarkersPub(marker_array,"robot");

        //! vis sdf
        bool vis_sdf = true;
        if(vis_sdf){
            visBoxesPclWorldFrame(pos_cur, quat_cur);

            // 动态查看sdf
            last_vis_time_ = ros::Time::now();
        }
    }

    void CH_RC_SDF::visBoxesPclWorldFrame(const Eigen::Vector3d& pos_cur, 
                                        const Eigen::Quaterniond& quat_cur)
    {
        for(size_t i = 0; i < box_num_; ++i){
            std::vector<std::pair<Eigen::Vector3d,double>> pcl_i;
            getBoxPclBoxFrame(i, pcl_i);

            for(size_t j = 0; j < pcl_i.size(); ++j){
                Eigen::Vector3d pt_f0, pt_fw;
                kine_ptr_->convertPosToFrame(thetas_est_, pcl_i[j].first, i, 0, pt_f0);
                pcl_i[j].first = quat_cur * pt_f0 + pos_cur;
            }

            vis_ptr_->visualize_pointcloud_itensity(pcl_i, "sdf_box"+std::to_string(i)+"_world_frame");
        }
    }

    void CH_RC_SDF::getBoxPclBoxFrame(const int& box_id,
                            std::vector<std::pair<Eigen::Vector3d,double>>& pcl_i)
    {
        pcl_i.clear();
        FieldData& box_data = box_data_list_[box_id];
        Eigen::Vector3i slice_idx;
        posToIndex(0.5*box_data.field_size+box_data.field_origin, 
                    box_data.field_origin, slice_idx);

        for (int i = 0; i < box_data.field_voxel_num(0); ++i) 
        for (int j = 0; j < box_data.field_voxel_num(1); ++j) 
        for (int k = 0; k < box_data.field_voxel_num(2); ++k) {
                Eigen::Vector3d pt;
                indexToPos(Eigen::Vector3i(i, j, k), box_data.field_origin, pt);
                double dist = box_data.distance_buffer[toAddress(Eigen::Vector3i(i, j, k), box_data.field_voxel_num)];

                if(k==slice_idx(2) && dist < 0.08)
                // if((i==slice_idx(0) || j==slice_idx(1) || k==slice_idx(2)))
                // if((i==slice_idx(0) || j==slice_idx(1) || k==slice_idx(2))&& dist < 0.2)
                    pcl_i.push_back(std::make_pair(pt,dist));
            }
    }

    void CH_RC_SDF::getRobotMarkerArray(const Eigen::VectorXd& state, 
                                        visualization_msgs::MarkerArray& marker_array,
                                        const visualization_rc_sdf::Color& color_mode,
                                        const double& alpha)
    {
        double yaw = state(3);
        Eigen::Quaterniond quat_cur = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
        Eigen::Vector3d pos_cur = state.head(3);
        Eigen::VectorXd arm_angles = state.tail(state.size()-4);
        getRobotMarkerArray(pos_cur, quat_cur, arm_angles, marker_array, color_mode);
    }

    void CH_RC_SDF::getRobotMarkerArray(const Eigen::Vector3d& pos_cur,
                                    const Eigen::Quaterniond& quat_cur, 
                                    const Eigen::VectorXd& arm_angles, 
                                    visualization_msgs::MarkerArray& marker_array,
                                    const visualization_rc_sdf::Color& color_mode,
                                    const double& alpha)
    {
        ros::Time stamp = ros::Time::now();
        std::string frame_id = "world";
        Eigen::Matrix4d transform_02w;
        transform_02w.block<3,3>(0,0) = quat_cur.toRotationMatrix();
        transform_02w.block<3,1>(0,3) = pos_cur;

        for(size_t i = 0; i < box_num_; ++i){
            Eigen::Matrix4d transform_cur20;
            kine_ptr_->getRelativeTransform(arm_angles, i, 0, transform_cur20);
            double shell_thickness = box_data_list_[i].shell_thickness;

            Eigen::Vector3d color;
            color.setZero();
            if(color_mode == visualization_rc_sdf::Color::colorful)
                color(i%3) = 1.0;
            else
                colorModeTo3D(color,color_mode);

            for(size_t j = 0; j < box_data_list_[i].disk_params_list.size(); ++j){
                DiskParams& marker_params = box_data_list_[i].disk_params_list[j];

                visualization_msgs::Marker marker;
                marker.header.frame_id = frame_id;
                marker.header.stamp = stamp;
                marker.ns = "shapes";
                marker.id = marker_array.markers.size();
                marker.type = visualization_msgs::Marker::CYLINDER;
                marker.action = visualization_msgs::Marker::ADD;

                // transform marker from cur to 0
                Eigen::Matrix4d transform_marker2w, transform_marker2cur;
                transform_marker2cur.setIdentity();
                transform_marker2cur.block<3,3>(0,0) = marker_params.R;
                transform_marker2cur.block<3,1>(0,3) = marker_params.T;
                transform_marker2w = transform_02w*transform_cur20*transform_marker2cur;

                Eigen::Quaterniond quat_marker2w = Eigen::Quaterniond(transform_marker2w.block<3,3>(0,0));

                marker.pose.position.x = transform_marker2w.block<3,1>(0,3)(0);
                marker.pose.position.y = transform_marker2w.block<3,1>(0,3)(1);
                marker.pose.position.z = transform_marker2w.block<3,1>(0,3)(2);
                marker.pose.orientation.w = quat_marker2w.w();
                marker.pose.orientation.x = quat_marker2w.x();
                marker.pose.orientation.y = quat_marker2w.y();
                marker.pose.orientation.z = quat_marker2w.z();

                double radius = marker_params.radius;
                marker.scale.x = 2*(radius+shell_thickness);
                marker.scale.y = 2*(radius+shell_thickness);
                marker.scale.z = 2*shell_thickness;

                marker.color.r = color(0);
                marker.color.g = color(1);
                marker.color.b = color(2);
                marker.color.a = alpha;
                marker_array.markers.push_back(marker);
            }

            for(size_t j = 0; j < box_data_list_[i].line_params_list.size(); ++j){
                LineParams& marker_params = box_data_list_[i].line_params_list[j];

                visualization_msgs::Marker marker;
                marker.header.frame_id = frame_id;
                marker.header.stamp = stamp;
                marker.ns = "shapes";
                marker.id = marker_array.markers.size();
                marker.type = visualization_msgs::Marker::CYLINDER;
                marker.action = visualization_msgs::Marker::ADD;

                double length = marker_params.length;

                Eigen::Vector3d pt_in, pt_out;
                pt_in << marker_params.T;
                pt_out << marker_params.T + marker_params.R*Eigen::Vector3d::UnitX()*length;
                Eigen::Vector3d center = (pt_in + pt_out) / 2.0;
                Eigen::Vector3d dir = pt_out - pt_in;
                Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
                Eigen::Quaterniond quat = Eigen::Quaterniond::FromTwoVectors(z_axis, dir);

                // transform marker from cur to 0
                Eigen::Matrix4d transform_marker2w, transform_marker2cur;
                transform_marker2cur.setIdentity();
                transform_marker2cur.block<3,3>(0,0) = quat.toRotationMatrix();
                transform_marker2cur.block<3,1>(0,3) = center;
                transform_marker2w = transform_02w*transform_cur20*transform_marker2cur;
                Eigen::Quaterniond quat_marker2w = Eigen::Quaterniond(transform_marker2w.block<3,3>(0,0));

                marker.pose.position.x = transform_marker2w.block<3,1>(0,3)(0);
                marker.pose.position.y = transform_marker2w.block<3,1>(0,3)(1);
                marker.pose.position.z = transform_marker2w.block<3,1>(0,3)(2);
                marker.pose.orientation.w = quat_marker2w.w();
                marker.pose.orientation.x = quat_marker2w.x();
                marker.pose.orientation.y = quat_marker2w.y();
                marker.pose.orientation.z = quat_marker2w.z();

                // marker.pose.position.x = 0;
                // marker.pose.position.y = 0;
                // marker.pose.position.z = 0;
                // marker.pose.orientation.w = 1;
                // marker.pose.orientation.x = 0;
                // marker.pose.orientation.y = 0;
                // marker.pose.orientation.z = 0;

                marker.scale.x = 2*shell_thickness+0.01;
                marker.scale.y = 2*shell_thickness+0.01;
                marker.scale.z = length;
                marker.color.r = color(0);
                marker.color.g = color(1);
                marker.color.b = color(2);
                marker.color.a = 0.4;
                marker_array.markers.push_back(marker);
            }
        }
        
    }

    void CH_RC_SDF::robotMarkersPub(const visualization_msgs::MarkerArray& marker_array,const std::string& topic)
    {
        auto got = publisher_map_.find(topic);
        if (got == publisher_map_.end()) {
            ros::Publisher pub = nh_.advertise<visualization_msgs::MarkerArray>(topic, 10);
            publisher_map_[topic] = pub;
        }

        publisher_map_[topic].publish(marker_array);
    }

    void CH_RC_SDF::visRobotSeq(const std::vector<Eigen::VectorXd>& pathXd, 
                                const std::string& topic,
                                const visualization_rc_sdf::Color& color_mode)
    {
        if(pathXd.size() == 0) return;
        // std::cout << "visRobotSeq" << std::endl;
        // std::cout << "pathXd[0].size(): " << pathXd[0].size() << std::endl;
        ros::Time stamp = ros::Time::now();
        std::string frame_id = "world";
        visualization_msgs::MarkerArray marker_array;
        bool is_3D = pathXd[0].size() == 3;

        visualization_msgs::Marker marker_delete;
        marker_delete.header.frame_id = frame_id;
        marker_delete.header.stamp = stamp;
        marker_delete.ns = "shapes";
        marker_delete.id = marker_array.markers.size();
        marker_delete.type = visualization_msgs::Marker::CYLINDER;
        marker_delete.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(marker_delete);

        for (const auto& ptXd : pathXd) {
            
            Eigen::VectorXd thetas;
            thetas = is_3D ? Eigen::VectorXd::Zero(box_num_-1) : Eigen::VectorXd(ptXd.tail(box_num_-1));
            double yaw = is_3D ? 0 : ptXd(3);
            Eigen::Quaterniond quat_cur(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix());
            Eigen::Vector3d pos_cur = Eigen::Map<const Eigen::Vector3d>(ptXd.data());

            visualization_msgs::MarkerArray marker_array_tmp;
            getRobotMarkerArray(pos_cur, quat_cur, thetas, marker_array_tmp, color_mode);

            for (auto& marker : marker_array_tmp.markers) {
                marker.id = marker_array.markers.size();
                marker_array.markers.push_back(marker);
            }
        }

        robotMarkersPub(marker_array, topic);
    }

//! vis SDF
    void CH_RC_SDF::callback_slice_coord(const geometry_msgs::PointStamped::ConstPtr& msg)
    {
        vis_map_sdf_silce_coord_ << msg->point.x, msg->point.y, msg->point.z;
    }


    void CH_RC_SDF::vis_box_sdf_box(const int& box_id){
        FieldData& box_data = box_data_list_[box_id];
        std::vector<std::pair<Eigen::Vector3d,double>> pcl_i;
        for (int i = 0; i < box_data.field_voxel_num(0); ++i) 
            for (int j = 0; j < box_data.field_voxel_num(1); ++j) 
                for (int k = 0; k < box_data.field_voxel_num(2); ++k) {

                    Eigen::Vector3d pt;
                    indexToPos(Eigen::Vector3i(i, j, k), box_data.field_origin, pt);
                    double dist = box_data.distance_buffer[toAddress(Eigen::Vector3i(i, j, k), box_data.field_voxel_num)];

                // if(i == round(0.5*box_data.field_voxel_num(0))
                //     || j == round(0.5*box_data.field_voxel_num(1))
                //     || k == round(0.5*box_data.field_voxel_num(2)))
                //     {
                if( fabs(pt.x()) < 0.5*resolution_ + 1e-3 
                    || fabs(pt.y()) < 0.5*resolution_ + 1e-3 
                    || fabs(pt.z()) < 0.5*resolution_ + 1e-3 )
                    {
                        pcl_i.push_back(std::make_pair(pt,dist));
                    }
                }
        vis_ptr_->visualize_pointcloud_itensity(pcl_i, "box" + std::to_string(box_id) + "_sdf_box");
    }

    void CH_RC_SDF::vis_box_sdf_body(const int& box_id){
        Eigen::Matrix4d transform_cur20;
        kine_ptr_->getRelativeTransform(thetas_est_, box_id, 0, transform_cur20);

        FieldData& box_data = box_data_list_[box_id];
        std::vector<std::pair<Eigen::Vector3d,double>> pcl_i;

        for (int i = 0; i < box_data.field_voxel_num(0); ++i) 
            for (int j = 0; j < box_data.field_voxel_num(1); ++j) 
                for (int k = 0; k < box_data.field_voxel_num(2); ++k) {
                    Eigen::Vector3d pt;
                    indexToPos(Eigen::Vector3i(i, j, k), box_data.field_origin, pt);
                    double dist = box_data.distance_buffer[toAddress(Eigen::Vector3i(i, j, k), box_data.field_voxel_num)];

                    bool in_plane  = (fabs(pt.x()) < 0.5*resolution_ + 1e-3)
                                    || (fabs(pt.y()) < 0.5*resolution_ + 1e-3)
                                    || (fabs(pt.z()) < 0.5*resolution_ + 1e-3);

                    Eigen::Vector3d pt_frame0 = transform_cur20.block<3,3>(0,0)*pt + 
                                                transform_cur20.block<3,1>(0,3);

                    // if(i == round(0.5*box_data.field_voxel_num(0))
                    // || j == round(0.5*box_data.field_voxel_num(1))
                    // || k == round(0.5*box_data.field_voxel_num(2)))

                    if (in_plane)
                        pcl_i.push_back(std::make_pair(pt_frame0,dist));
                }
        vis_ptr_->visualize_pointcloud_itensity(pcl_i, "box" + std::to_string(box_id) + "_sdf_body");
    }

    void CH_RC_SDF::vis_map_sdf()
    {

        std::cout  << "vis_map_sdf" << std::endl;

        double res = 0.02;
        double res_inv = 1.0 / res;

        Eigen::Vector3d map_size, map_origin;
        Eigen::Vector3i map_voxel_num;
        map_size << 4.0, 4.0, 3.0;
        for (int i = 0; i < 3; ++i)
            map_voxel_num(i) = ceil(map_size(i) * res_inv);

        map_origin << -map_size(0)/2.0, -map_size(1)/2.0, -map_size(2)/2.0;

        Eigen::Vector3i slice_idx;
        for (size_t i = 0; i < 3; ++i)
            slice_idx(i) = floor((vis_map_sdf_silce_coord_(i) - map_origin(i)) * res_inv);

        int cnt=0;
        int skip_num = 117;

        int slice_num = 10;
        std::vector<std::pair<Eigen::Vector3d,double>> pcl_i_x,pcl_i_y,pcl_i_z;
        std::vector<std::pair<Eigen::Vector3d,Eigen::Vector3d>> gd_x,gd_y,gd_z;
        for (int i = 0; i < map_voxel_num(0); ++i) 
            for (int j = 0; j < map_voxel_num(1); ++j) 
                for (int k = 0; k < map_voxel_num(2); ++k) {

                    if(j == slice_idx(1) || k == slice_idx(2))
                    // if(i == slice_idx(0) || j == slice_idx(1) || k == slice_idx(2))
                    {
                        Eigen::Vector3d pt;
                        Eigen::Vector3i id(i, j, k);

                        for (size_t l = 0; l < 3; ++l)
                            pt(l) = (id(l) + 0.5) * res + map_origin(l);

                        Eigen::Vector3d gd;
                        Eigen::Vector3d idx;
                        int box_id = 0;
                        double dist = getDistWithGrad_body(thetas_est_, pt, box_id, gd);

                        Eigen::Vector3d arr_end = pt + gd * res;
                        // std::cout << "gd.norm(): " << gd.norm() << std::endl;
                        bool vis_gd = cnt%skip_num==0 && gd.norm() > 1e-3;

                        // std::cout << "slice_idx: " << slice_idx.transpose() << std::endl;
                        // std::cout << "ijk:" << id.transpose() << std::endl;

                        if(i == slice_idx(0))
                        {
                            pcl_i_x.push_back(std::make_pair(pt,dist));

                            if(vis_gd)
                                gd_x.push_back(std::make_pair(pt,arr_end));
                        }
                        if(j == slice_idx(1))
                        {
                            pcl_i_y.push_back(std::make_pair(pt,dist));

                            if(vis_gd)
                                gd_y.push_back(std::make_pair(pt,arr_end));
                        }
                        if(k == slice_idx(2))
                        {
                            pcl_i_z.push_back(std::make_pair(pt,dist));

                            if(vis_gd)
                                gd_z.push_back(std::make_pair(pt,arr_end));
                        }
                        cnt++;
                    }

                }

        std::cout << "pcl_i_x size: " << pcl_i_x.size() << std::endl;
        std::cout << "pcl_i_y size: " << pcl_i_y.size() << std::endl;
        std::cout << "pcl_i_z size: " << pcl_i_z.size() << std::endl;
        vis_ptr_->visualize_pointcloud_itensity(pcl_i_x, "map_sdf_x");
        vis_ptr_->visualize_pointcloud_itensity(pcl_i_y, "map_sdf_y");
        vis_ptr_->visualize_pointcloud_itensity(pcl_i_z, "map_sdf_z");

        vis_ptr_->visualize_arrows(gd_x, "map_sdf_grad_x",visualization_rc_sdf::Color::red);
        vis_ptr_->visualize_arrows(gd_y, "map_sdf_grad_y",visualization_rc_sdf::Color::green);
        vis_ptr_->visualize_arrows(gd_z, "map_sdf_grad_z",visualization_rc_sdf::Color::blue);
    }


    void CH_RC_SDF::colorModeTo3D(Eigen::Vector3d& color, const visualization_rc_sdf::Color& color_mode){
        switch(color_mode){
            case visualization_rc_sdf::Color::red:
                color << 1,0,0;
                break;
            case visualization_rc_sdf::Color::green:
                color << 0,1,0;
                break;
            case visualization_rc_sdf::Color::blue:
                color << 0,0,1;
                break;
            case visualization_rc_sdf::Color::yellow:
                color << 1,1,0;
                break;
            case visualization_rc_sdf::Color::grey:
                color << 0.5,0.5,0.5;
                break;
            default:
                color << 1,1,1;
                break;
        }
    }
}  // namespace 

// [GOAL_VIS] thetas:      0 1.10715       0
// [GOAL_VIS]
// T_e2b:
//  0.447214         0  0.894427  0.204417
//         0         1         0   -0.0002
// -0.894427         0  0.447214 -0.197593
//         0         0         0         1
// T_drone2w:
//        1        0        0 0.771455
//        0        1        0   0.0002
//        0        0        1  1.22092
//        0        0        0        1
// T_end2w:
//  0.447214         0  0.894427  0.975872
//         0         1         0         0
// -0.894427         0  0.447214   1.02332
//         0         0         0         1

// [getBaseCB] thetas:      0 1.10715       0
// [getBaseCB] T_e2w
// :0.5   0   1   1
//   0   1   0   0
//  -1   0 0.5   1
//   0   0   0   1
// [getBaseCB] T_e2b:
//  0.447214         0  0.894427  0.204417
//         0         1         0   -0.0002
// -0.894427         0  0.447214 -0.197593
//         0         0         0         1
// [getBaseCB] T_b2w:
//  1.11803        0        0 0.771455
//        0        1        0   0.0002
//        0        0  1.11803  1.22092
//        0        0        0        1
