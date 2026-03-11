#ifndef _SDF_MAP_HPP_
#define _SDF_MAP_HPP_

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include <memory>
#include <cmath>
#include <ros/ros.h>

namespace clutter_hand {

// Geometric primitive: Disk
struct DiskParams {
    Eigen::Vector3d T;      // Translation
    Eigen::Matrix3d R;      // Rotation
    double radius;          // Disk radius
};

// Geometric primitive: Line segment
struct LineParams {
    Eigen::Vector3d T;      // Translation
    Eigen::Matrix3d R;      // Rotation
    double length;          // Line length
};

// SDF box data containing field information and distance buffer
struct SDFBoxData {
    Eigen::Vector3d field_size;
    Eigen::Vector3d field_min_boundary, field_max_boundary, field_origin;
    Eigen::Vector3i field_voxel_num;
    std::vector<double> distance_buffer;
    double shell_thickness;
    
    std::vector<DiskParams> disk_params_list;
    std::vector<LineParams> line_params_list;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class SDFMap {
public:
    typedef std::shared_ptr<SDFMap> Ptr;
    
    SDFMap() = default;
    ~SDFMap() = default;
    
    /**
     * @brief Load SDF box parameters and initialize all boxes
     * @param nh ROS NodeHandle
     * @param box_num Number of boxes
     * @param resolution Voxel resolution
     * @param default_dist Default distance value
     */
    void loadAndInitialize(ros::NodeHandle& nh, int box_num, double resolution, double default_dist) {
        box_num_ = box_num;
        resolution_ = resolution;
        default_dist_ = default_dist;
        resolution_inv_ = 1.0 / resolution;
        
        box_data_list_.clear();
        box_data_list_.reserve(box_num);
        
        for (int i = 0; i < box_num; i++) {
            SDFBoxData box_data;
            
            // Load box size
            Eigen::Vector3d box_size;
            nh.param("ch_rc_sdf/box" + std::to_string(i) + "/size_x", box_size(0), 0.0);
            nh.param("ch_rc_sdf/box" + std::to_string(i) + "/size_y", box_size(1), 0.0);
            nh.param("ch_rc_sdf/box" + std::to_string(i) + "/size_z", box_size(2), 0.0);
            std::cout << "[SDFMap]: box_" << i << " size: " << box_size.transpose() << std::endl;
            
            // Load origin position
            Eigen::Vector3d ori_pos;
            nh.param("ch_rc_sdf/box" + std::to_string(i) + "/ori_x", ori_pos(0), 0.0);
            nh.param("ch_rc_sdf/box" + std::to_string(i) + "/ori_y", ori_pos(1), 0.0);
            nh.param("ch_rc_sdf/box" + std::to_string(i) + "/ori_z", ori_pos(2), 0.0);
            std::cout << "[SDFMap]: box_" << i << " ori_pos: " << ori_pos.transpose() << std::endl;
            
            // Load shell thickness
            double shell_thickness;
            nh.param("ch_rc_sdf/box" + std::to_string(i) + "/shell_thickness", shell_thickness, 0.0);
            
            // Initialize field
            initializeBox(box_size, ori_pos, shell_thickness, box_data);
            
            // Load geometric primitives
            loadGeometricPrimitives(nh, i, box_data);
            
            // Compute SDF values
            feedBoxSDF(box_data, i);
            
            box_data_list_.push_back(box_data);
        }
        
        std::cout << "[SDFMap]: All " << box_num << " SDF boxes initialized." << std::endl;
    }
    
    /**
     * @brief Get box data by index
     */
    const SDFBoxData& getBoxData(int box_id) const {
        return box_data_list_[box_id];
    }
    
    SDFBoxData& getBoxData(int box_id) {
        return box_data_list_[box_id];
    }
    
    const std::vector<SDFBoxData>& getAllBoxes() const {
        return box_data_list_;
    }
    
    int getBoxNum() const { return box_num_; }
    double getResolution() const { return resolution_; }
    double getDefaultDist() const { return default_dist_; }
    
    /**
     * @brief Get distance value with gradient using trilinear interpolation
     * @param pt Point in box frame
     * @param box_id Box index
     * @param grad Output gradient vector (normalized)
     * @return Distance value
     */
    double getDistWithGradInFrameBox(const Eigen::Vector3d& pt, int box_id, Eigen::Vector3d& grad) const {
        if (!isInBox(pt, box_id)) {
            grad.setZero();
            return default_dist_;
        }
        
        const SDFBoxData& box_data = box_data_list_[box_id];
        
        /* trilinear interpolation */
        Eigen::Vector3d pos_m = pt - 0.5 * resolution_ * Eigen::Vector3d::Ones();
        Eigen::Vector3i idx;
        posToIndex(pos_m, box_id, idx);
        Eigen::Vector3d idx_pos, diff;
        indexToPos(idx, box_data.field_origin, idx_pos);
        diff = (pt - idx_pos) * resolution_inv_;
        
        double values[2][2][2];
        for (int x = 0; x < 2; x++)
            for (int y = 0; y < 2; y++)
                for (int z = 0; z < 2; z++) {
                    Eigen::Vector3i current_idx = idx + Eigen::Vector3i(x, y, z);
                    values[x][y][z] = getDistance(current_idx, box_id);
                }
        
        double v00 = (1 - diff[0]) * values[0][0][0] + diff[0] * values[1][0][0];
        double v01 = (1 - diff[0]) * values[0][0][1] + diff[0] * values[1][0][1];
        double v10 = (1 - diff[0]) * values[0][1][0] + diff[0] * values[1][1][0];
        double v11 = (1 - diff[0]) * values[0][1][1] + diff[0] * values[1][1][1];
        double v0 = (1 - diff[1]) * v00 + diff[1] * v10;
        double v1 = (1 - diff[1]) * v01 + diff[1] * v11;
        double dist = (1 - diff[2]) * v0 + diff[2] * v1;
        
        grad[2] = (v1 - v0) * resolution_inv_;
        grad[1] = ((1 - diff[2]) * (v10 - v00) + diff[2] * (v11 - v01)) * resolution_inv_;
        grad[0] = (1 - diff[2]) * (1 - diff[1]) * (values[1][0][0] - values[0][0][0]);
        grad[0] += (1 - diff[2]) * diff[1] * (values[1][1][0] - values[0][1][0]);
        grad[0] += diff[2] * (1 - diff[1]) * (values[1][0][1] - values[0][0][1]);
        grad[0] += diff[2] * diff[1] * (values[1][1][1] - values[0][1][1]);
        grad[0] *= resolution_inv_;
        
        grad.normalize();
        
        return dist;
    }
    
    /**
     * @brief Get rough distance (nearest voxel) in box frame
     * @param pt Point in box frame
     * @param box_id Box index
     * @return Distance value
     */
    double getRoughDistInFrameBox(const Eigen::Vector3d& pt, int box_id) const {
        if (!isInBox(pt, box_id))
            return default_dist_;
        
        Eigen::Vector3i idx;
        posToIndex(pt, box_id, idx);
        return getDistance(idx, box_id);
    }
    
    /**
     * @brief Get distance value at specific voxel index
     * @param idx Voxel index
     * @param box_id Box index
     * @return Distance value
     */
    double getDistance(const Eigen::Vector3i& idx, int box_id) const {
        if (!isInBox(idx, box_id))
            return default_dist_;
        const SDFBoxData& box_data = box_data_list_[box_id];
        return box_data.distance_buffer[toAddress(idx, box_data.field_voxel_num)];
    }
    
    /**
     * @brief Convert position to voxel index
     */
    void posToIndex(const Eigen::Vector3d& pos, int box_id, Eigen::Vector3i& idx) const {
        const SDFBoxData& box_data = box_data_list_[box_id];
        for (size_t i = 0; i < 3; ++i)
            idx(i) = floor((pos(i) - box_data.field_origin(i)) * resolution_inv_);
    }
    
    /**
     * @brief Check if position is inside box bounds
     */
    bool isInBox(const Eigen::Vector3d& pos, int box_id) const {
        const SDFBoxData& box_data = box_data_list_[box_id];
        if (pos(0) < box_data.field_min_boundary(0) + 1e-4 || 
            pos(1) < box_data.field_min_boundary(1) + 1e-4 ||
            pos(2) < box_data.field_min_boundary(2) + 1e-4)
            return false;
        if (pos(0) > box_data.field_max_boundary(0) - 1e-4 || 
            pos(1) > box_data.field_max_boundary(1) - 1e-4 ||
            pos(2) > box_data.field_max_boundary(2) - 1e-4)
            return false;
        return true;
    }
    
    /**
     * @brief Check if voxel index is inside box bounds
     */
    bool isInBox(const Eigen::Vector3i& idx, int box_id) const {
        const SDFBoxData& box_data = box_data_list_[box_id];
        if (idx(0) < 0 || idx(1) < 0 || idx(2) < 0)
            return false;
        if (idx(0) > box_data.field_voxel_num(0) - 1 || 
            idx(1) > box_data.field_voxel_num(1) - 1 ||
            idx(2) > box_data.field_voxel_num(2) - 1)
            return false;
        return true;
    }

private:
    /**
     * @brief Initialize field boundaries and allocate distance buffer
     */
    void initializeBox(const Eigen::Vector3d& field_size,
                      const Eigen::Vector3d& field_origin,
                      double shell_thickness,
                      SDFBoxData& box_data) {
        box_data.field_size = field_size;
        box_data.shell_thickness = shell_thickness;
        
        box_data.field_origin = -field_origin - Eigen::Vector3d(
            0.5 * field_size(0), 0.5 * field_size(1), 0.5 * field_size(2));
        
        box_data.field_min_boundary = box_data.field_origin;
        box_data.field_max_boundary = box_data.field_origin + field_size;
        
        for (int i = 0; i < 3; ++i)
            box_data.field_voxel_num(i) = ceil(field_size(i) * resolution_inv_);
        
        // Initialize distance buffer
        int buffer_size = box_data.field_voxel_num(0) * 
                         box_data.field_voxel_num(1) * 
                         box_data.field_voxel_num(2);
        box_data.distance_buffer = std::vector<double>(buffer_size, default_dist_);
    }
    
    /**
     * @brief Load disk and line geometric primitives from ROS parameters
     */
    void loadGeometricPrimitives(ros::NodeHandle& nh, int box_id, SDFBoxData& box_data) {
        int disk_num, line_num;
        nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/disk_num", disk_num, 0);
        nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/line_num", line_num, 0);
        
        // Load disk parameters
        if (disk_num > 0) {
            for (int j = 0; j < disk_num; j++) {
                DiskParams disk_params;
                
                nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/disk" + 
                        std::to_string(j) + "/radius", disk_params.radius, 0.0);
                
                nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/disk" + 
                        std::to_string(j) + "/x", disk_params.T(0), 0.0);
                nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/disk" + 
                        std::to_string(j) + "/y", disk_params.T(1), 0.0);
                nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/disk" + 
                        std::to_string(j) + "/z", disk_params.T(2), 0.0);
                
                double roll, pitch, yaw;
                nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/disk" + 
                        std::to_string(j) + "/roll", roll, 0.0);
                nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/disk" + 
                        std::to_string(j) + "/pitch", pitch, 0.0);
                nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/disk" + 
                        std::to_string(j) + "/yaw", yaw, 0.0);
                
                roll *= M_PI / 180.0;
                pitch *= M_PI / 180.0;
                yaw *= M_PI / 180.0;
                
                disk_params.R = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
                               Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                               Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
                
                box_data.disk_params_list.push_back(disk_params);
            }
            std::cout << "[SDFMap]: box_" << box_id << " loaded " << disk_num << " disks" << std::endl;
        }
        
        // Load line parameters
        if (line_num > 0) {
            for (int j = 0; j < line_num; j++) {
                LineParams line_params;
                
                nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/line" + 
                        std::to_string(j) + "/length", line_params.length, 0.0);
                
                nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/line" + 
                        std::to_string(j) + "/x", line_params.T(0), 0.0);
                nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/line" + 
                        std::to_string(j) + "/y", line_params.T(1), 0.0);
                nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/line" + 
                        std::to_string(j) + "/z", line_params.T(2), 0.0);
                
                double roll, pitch, yaw;
                nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/line" + 
                        std::to_string(j) + "/roll", roll, 0.0);
                nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/line" + 
                        std::to_string(j) + "/pitch", pitch, 0.0);
                nh.param("ch_rc_sdf/box" + std::to_string(box_id) + "/line" + 
                        std::to_string(j) + "/yaw", yaw, 0.0);
                
                roll *= M_PI / 180.0;
                pitch *= M_PI / 180.0;
                yaw *= M_PI / 180.0;
                
                line_params.R = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()) *
                               Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                               Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
                
                box_data.line_params_list.push_back(line_params);
            }
            std::cout << "[SDFMap]: box_" << box_id << " loaded " << line_num << " lines" << std::endl;
        }
    }
    
    /**
     * @brief Compute SDF values for a box given geometric primitives
     */
    void feedBoxSDF(SDFBoxData& box_data, int box_id) {
        std::cout << "[SDFMap]: Computing SDF for box_" << box_id << std::endl;
        
        for (int i = 0; i < box_data.field_voxel_num(0); ++i) {
            for (int j = 0; j < box_data.field_voxel_num(1); ++j) {
                for (int k = 0; k < box_data.field_voxel_num(2); ++k) {
                    Eigen::Vector3d pt;
                    indexToPos(Eigen::Vector3i(i, j, k), box_data.field_origin, pt);
                    
                    double dist = 999.9;
                    double dist_temp;
                    
                    // Compute distance to disks
                    for (const auto& disk : box_data.disk_params_list) {
                        dist2disk(disk, pt, dist_temp);
                        if (dist_temp < dist)
                            dist = dist_temp;
                    }
                    
                    // Compute distance to lines
                    for (const auto& line : box_data.line_params_list) {
                        dist2line(line, pt, dist_temp);
                        if (dist_temp < dist)
                            dist = dist_temp;
                    }
                    
                    dist -= box_data.shell_thickness;
                    int addr = toAddress(Eigen::Vector3i(i, j, k), box_data.field_voxel_num);
                    box_data.distance_buffer[addr] = dist;
                }
            }
        }
    }
    
    /**
     * @brief Compute distance from point to disk
     */
    void dist2disk(const DiskParams& disk_param, const Eigen::Vector3d& pt, double& dist) const {
        double d2o_h = sqrt(pt.x() * pt.x() + pt.y() * pt.y()) - disk_param.radius;
        
        if (d2o_h <= disk_param.radius)
            dist = fabs(pt.z());
        else {
            double d2rim = d2o_h - disk_param.radius;
            dist = sqrt(d2rim * d2rim + pt.z() * pt.z());
        }
    }
    
    /**
     * @brief Compute distance from point to line segment
     */
    void dist2line(const LineParams& line_param, const Eigen::Vector3d& pt, double& dist) const {
        Eigen::Vector3d pt_local = line_param.R.inverse() * (pt - line_param.T);
        
        double len = line_param.length;
        if (pt_local.x() > len) {
            double dis2end_x = pt_local.x() - len;
            dist = sqrt(dis2end_x * dis2end_x + pt_local.y() * pt_local.y() + 
                       pt_local.z() * pt_local.z());
        } else if (pt_local.x() < 0) {
            double dis2start_x = -pt_local.x();
            dist = sqrt(dis2start_x * dis2start_x + pt_local.y() * pt_local.y() + 
                       pt_local.z() * pt_local.z());
        } else {
            dist = sqrt(pt_local.y() * pt_local.y() + pt_local.z() * pt_local.z());
        }
    }
    
    /**
     * @brief Convert voxel index to position
     */
    void indexToPos(const Eigen::Vector3i& id, const Eigen::Vector3d& map_origin, 
                   Eigen::Vector3d& pos) const {
        for (size_t i = 0; i < 3; ++i)
            pos(i) = (id(i) + 0.5) * resolution_ + map_origin(i);
    }
    
    /**
     * @brief Convert voxel index to linear address
     */
    int toAddress(const Eigen::Vector3i& id, const Eigen::Vector3i& map_voxel_num) const {
        return id[0] * map_voxel_num(1) * map_voxel_num(2) + id[1] * map_voxel_num(2) + id[2];
    }

private:
    std::vector<SDFBoxData> box_data_list_;
    int box_num_ = 0;
    double resolution_ = 0.1;
    double resolution_inv_ = 10.0;
    double default_dist_ = 5.0;
};

} // namespace clutter_hand

#endif // _SDF_MAP_HPP_
