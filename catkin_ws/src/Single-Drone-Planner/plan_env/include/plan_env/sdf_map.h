#ifndef _SDF_MAP_H
#define _SDF_MAP_H

#include <Eigen/Eigen>
#include <Eigen/StdVector>

#include <queue>
#include <ros/ros.h>
#include <tuple>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <plan_env/visualization.hpp>

#include <random>  
#include <iostream>

#include <tf/tf.h>

#include "ch_rc_sdf/ch_rc_sdf.h"

using namespace std;

namespace cv {
    class Mat;
}

class RayCaster;

namespace fast_planner {
    struct MapParam;
    struct MapData;

struct CameraParam {
  double cx, cy, fx, fy;
  double min_range, max_range;
  double width, height;
};

struct IgnoreBox {
    Eigen::Vector3d pt_min, pt_max;
    Eigen::Vector3i idx_min, idx_max;
};

    class MapROS;

    class SDFMap {
    public:
        SDFMap();

        ~SDFMap();

        enum OCCUPANCY {
            UNKNOWN, FREE, OCCUPIED
        };

        void initMap(ros::NodeHandle &nh, std::shared_ptr<clutter_hand::CH_RC_SDF> rc_sdf_ptr);

        void inputPointCloud(const pcl::PointCloud<pcl::PointXYZ> &points, const int &point_num,
                             const Eigen::Vector3d &camera_pos);

        Eigen::Vector3i pos2idx(const Eigen::Vector3d &pt);

        Eigen::Vector3d idx2pos(const Eigen::Vector3i &id);

        void posToIndex(const Eigen::Vector3d &pos, Eigen::Vector3i &id);

        void indexToPos(const Eigen::Vector3i &id, Eigen::Vector3d &pos);

        void boundIndex(Eigen::Vector3i &id);

        int toAddress(const Eigen::Vector3i &id);

        int toAddress(const int &x, const int &y, const int &z);

        bool isInMap(const Eigen::Vector3d &pos);

        bool isInMap(const Eigen::Vector3i &idx);

        bool isUnknown(const Eigen::Vector3d &pos);

        bool isUnknown(const Eigen::Vector3i &id);

        // partical
        bool isOccupied(const Eigen::Vector3d &pos);

        bool isOccupied(const Eigen::Vector3i &id);

        bool isInfOccupied(const Eigen::Vector3d &pos);

        bool isInfOccupied(const Eigen::Vector3i &id);

        bool isValid(const Eigen::Vector3d &pos);

        bool isValid(const Eigen::Vector3i &id);

        bool checkRayValid(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1);

        // whole-body
        bool isOccupied(const Eigen::VectorXd &state);

        bool checkRayValid(const Eigen::VectorXd& s0, const Eigen::VectorXd& s1);

        bool isValid(const Eigen::VectorXd &state);

        bool isInBox(const Eigen::Vector3i &id);

        bool isInBox(const Eigen::Vector3d &pos);

        bool isFrontUnk(const Eigen::Vector3i &id);

        void findSecondUnk(const Eigen::Vector3i &id, vector<Eigen::Vector3i> &sec_unk_pts);

        void boundBox(Eigen::Vector3d &low, Eigen::Vector3d &up);

        int getOccupancy(const Eigen::Vector3d &pos);

        int getOccupancy(const Eigen::Vector3i &id);

        void setOccupied(const Eigen::Vector3d &pos, const int &occ = 1);

        int getInflateOccupancy(const Eigen::Vector3d &pos);

        int getInflateOccupancy(const Eigen::Vector3i &id);

        double getDistance(const Eigen::Vector3d &pos);

        double getDistance(const Eigen::Vector3i &id);

        double getDistWithGrad(const Eigen::Vector3d &pos, Eigen::Vector3d &grad);

        void updateESDF3d();

        void resetBuffer();

        void resetBuffer(const Eigen::Vector3d &min, const Eigen::Vector3d &max);

        void getRegion(Eigen::Vector3d &ori, Eigen::Vector3d &size);

        void getBox(Eigen::Vector3d &bmin, Eigen::Vector3d &bmax);

        void getUpdatedBox(Eigen::Vector3d &bmin, Eigen::Vector3d &bmax, bool reset = false);

        double getResolution();

        Eigen::Vector3d getMinBound();

        Eigen::Vector3d getMaxBound();

        double getInfDis();
        
        int getVoxelNum();

        void setCamearPose2free(const Eigen::Vector3d& cam_p);

        void getAABBPoints(Eigen::MatrixXd& aabb_pts, 
                        const Eigen::Vector3d& start_pos, 
                        const Eigen::Vector3d& end_pos);

        void getAABBPoints(std::vector<Eigen::Vector3d>& aabb_pts, 
                        const Eigen::Vector3d& start_pos, 
                        const Eigen::Vector3d& end_pos);

        void getAABBPointsSample(std::vector<Eigen::Vector3d>& aabb_pts, 
                        const std::vector<Eigen::Vector3d>& sample_pts);


        void getRelatedUnkPcl(const std::vector<Eigen::Vector3d> &cams_p, pcl::PointCloud<pcl::PointXYZ>& pcl_unk);

        bool inPriorBoxes(const Eigen::Vector3i& id);
        // void idSort(Eigen::Vector3i& min_id, Eigen::Vector3i& max_id);
        void poseSort(Eigen::Vector3d& min_pt, Eigen::Vector3d& max_pt);

        //! clutter-hand

        void setIgnoreBox(const Eigen::Vector3d& pt0, const Eigen::Vector3d& pt1)
        {
            Eigen::Vector3d ignore_margin = 1.0*Eigen::Vector3d::Ones();

            ignore_box_.pt_min.x() = pt0(0)<pt1(0)?pt0(0):pt1(0);
            ignore_box_.pt_min.y() = pt0(1)<pt1(1)?pt0(1):pt1(1);
            ignore_box_.pt_min.z() = pt0(2)<pt1(2)?pt0(2):pt1(2);
            ignore_box_.pt_min -= ignore_margin;
            ignore_box_.idx_min = pos2idx(ignore_box_.pt_min);

            ignore_box_.pt_max.x() = pt0(0)>pt1(0)?pt0(0):pt1(0);
            ignore_box_.pt_max.y() = pt0(1)>pt1(1)?pt0(1):pt1(1);
            ignore_box_.pt_max.z() = pt0(2)>pt1(2)?pt0(2):pt1(2);
            ignore_box_.pt_max += ignore_margin;
            ignore_box_.idx_max = pos2idx(ignore_box_.pt_max);
            ignore_obs_flag_ = true;
        }
        void cancelIgnoreBox(){ignore_obs_flag_ = false;}

        bool isInIgnoreBox(const Eigen::Vector3d& pos){
            if(ignore_obs_flag_ && 
               (pos(0) > ignore_box_.pt_min(0) && pos(0) < ignore_box_.pt_max(0) &&
                pos(1) > ignore_box_.pt_min(1) && pos(1) < ignore_box_.pt_max(1) &&
                pos(2) > ignore_box_.pt_min(2) && pos(2) < ignore_box_.pt_max(2)))
                return true;

                return false;
        }

        bool isInIgnoreBox(const Eigen::Vector3i& idx){
            if(ignore_obs_flag_ && 
                idx(0) > ignore_box_.idx_min(0) && idx(0) < ignore_box_.idx_max(0) &&
                idx(1) > ignore_box_.idx_min(1) && idx(1) < ignore_box_.idx_max(1) &&
                idx(2) > ignore_box_.idx_min(2) && idx(2) < ignore_box_.idx_max(2))
                return true;

                return false;
        }

        void debugCheckRayValid(const Eigen::Vector3d& pt, const double& yaw)
        {
            double height = 1.5;
            if(pt_yaw_vec_.size() == 0)
            {
                pt_yaw_vec_.push_back(std::make_pair(pt, yaw));
                std::cout << "get goal_0: " << pt.transpose() << " " << yaw << std::endl;
            }
            else if(pt_yaw_vec_.size() == 1){
                pt_yaw_vec_.clear();
                std::cout << "get goal_2: " << pt.transpose() << " " << yaw << std::endl;
                Eigen::VectorXd in_states(7), out_states(7);
                in_states << pt_yaw_vec_[0].first.head(2), height , pt_yaw_vec_[0].second, 0.1, 0.2, -0.3;
                out_states << pt.head(2), height, yaw, 0, 1.1, -1.0;
                bool is_valid = checkRayValid(in_states, out_states);
            }
        }

        void debugIsOccupied(const Eigen::Vector3d& pt, const double& yaw)
        {
            
        }


    private:
        void clearAndInflateLocalMap();

        static void inflatePoint(const Eigen::Vector3i &pt, int step, vector<Eigen::Vector3i> &pts);

        void setCacheOccupancy(const int &adr, const int &occ);

        Eigen::Vector3d closetPointInMap(const Eigen::Vector3d &pt, const Eigen::Vector3d &camera_pt);

        void resetVisBuffer();
        template<typename F_get_val, typename F_set_val>
        void fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim);

        unique_ptr<MapParam> mp_;
        unique_ptr<MapData> md_;
        unique_ptr<MapROS> mr_;
        unique_ptr<RayCaster> caster_;
        std::shared_ptr<visualization_map::Visualization> vis_ptr_;
        std::vector<Eigen::Vector3d> img_pix_dir_vec_;

        std::atomic_flag vis_lock_ = ATOMIC_FLAG_INIT;

        double fov_v_,fov_h_;

        std::vector<Eigen::Vector3i> prior_boxes_max_id_,prior_boxes_min_id_;

        friend MapROS;
        
        CameraParam cam_param_;

        //! clutter-hand
        std::shared_ptr<clutter_hand::CH_RC_SDF> rc_sdf_ptr_;

        // 抓取时暂时消除部分occ
        bool ignore_obs_flag_ = false;
        IgnoreBox ignore_box_;

        //! debug
        std::vector<std::pair<Eigen::Vector3d, double>> pt_yaw_vec_;

    public:
        typedef std::shared_ptr<SDFMap> Ptr;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    struct MapParam {
        // map properties
        Eigen::Vector3d map_origin_, map_size_;
        Eigen::Vector3d map_min_boundary_, map_max_boundary_;
        Eigen::Vector3i map_voxel_num_;
        double resolution_, resolution_inv_;
        double obstacles_inflation_;
        double virtual_ceil_height_, ground_height_;
        Eigen::Vector3i box_min_, box_max_;
        Eigen::Vector3d box_mind_, box_maxd_;
        double default_dist_;
        bool optimistic_, signed_dist_;
        // map fusion
        double p_hit_, p_miss_, p_min_, p_max_, p_occ_;  // occupancy probability
        double prob_hit_log_, prob_miss_log_, clamp_min_log_, clamp_max_log_, min_occupancy_log_;  // logit
        double max_ray_length_;
        double local_bound_inflate_;
        int local_map_margin_;
        double unknown_flag_;

        Eigen::Vector3d aabb_box_;
    };

    struct MapData {
        // main map data, occupancy of each voxel and Euclidean distance
        std::vector<double> occupancy_buffer_;
        std::vector<char> occupancy_buffer_inflate_;
        std::vector<double> distance_buffer_neg_;
        std::vector<double> distance_buffer_;
        std::vector<double> tmp_buffer1_;
        std::vector<double> tmp_buffer2_;
        std::vector<bool> vis_buffer_;

        // for info map
        std::vector<double> loss_buffer_all_; // 0~1
        std::vector<double> loss_buffer_temp_; // 求和
        std::vector<int> loss_cnt_buffer_; // 计数;

        // data for updating
        vector<short> count_hit_, count_miss_, count_hit_and_miss_;
        vector<char> flag_rayend_, flag_visited_;
        char raycast_num_;
        queue<int> cache_voxel_;
        Eigen::Vector3i local_bound_min_, local_bound_max_;
        Eigen::Vector3d update_min_, update_max_;
        bool reset_updated_box_;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    inline Eigen::Vector3i SDFMap::pos2idx(const Eigen::Vector3d &pt) {
        Eigen::Vector3i id;
        posToIndex(pt, id);
        return id;
    }

    inline Eigen::Vector3d SDFMap::idx2pos(const Eigen::Vector3i &id) {
        Eigen::Vector3d pt;
        indexToPos(id, pt);
        return pt;
    }

    inline void SDFMap::posToIndex(const Eigen::Vector3d &pos, Eigen::Vector3i &id) {
        for (size_t i = 0; i < 3; ++i)
            id(i) = floor((pos(i) - mp_->map_origin_(i)) * mp_->resolution_inv_);
    }

    inline void SDFMap::indexToPos(const Eigen::Vector3i &id, Eigen::Vector3d &pos) {
        for (size_t i = 0; i < 3; ++i)
            pos(i) = (id(i) + 0.5) * mp_->resolution_ + mp_->map_origin_(i);
    }

    inline void SDFMap::boundIndex(Eigen::Vector3i &id) {
        Eigen::Vector3i id1;
        id1(0) = max(min(id(0), mp_->map_voxel_num_(0) - 1), 0);
        id1(1) = max(min(id(1), mp_->map_voxel_num_(1) - 1), 0);
        id1(2) = max(min(id(2), mp_->map_voxel_num_(2) - 1), 0);
        id = id1;
    }

    inline int SDFMap::toAddress(const int &x, const int &y, const int &z) {
        return x * mp_->map_voxel_num_(1) * mp_->map_voxel_num_(2) + y * mp_->map_voxel_num_(2) + z;
    }

    inline int SDFMap::toAddress(const Eigen::Vector3i &id) {
        return toAddress(id[0], id[1], id[2]);
    }

    inline bool SDFMap::isInMap(const Eigen::Vector3d &pos) {
        if (pos(0) < mp_->map_min_boundary_(0) + 1e-4 || pos(1) < mp_->map_min_boundary_(1) + 1e-4 ||
            pos(2) < mp_->map_min_boundary_(2) + 1e-4)
            return false;
        if (pos(0) > mp_->map_max_boundary_(0) - 1e-4 || pos(1) > mp_->map_max_boundary_(1) - 1e-4 ||
            pos(2) > mp_->map_max_boundary_(2) - 1e-4)
            return false;
        return true;
    }

    inline bool SDFMap::isInMap(const Eigen::Vector3i &idx) {
        if (idx(0) < 0 || idx(1) < 0 || idx(2) < 0) return false;
        if (idx(0) > mp_->map_voxel_num_(0) - 1 || idx(1) > mp_->map_voxel_num_(1) - 1 ||
            idx(2) > mp_->map_voxel_num_(2) - 1)
            return false;
        return true;
    }

    inline bool SDFMap::isUnknown(const Eigen::Vector3d &pos) {
        Eigen::Vector3i id;
        posToIndex(pos, id);
        return isUnknown(id);
    }

    inline bool SDFMap::isUnknown(const Eigen::Vector3i &id) {
        if (!isInMap(id)) return true;
        return md_->occupancy_buffer_[toAddress(id)] < mp_->clamp_min_log_ - 1e-3;
    }

    //! ------------- partical -------------
    inline bool SDFMap::isOccupied(const Eigen::Vector3d &pos) {
        Eigen::Vector3i id;
        posToIndex(pos, id);
        return isOccupied(id);
    }

    inline bool SDFMap::isOccupied(const Eigen::Vector3i &id) {
        if(!isInMap(id)) return true;
        if(isInIgnoreBox(id)) return false;
        return md_->occupancy_buffer_[toAddress(id)] > mp_->min_occupancy_log_;
    }

    // ch_rc_sdf related
    inline bool SDFMap::isOccupied(const Eigen::VectorXd &state) {

    // std::cout << "[isOccupiedXd] 1 state: " << state.transpose() << std::endl;

        if (state.size() == 3)
            return isOccupied(Eigen::Vector3d(state));
        else if (state.size() == 4 + rc_sdf_ptr_->getDofNum()) {

            // 1 get base states
            Eigen::Vector3d p_body = state.head(3);
            Eigen::Matrix3d R_b2w; R_b2w = Eigen::AngleAxisd(state(3), Eigen::Vector3d::UnitZ());
            Eigen::Vector3i id_body = pos2idx(p_body);

    // std::cout << "[isOccupied] 2 p_body: " << p_body.transpose() << std::endl;

            // 2 get AABB box
            Eigen::Vector3d min_cut_p, max_cut_p;
            for(int i=0; i<3; ++i){
                min_cut_p(i) = p_body(i)-0.5*mp_->aabb_box_(i);
                max_cut_p(i) = p_body(i)+0.5*mp_->aabb_box_(i);
            }

            Eigen::Vector3i min_cut, max_cut;
            posToIndex(min_cut_p, min_cut);
            posToIndex(max_cut_p, max_cut);
            // boundIndex(min_cut);
            // boundIndex(max_cut);

            // 3 check occ
            for(int x=min_cut(0); x<=max_cut(0); x++)
            for(int y=min_cut(1); y<=max_cut(1); y++)
            for(int z=min_cut(2); z<=max_cut(2); z++){
                Eigen::Vector3i id(x, y, z);
                if(isOccupied(id)){
                    // trans to body frame
                    Eigen::Vector3d p_obs_body;
                    p_obs_body = R_b2w.transpose()*(idx2pos(id)-p_body);
                    
                    if(rc_sdf_ptr_->isInRob(state.tail(rc_sdf_ptr_->getDofNum()), p_obs_body))
                        return true;
                }
            }
            return false;
        }
        else
        {
            ROS_ERROR("isOccupied error state.size() == %d", state.size());
        }

    }

    inline bool SDFMap::isInfOccupied(const Eigen::Vector3d &pos) {
        Eigen::Vector3i id;
        posToIndex(pos, id);
        return isInfOccupied(id);
    }

    inline bool SDFMap::isInfOccupied(const Eigen::Vector3i &id) {
        if (!isInMap(id)) return true;
        if (isInIgnoreBox(id)) return false;
        return md_->occupancy_buffer_inflate_[toAddress(id)] == 1
               || md_->occupancy_buffer_[toAddress(id)] > mp_->min_occupancy_log_;
    }

    inline bool SDFMap::isValid(const Eigen::Vector3d &pos) {
        Eigen::Vector3i id;
        posToIndex(pos, id);
        return isValid(id);
    }

    inline bool SDFMap::isValid(const Eigen::Vector3i &id) {
        // return goal_method_== RVIZ_GOAL ? 
        // !isInfOccupied(id) : (!isInfOccupied(id) && !isUnknown(id));

        return !isInfOccupied(id);
    }
    //! ------------- partical -------------

    inline bool SDFMap::isInBox(const Eigen::Vector3i &id) {
        for (size_t i = 0; i < 3; ++i) {
            if (id[i] < mp_->box_min_[i] || id[i] >= mp_->box_max_[i]) {
                return false;
            }
        }
        return true;
    }

    inline bool SDFMap::isInBox(const Eigen::Vector3d &pos) {
        for (size_t i = 0; i < 3; ++i) {
            if (pos[i] <= mp_->box_mind_[i] || pos[i] >= mp_->box_maxd_[i]) {
                return false;
            }
        }
        return true;
    }

    inline void SDFMap::boundBox(Eigen::Vector3d &low, Eigen::Vector3d &up) {
        for (size_t i = 0; i < 3; ++i) {
            low[i] = max(low[i], mp_->box_mind_[i]);
            up[i] = min(up[i], mp_->box_maxd_[i]);
        }
    }

    inline int SDFMap::getOccupancy(const Eigen::Vector3i &id) {
        if (!isInMap(id)) return -1;
        double occ = md_->occupancy_buffer_[toAddress(id)];
        if (occ < mp_->clamp_min_log_ - 1e-3) return UNKNOWN;
        if (occ > mp_->min_occupancy_log_) return OCCUPIED;
        return FREE;
    }

    inline int SDFMap::getOccupancy(const Eigen::Vector3d &pos) {
        Eigen::Vector3i id;
        posToIndex(pos, id);
        return getOccupancy(id);
    }

    inline void SDFMap::setOccupied(const Eigen::Vector3d &pos, const int &occ) {
        if (!isInMap(pos)) return;
        Eigen::Vector3i id;
        posToIndex(pos, id);
        md_->occupancy_buffer_inflate_[toAddress(id)] = occ;
    }

    inline int SDFMap::getInflateOccupancy(const Eigen::Vector3i &id) {
        if (!isInMap(id)) return -1;
        return int(md_->occupancy_buffer_inflate_[toAddress(id)]);
    }

    inline int SDFMap::getInflateOccupancy(const Eigen::Vector3d &pos) {
        Eigen::Vector3i id;
        posToIndex(pos, id);
        return getInflateOccupancy(id);
    }

    inline double SDFMap::getDistance(const Eigen::Vector3i &id) {
        if (!isInMap(id)) return -1;
        return md_->distance_buffer_[toAddress(id)];
    }

    inline double SDFMap::getDistance(const Eigen::Vector3d &pos) {
        Eigen::Vector3i id;
        posToIndex(pos, id);
        return getDistance(id);
    }

    inline void SDFMap::inflatePoint(const Eigen::Vector3i &pt, int step, vector<Eigen::Vector3i> &pts) {
        int num = 0;

        /* ---------- all inflate ---------- */
        for (int x = -step; x <= step; ++x)
            for (int y = -step; y <= step; ++y)
                for (int z = -step; z <= step; ++z) {
                    pts[num++] = Eigen::Vector3i(pt(0) + x, pt(1) + y, pt(2) + z);
                }
    }

    inline bool SDFMap::isFrontUnk(const Eigen::Vector3i &id){
        int step = 1;
        vector<Eigen::Vector3i> pts((2*step+1)*(2*step+1)*(2*step+1));
        inflatePoint(id, step, pts);

        for (size_t i = 0; i < pts.size(); ++i){
            //? 思考：地图边界的unk需要吗
            // 应该不需要，因为之后不会有gaus需要mask
            if(!isInMap(pts[i]))
                return false;

            // 只要不是unk就在边界
            if(!isUnknown(pts[i]))
                return true;
        }

        return false;
    }

// 寻找第二层unk
    inline void SDFMap::findSecondUnk(const Eigen::Vector3i &id, vector<Eigen::Vector3i> &sec_unk_pts){
        int step = 1;
        vector<Eigen::Vector3i> pts((2*step+1)*(2*step+1)*(2*step+1));
        inflatePoint(id, step, pts);

        for (size_t i = 0; i < pts.size(); ++i){
            if(!isInMap(pts[i]))
                continue;

            

        }
    }

    inline void SDFMap::poseSort(Eigen::Vector3d& min_pt, Eigen::Vector3d& max_pt)
    {
        for(size_t i = 0; i < 3; i++){
            if(min_pt(i) > max_pt(i))
                swap(min_pt(i), max_pt(i));
        }
    }

// palworld
    inline bool SDFMap::inPriorBoxes(const Eigen::Vector3i& id)
    {
        for(size_t i = 0; i < prior_boxes_min_id_.size(); i++){
            if(id(0) >= prior_boxes_min_id_[i](0) && id(0) < prior_boxes_max_id_[i](0) &&
               id(1) >= prior_boxes_min_id_[i](1) && id(1) < prior_boxes_max_id_[i](1) &&
               id(2) >= prior_boxes_min_id_[i](2) && id(2) < prior_boxes_max_id_[i](2))
                return true;
        }

        return false;
    }

}
#endif