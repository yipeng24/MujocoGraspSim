#pragma once
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace rotation_util{
class RotUtil{
public:
    // quaternion to euler: roll, pitch, yaw
    static Eigen::Matrix<double, 3, 1> quaternion2euler(const Eigen::Quaterniond &q){
        Eigen::Matrix<double, 3, 1> e;

        // roll (x-axis rotation)
        double sinr_cosp = 2 * (q.w() * q.x() + q.y() * q.z());
        double cosr_cosp = 1 - 2 * (q.x() * q.x() + q.y() * q.y());
        e(0) = std::atan2(sinr_cosp, cosr_cosp);

        // pitch (y-axis rotation)
        double sinp = 2 * (q.w() * q.y() - q.z() * q.x());
        if (std::abs(sinp) >= 1)
            e(1) = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
        else
            e(1) = std::asin(sinp);

        // yaw (z-axis rotation)
        double siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
        e(2) = std::atan2(siny_cosp, cosy_cosp);

        return e;
    }

    // obtain yaw from quaternion
    static double quaternion2yaw(const Eigen::Quaterniond &q){
        // yaw (z-axis rotation)
        double siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = 1 - 2 * (q.y() * q.y() + q.z() * q.z());
        return std::atan2(siny_cosp, cosy_cosp);
    }

    // euler to quaternion: roll, pitch, yaw
    static Eigen::Quaterniond euler2quaternion(const double& roll, const double& pitch, const double& yaw){
        // Abbreviations for the various angular functions
        double cy = cos(yaw * 0.5);
        double sy = sin(yaw * 0.5);
        double cp = cos(pitch * 0.5);
        double sp = sin(pitch * 0.5);
        double cr = cos(roll * 0.5);
        double sr = sin(roll * 0.5);

        Eigen::Quaterniond q;
        q.w() = cr * cp * cy + sr * sp * sy;
        q.x() = sr * cp * cy - cr * sp * sy;
        q.y() = cr * sp * cy + sr * cp * sy;
        q.z() = cr * cp * sy - sr * sp * cy;

        return q;
    }

    // euler to quaternion: roll, pitch, yaw
    static Eigen::Quaterniond yaw2quaternion(const double& yaw){
        // Abbreviations for the various angular functions
        double cy = cos(yaw * 0.5);
        double sy = sin(yaw * 0.5);

        Eigen::Quaterniond q;
        q.w() = cy;
        q.x() = 0;
        q.y() = 0;
        q.z() = sy;

        return q;
    }

    // euler to quaternion: roll, pitch, yaw
    static Eigen::Quaterniond euler2quaternion(const Eigen::Vector3d& rpy){
        Eigen::Quaterniond q = euler2quaternion(rpy(0), rpy(1), rpy(2));
        return q;
    }

    // rotation matrix to rotation axis-angle
    static void R2axial_angle(const Eigen::Matrix3d& R, Eigen::Vector3d& v, double& theta){
        double epsilon = 1E-12;
        std::vector<double> data;
        for (int i = 0; i < 3; i++){
            for (int j = 0; j < 3; j++){
                data.push_back(R(i, j));
            }
        }
        double vv = (data[0] + data[4] + data[8] - 1.0f) / 2.0f;
        if (fabs(vv) < 1 - epsilon) {
            theta = acos(vv);
            v.x() = 1 / (2 * sin(theta)) * (data[7] - data[5]);
            v.y() = 1 / (2 * sin(theta)) * (data[2] - data[6]);
            v.z() = 1 / (2 * sin(theta)) * (data[3] - data[1]);
        } else {
            if (vv > 0.0f) {
            // \theta = 0, diagonal elements approaching 1
            theta = 0;
            v.x() = 0;
            v.y() = 0;
            v.z() = 1;
            } else {
            // \theta = \pi
            // find maximum element in the diagonal elements
            theta = M_PI;
            if (data[0] >= data[4] && data[0] >= data[8]) {
                // calculate x first
                v.x() = sqrt((data[0] + 1) / 2);
                v.y() = data[1] / (2 * v.x());
                v.z() = data[2] / (2 * v.x());
            } else if (data[4] >= data[0] && data[4] >= data[8]) {
                // calculate y first
                v.y() = sqrt((data[4] + 1) / 2);
                v.x() = data[3] / (2 * v.y());
                v.z() = data[5] / (2 * v.y());
            } else {
                // calculate z first
                v.z() = sqrt((data[8] + 1) / 2);
                v.x() = data[6] / (2 * v.z());
                v.y() = data[7] / (2 * v.z());
            }
            }
        }
    }

    // rotation axis-angle to rotation matrix
    static void axial_angle2R(const Eigen::Vector3d& v, const double& theta,  Eigen::Matrix3d& R){
        double x, y, z;
        x = v.x();
        y = v.y();
        z = v.z();
        R(0,0) = x * x * (1 - cos(theta)) + cos(theta);
        R(0,1) = x * y * (1 - cos(theta)) - z * sin(theta);
        R(0,2) = x * z * (1 - cos(theta)) + y * sin(theta);
        R(1,0) = x * y * (1 - cos(theta)) + z * sin(theta);
        R(1,1) = y * y * (1 - cos(theta)) + cos(theta);
        R(1,2) = y * z * (1 - cos(theta)) - x * sin(theta);
        R(2,0) = x * z * (1 - cos(theta)) - y * sin(theta);
        R(2,1) = y * z * (1 - cos(theta)) + x * sin(theta);
        R(2,2) = z * z * (1 - cos(theta)) + cos(theta);
    }

    static Eigen::Matrix2d angle2rot_matrix2D(const double& angle){
        Eigen::Rotation2Dd rot(angle);
        return rot.toRotationMatrix();
    }

    static double rot_matrix2D2angle(const Eigen::Matrix2d& mat){
        Eigen::Rotation2Dd rot;
        rot.fromRotationMatrix(mat);
        return rot.angle();
    }

    // calculate (ang_s -> ang_t) on SO(2), return within -pi~pi, ang_s and ang_t can be arbitrary value
    static double error_angle(const double& ang_s, const double& ang_t){
        Eigen::Matrix2d s = angle2rot_matrix2D(ang_s);
        Eigen::Matrix2d t = angle2rot_matrix2D(ang_t);
        Eigen::Matrix2d err = t*s.inverse();
        double err_angle = rot_matrix2D2angle(err); // -pi~pi
        return err_angle;
    }

    // the return angle is limited witin (start_angle - limit_d_angle, start_angle + limit_d_angle)
    static double truncate_error_angle(const double& start_angle, const double& expect_angle, const double& limit_d_angle){
        assert(limit_d_angle > 0.0);
        Eigen::Matrix2d s = angle2rot_matrix2D(start_angle);
        Eigen::Matrix2d t = angle2rot_matrix2D(expect_angle);
        Eigen::Matrix2d err = t*s.inverse();
        double err_angle = rot_matrix2D2angle(err); // -pi~pi
        double ret_angle;

        if (abs(err_angle) < limit_d_angle){
            ret_angle = rot_matrix2D2angle(t);
        }else{
            Eigen::Matrix2d ret = s * (angle2rot_matrix2D(limit_d_angle * (err_angle/ abs(err_angle))));
            ret_angle = rot_matrix2D2angle(ret);
        }
        return ret_angle;
    }

    static double rad2deg(const double& rad){
        return rad * 180.0 / M_PI;
    }

    // limit the rad-angle within (-pi, pi]
    static double rad_limit(const double& rad){
        double rad_n = rad;
        while (rad_n <= -M_PI){
            rad_n += 2*M_PI;
        }
        while (rad_n > M_PI){
            rad_n -= 2*M_PI;
        }
        return rad_n;
    }

};

} // namespace rotation_util