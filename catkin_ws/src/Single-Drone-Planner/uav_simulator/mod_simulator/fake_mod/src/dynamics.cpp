#include "fake_mod/dynamics.h"

namespace modquad
{
    // TODO 加入舵机动力学模拟
    void XModQuad::simOneStep(mavros_msgs::AttitudeTarget cmd, sensor_msgs::JointState cmd_theta)
    {
        // ==========================
        // 0. 解析舵机控制指令 (新增部分)
        // ==========================
        // 假设 joint state 的顺序对应的就是 arm 0, 1, 2 的顺序
        if (cmd_theta.position.size() >= 3)
        {
            theta_ref_ << cmd_theta.position[0], cmd_theta.position[1], cmd_theta.position[2];
        }

        // 如果包含速度前馈则读取，否则设为0
        if (cmd_theta.velocity.size() >= 3)
        {
            dtheta_ref_ << cmd_theta.velocity[0], cmd_theta.velocity[1], cmd_theta.velocity[2];
        }else{
            dtheta_ref_.setZero();
        }


        quadrotor_msgs::SimDebug sim_debug_msg;
        sim_debug_msg.header.stamp = ros::Time::now();
        sim_debug_msg.header.frame_id = "world";


        // ==========================
        // 1. 控制逻辑选择 (关键部分)
        // ==========================
        
        Eigen::Vector3d moment_des;
        // 检查指令掩码: 如果设置了 IGNORE_ATTITUDE (128)，则视为角速度控制
        // 或者根据你的业务逻辑自定义切换标志
        bool use_rate_control = (cmd.type_mask & mavros_msgs::AttitudeTarget::IGNORE_ATTITUDE);

        if (use_rate_control){
            // --- 角速度控制模式 (Rate Control) ---
            // 直接从消息中获取目标角速度 (rad/s)
            Eigen::Vector3d omega_des(cmd.body_rate.x, cmd.body_rate.y, cmd.body_rate.z);
            Eigen::Vector3d omega_now = now_state.omega;
            Eigen::Vector3d e_omega = omega_now - omega_des;

            // 角速度增益 (通常比姿态环的阻尼项大)
            Eigen::Vector3d Kr_rate(0.15, 0.15, 0.1); 
            moment_des = -Kr_rate.cwiseProduct(e_omega);

            // Debug 信息记录
            sim_debug_msg.des_rpy[0] = omega_des.x(); // 借用位置记录目标角速度
            sim_debug_msg.des_rpy[1] = omega_des.y();
            sim_debug_msg.des_rpy[2] = omega_des.z();
        }else{
            // --- 姿态控制模式 (Attitude Control - 你原有的逻辑) ---
            Eigen::Quaterniond q_des(cmd.orientation.w, cmd.orientation.x, cmd.orientation.y, cmd.orientation.z);
            q_des.normalize();

            Eigen::Vector3d des_ypr = uav_utils::quaternion_to_ypr(q_des);
            Eigen::Vector3d est_ypr = uav_utils::quaternion_to_ypr(Eigen::Quaterniond(now_state.R));

            Eigen::Vector3d err_rpy;
            err_rpy << - des_ypr(2) + est_ypr(2),
                    - des_ypr(1) + est_ypr(1),
                    - des_ypr(0) + est_ypr(0);
            for(size_t i=0; i<3; i++) LimitAngle(err_rpy(i));

            Eigen::Vector3d Kp_att(0.5, 0.5, 0.1);
            Eigen::Vector3d Kd_att(0.1, 0.1, 0.05);
            
            // Eigen::Vector3d Kp_att(0.2, 0.2, 0.05); // 显著降低比例增益
            // Eigen::Vector3d Kd_att(0.15, 0.15, 0.08); // 稍微增加微分项提供过阻尼感

            moment_des = -2.0 * Kp_att.cwiseProduct(err_rpy) - Kd_att.cwiseProduct(now_state.omega);
            
            // 填充 Debug RPY
            sim_debug_msg.des_rpy[0]= des_ypr(2); sim_debug_msg.des_rpy[1]= des_ypr(1); sim_debug_msg.des_rpy[2]= des_ypr(0);
        }

        double thrust_des = cmd.thrust * 1.5 * params.mass * params.g;

        // [T, Mx, My, Mz]^T
        Eigen::Vector4d thrust_moment_des;
        thrust_moment_des << thrust_des,
                            moment_des.x(),
                            moment_des.y(),
                            moment_des.z();
        
        sim_debug_msg.des_thr_mom[0] = thrust_moment_des(0);
        sim_debug_msg.des_thr_mom[1] = thrust_moment_des(1);
        sim_debug_msg.des_thr_mom[2] = thrust_moment_des(2);
        sim_debug_msg.des_thr_mom[3] = thrust_moment_des(3);

        // ros::Time t = ros::Time::now();
        // std::cout << "timestamp: " << t.sec << "." << t.nsec << std::endl;
        // std::cout << "thrust_moment_des: " << thrust_moment_des.transpose() << std::endl;

        // ================================
        // 4. 通过混控矩阵反解每个电机推力
        //    mix_matrix * thrusts = thrust_moment_des
        //    thrusts = [T1, T2, T3, T4]^T
        // ================================
        Eigen::Vector4d thrusts = params.mix_matrix.inverse() * thrust_moment_des;

        // 推力不能为负
        for (size_t i = 0; i < 4; ++i)
        {
            thrusts[i] = std::max(thrusts[i], 0.0);
        }

        // ================================
        // 5. 推力 -> 目标电机转速 rpm
        //    T_i = kf * omega_i^2,   omega[rad/s]
        //    rpm = omega * 30 / π
        // ================================
        for (size_t i = 0; i < 4; ++i)
        {
            // double omega_des = std::sqrt(thrusts[i] / params.kf);  // rad/s
            // double rpm_des   = omega_des * 30.0 / M_PI;

            double rpm_des   = std::sqrt(thrusts[i] / params.kf);

            rpm[i] = rpm_des;
        }

        // 如果你还想保留一点随机噪声，可以加在这里
        rpm += guassRandom4d(params.noise_rpm);

        // 饱和
        for (size_t i = 0; i < 4; i++)
        {
            rpm[i] = std::min(std::max(rpm[i], params.min_rpm), params.max_rpm);
        }

        // ================================
        // 6. Runge–Kutta 积分（保持原来逻辑不变）
        // ================================
        double dt_2 = params.time_resolution / 2.0;
        State k1 = getDiff(now_state);
        State k2 = getDiff(now_state + k1 * dt_2);
        State k3 = getDiff(now_state + k2 * dt_2);
        State k4 = getDiff(now_state + k3 * params.time_resolution);
        now_state = now_state + (k1 + k2*2 + k3*2 + k4) * (params.time_resolution/6.0);

        sim_time_ += params.time_resolution;

        imu.linear_acceleration.x = k1.v.x();
        imu.linear_acceleration.y = k1.v.y();
        imu.linear_acceleration.z = k1.v.z() + params.g;

        // ==== 下面这些 odom / imu / marker 的更新不动 ====
        Eigen::Vector3d now_pos = now_state.p + guassRandom3d(params.noise_pos);
        Eigen::Vector3d now_vel = now_state.v + guassRandom3d(params.noise_vel);
        Eigen::Vector3d now_omega = now_state.omega;
        Eigen::Vector3d now_acc = Eigen::Vector3d(imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z)
                                + guassRandom3d(params.noise_acc);
        Eigen::Quaterniond now_q(now_state.R);
        now_q.normalize();
        now_acc = now_state.R.transpose() * now_acc;

        ros::Time stamp = ros::Time::now();

        odom.header.stamp = stamp;
        odom.pose.pose.position.x = now_pos(0);
        odom.pose.pose.position.y = now_pos(1);
        odom.pose.pose.position.z = now_pos(2);
        odom.pose.pose.orientation.w = now_q.w();
        odom.pose.pose.orientation.x = now_q.x();
        odom.pose.pose.orientation.y = now_q.y();
        odom.pose.pose.orientation.z = now_q.z();
        odom.twist.twist.linear.x = now_vel(0);
        odom.twist.twist.linear.y = now_vel(1);
        odom.twist.twist.linear.z = now_vel(2);
        odom.twist.twist.angular.x = now_omega(0);
        odom.twist.twist.angular.y = now_omega(1);
        odom.twist.twist.angular.z = now_omega(2);
 
        imu.header.stamp = stamp;
        imu.orientation = odom.pose.pose.orientation;
        imu.angular_velocity.x = now_omega(0);
        imu.angular_velocity.y = now_omega(1);
        imu.angular_velocity.z = now_omega(2);
        imu.linear_acceleration.x = now_acc(0);
        imu.linear_acceleration.y = now_acc(1);
        imu.linear_acceleration.z = now_acc(2);
        
        sim_debug_msg.est_d_rpy[0] = now_omega(0);
        sim_debug_msg.est_d_rpy[1] = now_omega(1);
        sim_debug_msg.est_d_rpy[2] = now_omega(2);

        marker.header.stamp = stamp;
        marker.pose = odom.pose.pose;

        esc_tele.header.stamp = stamp;
        for (size_t i=0; i<4; i++)
        {
            double rpm_temp = now_state.rotor_angular_rate[i] * 30.0 / M_PI;
            esc_tele.esc_telemetry[i].rpm = int32_t(rpm_temp);
        }

        arm_js_.header.stamp = stamp;
        arm_js_.position[0] = now_state.theta(0);
        arm_js_.position[1] = now_state.theta(1);
        arm_js_.position[2] = now_state.theta(2);
        arm_js_.velocity[0] = now_state.dtheta(0);
        arm_js_.velocity[1] = now_state.dtheta(1);
        arm_js_.velocity[2] = now_state.dtheta(2);

        // ★ 在这里调用刚才封装好的 tf 函数
        publishWorldToBaseTf(now_pos, now_q, stamp);

        sim_debug_msg.Fw[0] = dist_force_world_(0);
        sim_debug_msg.Fw[1] = dist_force_world_(1);
        sim_debug_msg.Fw[2] = dist_force_world_(2);

        sim_debug_msg.Mb[0] = dist_moment_body_(0);
        sim_debug_msg.Mb[1] = dist_moment_body_(1);
        sim_debug_msg.Mb[2] = dist_moment_body_(2);

        sim_debug_msg_ = sim_debug_msg;
    }

    State XModQuad::getDiff(const State &state)
    {
        // std::cout << "----------------- in getDiff -----------------" << std::endl;

        State state_dot;

        // Re-orthonormalize R (polar decomposition)
        Eigen::LLT<Eigen::Matrix3d> llt(state.R.transpose() * state.R);
        Eigen::Matrix3d             P = llt.matrixL();
        Eigen::Matrix3d             R = state.R * P.inverse();

        // rotor drag variables
        Eigen::Vector3d             D(params.rotor_drag_dx, params.rotor_drag_dy, params.rotor_drag_dz);
        Eigen::Vector3d             rotor_drag = -R*D.asDiagonal()*R.transpose()*state.v;
        double                      v_h = state.v.dot(R.col(0)+R.col(1));
        double                      drag_f = params.rotor_drag_kh * v_h * v_h;

        // get thrust and moment
        Eigen::Vector4d now_rpm = state.rotor_angular_rate * 30.0 / M_PI;
        Eigen::Vector4d thrusts = params.kf * now_rpm.array().square();
        Eigen::Vector4d thrust_moment = params.mix_matrix * thrusts;

        // add angular acceleration of rotors and gyroscopic effects
        Eigen::Vector3d             moment = thrust_moment.tail(3);
        Eigen::Vector4d             tau_x_gyro(-state.omega(1), -state.omega(1), state.omega(1), state.omega(1));
        Eigen::Vector4d             tau_y_gyro(state.omega(0), state.omega(0), -state.omega(0), -state.omega(0));
        Eigen::Vector4d             tau_z_rotor_acc(-1.0, -1.0, 1.0, 1.0);
        Eigen::Vector4d             rotor_angular_acc = (rpm * M_PI / 30.0 - state.rotor_angular_rate) / params.rotor_time_constant;
        
        moment(0) += params.Ip * tau_x_gyro.dot(now_rpm);
        moment(1) += params.Ip * tau_y_gyro.dot(now_rpm);
        moment(2) += params.Ip * tau_z_rotor_acc.dot(rotor_angular_acc * 30.0 / M_PI);
        moment    += Eigen::Vector3d::Constant(params.ext_tau);

        Eigen::Matrix3d omega_vee(Eigen::Matrix3d::Zero());
        omega_vee(2, 1) = state.omega(0);
        omega_vee(1, 2) = -state.omega(0);
        omega_vee(0, 2) = state.omega(1);
        omega_vee(2, 0) = -state.omega(1);
        omega_vee(1, 0) = state.omega(2);
        omega_vee(0, 1) = -state.omega(2);
        
        
        state_dot.p = state.v;
        state_dot.R = R * omega_vee;
        state_dot.rotor_angular_rate = rotor_angular_acc;

        // state_dot.v = -Eigen::Vector3d(0, 0, params.g) + (thrust_moment(0) + drag_f) * R.col(2) / params.mass + rotor_drag;
        // state_dot.omega = params.J.inverse() * (moment - state.omega.cross(params.J * state.omega));

        // 平动：加上世界系扰动力
        state_dot.v = -Eigen::Vector3d(0, 0, params.g)
                    + (thrust_moment(0) + drag_f) * R.col(2) / params.mass
                    + rotor_drag
                    + dist_force_world_ / params.mass;
        // 姿态：加上机体系扰动力矩
        state_dot.omega = params.J.inverse() *
                        (moment + dist_moment_body_
                        - state.omega.cross(params.J * state.omega));

        // === 机械臂关节动力学 ===
        // 误差: e_q = q_ref - q, e_dq = dq_ref - dq
        Eigen::Vector3d e_q  = theta_ref_  - state.theta;
        Eigen::Vector3d e_dq = dtheta_ref_ - state.dtheta;

        // 减小控制强度 (如果 params.arm_kp 很大，跟踪就会太准)
        // 这里我们可以人为给控制指令打个折扣，并增加物理摩擦
        double arm_friction_coeff = 0.1; // 关节摩擦系数

        // 简单 PD：tau = Kp * e_q + Kd * e_dq
        Eigen::Vector3d tau_control = params.arm_kp.cwiseProduct(e_q)
                            + params.arm_kd.cwiseProduct(e_dq);

        // 最终力矩 = 控制力矩 - 物理摩擦
        Eigen::Vector3d tau_total = tau_control - arm_friction_coeff * state.dtheta;

        // 4. 动力学方程: qdd = tau / I
        Eigen::Vector3d qdd;
        for (int i = 0; i < 3; ++i)
        {
          // 防止除 0 保护
          double I = std::max(params.arm_inertia[i], 1e-6);
          qdd[i] = tau_total[i] / I;
        }

        state_dot.theta  = state.dtheta;
        state_dot.dtheta = qdd;
        
        return state_dot;
    }

}
