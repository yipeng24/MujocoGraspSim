#include "controller.h"

using namespace std;

template <typename T> inline T clamp_val(const T &v, const T &lo, const T &hi) {
  return (v < lo) ? lo : (v > hi) ? hi : v;
}

Controller::Controller(Parameter_t &param_) : param(param_) {
  int_e_v.setZero();
  Kp(0) = param.gain.Kp0;
  Kp(1) = param.gain.Kp1;
  Kp(2) = param.gain.Kp2;
  Kv(0) = param.gain.Kv0;
  Kv(1) = param.gain.Kv1;
  Kv(2) = param.gain.Kv2;
  Kvi(0) = param.gain.Kvi0;
  Kvi(1) = param.gain.Kvi1;
  Kvi(2) = param.gain.Kvi2;
  Kvd(0) = param.gain.Kvd0;
  Kvd(1) = param.gain.Kvd1;
  Kvd(2) = param.gain.Kvd2;
  KAng(0) = param.gain.KAngR;
  KAng(1) = param.gain.KAngP;
  KAng(2) = param.gain.KAngY;

  resetThrustMapping();
  Gravity = Eigen::Vector3d(0.0, 0.0, -param.gra);
}

/************* Algorithm0 from the Zhepei Wang, start ***************/

quadrotor_msgs::Px4ctrlDebug Controller::update_alg0(const Desired_State_t &des,
                                                     const Odom_Data_t &odom,
                                                     const Imu_Data_t &imu,
                                                     Controller_Output_t &u,
                                                     double voltage) {
  // Check the given velocity is valid.
  if (des.v(2) < -3.0)
    ROS_WARN("[px4ctrl] Desired z-Velocity = %6.3fm/s, < -3.0m/s, which is "
             "dangerous since the drone will be unstable!",
             des.v(2));

  // Compute desired control commands
  const Eigen::Vector3d pid_error_accelerations =
      computePIDErrorAcc(odom, des, param);
  Eigen::Vector3d translational_acc = pid_error_accelerations + des.a;
  Eigen::Quaterniond desired_attitude, idel_att;
  Eigen::Vector3d omega;
  double thrust, debug_thrust;
  translational_acc = (Gravity + computeLimitedTotalAccFromThrustForce(
                                     translational_acc - Gravity, 1.0))
                          .eval();

  // wmywmy
  minimumSingularityFlatWithDrag(param.mass, param.gra, des.v, des.a, des.j,
                                 des.yaw, des.yaw_rate, odom.q, idel_att, omega,
                                 debug_thrust);
  // wmywmy

  minimumSingularityFlatWithDrag(
      param.mass, param.gra, des.v, translational_acc, des.j, des.yaw,
      des.yaw_rate, odom.q, desired_attitude, u.bodyrates, thrust);

  Eigen::Vector3d thrustforce =
      desired_attitude * (thrust * Eigen::Vector3d::UnitZ());
  Eigen::Vector3d total_des_acc =
      computeLimitedTotalAccFromThrustForce(thrustforce, param.mass);

  u.thrust = computeDesiredCollectiveThrustSignal(odom.q, odom.v, total_des_acc,
                                                  param, voltage);

  const Eigen::Vector3d feedback_bodyrates =
      computeFeedBackControlBodyrates(desired_attitude, odom.q, param);

  // Compute the error quaternion wmywmy
  const Eigen::Quaterniond q_e = idel_att.inverse() * desired_attitude;

  Eigen::AngleAxisd rotation_vector(q_e); // debug
  Eigen::Vector3d axis = rotation_vector.axis();
  debug.fb_axisang_x = axis(0);
  debug.fb_axisang_y = axis(1);
  debug.fb_axisang_z = axis(2);
  debug.fb_axisang_ang = rotation_vector.angle();
  // wmywmy

  debug.fb_a_x = pid_error_accelerations(0); // debug
  debug.fb_a_y = pid_error_accelerations(1);
  debug.fb_a_z = pid_error_accelerations(2);
  debug.des_a_x = total_des_acc(0);
  debug.des_a_y = total_des_acc(1);
  debug.des_a_z = total_des_acc(2);
  debug.des_q_w = desired_attitude.w(); // debug
  debug.des_q_x = desired_attitude.x();
  debug.des_q_y = desired_attitude.y();
  debug.des_q_z = desired_attitude.z();

  u.q = imu.q * odom.q.inverse() * desired_attitude; // Align with FCU frame
  const Eigen::Vector3d bodyrate_candidate = u.bodyrates + feedback_bodyrates;

  // limit the angular acceleration
  u.bodyrates = computeLimitedAngularAcc(bodyrate_candidate);
  // u.bodyrates += feedback_bodyrates;

  // Used for thrust-accel mapping estimation
  timed_thrust.push(std::pair<ros::Time, double>(ros::Time::now(), u.thrust));
  while (timed_thrust.size() > 100)
    timed_thrust.pop();

  return debug; // debug
};

Eigen::Vector3d Controller::computeLimitedTotalAccFromThrustForce(
    const Eigen::Vector3d &thrustforce, const double &mass) const {
  Eigen::Vector3d total_acc = thrustforce / mass;

  // Limit magnitude
  if (total_acc.norm() < kMinNormalizedCollectiveAcc_) {
    total_acc = total_acc.normalized() * kMinNormalizedCollectiveAcc_;
  }

  // Limit angle
  if (param.max_angle > 0) {
    double z_acc = total_acc.dot(Eigen::Vector3d::UnitZ());
    Eigen::Vector3d z_B = total_acc.normalized();
    if (z_acc < kMinNormalizedCollectiveAcc_) {
      z_acc = kMinNormalizedCollectiveAcc_; // Not allow too small z-force when
                                            // angle limit is enabled.
    }
    Eigen::Vector3d rot_axis = Eigen::Vector3d::UnitZ().cross(z_B).normalized();
    double rot_ang = std::acos(Eigen::Vector3d::UnitZ().dot(z_B) / (1 * 1));
    if (rot_ang > param.max_angle) // Exceed the angle limit
    {
      Eigen::Vector3d limited_z_B =
          Eigen::AngleAxisd(param.max_angle, rot_axis) *
          Eigen::Vector3d::UnitZ();
      total_acc = z_acc / std::cos(param.max_angle) * limited_z_B;
    }
  }

  return total_acc;
}

bool Controller::flatnessWithDrag(const Eigen::Vector3d &vel,
                                  const Eigen::Vector3d &acc,
                                  const Eigen::Vector3d &jer, const double &psi,
                                  const double &dpsi, double &thr,
                                  Eigen::Vector4d &quat, Eigen::Vector3d &omg,
                                  const double &mass, const double &grav,
                                  const double &dh, const double &dv,
                                  const double &cp, const double &veps) const {
  const double almost_zero = 1.0e-6;

  double w0, w1, w2, dw0, dw1, dw2;
  double v0, v1, v2, a0, a1, a2, v_dot_a;
  double z0, z1, z2, dz0, dz1, dz2;
  double cp_term, w_term, dh_over_m;
  double zu_sqr_norm, zu_norm, zu0, zu1, zu2;
  double zu_sqr0, zu_sqr1, zu_sqr2, zu01, zu12, zu02;
  double ng00, ng01, ng02, ng11, ng12, ng22, ng_den;
  double dw_term, dz_term0, dz_term1, dz_term2, f_term0, f_term1, f_term2;
  double tilt_den, tilt0, tilt1, tilt2, c_half_psi, s_half_psi;
  double c_psi, s_psi, omg_den, omg_term;

  v0 = vel(0);
  v1 = vel(1);
  v2 = vel(2);
  a0 = acc(0);
  a1 = acc(1);
  a2 = acc(2);
  cp_term = sqrt(v0 * v0 + v1 * v1 + v2 * v2 + veps);
  w_term = 1.0 + cp * cp_term;
  w0 = w_term * v0;
  w1 = w_term * v1;
  w2 = w_term * v2;
  dh_over_m = dh / mass;
  zu0 = a0 + dh_over_m * w0;
  zu1 = a1 + dh_over_m * w1;
  zu2 = a2 + dh_over_m * w2 + grav;
  zu_sqr0 = zu0 * zu0;
  zu_sqr1 = zu1 * zu1;
  zu_sqr2 = zu2 * zu2;
  zu01 = zu0 * zu1;
  zu12 = zu1 * zu2;
  zu02 = zu0 * zu2;
  zu_sqr_norm = zu_sqr0 + zu_sqr1 + zu_sqr2;
  zu_norm = sqrt(zu_sqr_norm);
  if (zu_norm < almost_zero) {
    return false;
  }
  z0 = zu0 / zu_norm;
  z1 = zu1 / zu_norm;
  z2 = zu2 / zu_norm;
  ng_den = zu_sqr_norm * zu_norm;
  ng00 = (zu_sqr1 + zu_sqr2) / ng_den;
  ng01 = -zu01 / ng_den;
  ng02 = -zu02 / ng_den;
  ng11 = (zu_sqr0 + zu_sqr2) / ng_den;
  ng12 = -zu12 / ng_den;
  ng22 = (zu_sqr0 + zu_sqr1) / ng_den;
  v_dot_a = v0 * a0 + v1 * a1 + v2 * a2;
  dw_term = cp * v_dot_a / cp_term;
  dw0 = w_term * a0 + dw_term * v0;
  dw1 = w_term * a1 + dw_term * v1;
  dw2 = w_term * a2 + dw_term * v2;
  dz_term0 = jer(0) + dh_over_m * dw0;
  dz_term1 = jer(1) + dh_over_m * dw1;
  dz_term2 = jer(2) + dh_over_m * dw2;
  dz0 = ng00 * dz_term0 + ng01 * dz_term1 + ng02 * dz_term2;
  dz1 = ng01 * dz_term0 + ng11 * dz_term1 + ng12 * dz_term2;
  dz2 = ng02 * dz_term0 + ng12 * dz_term1 + ng22 * dz_term2;
  f_term0 = mass * a0 + dv * w0;
  f_term1 = mass * a1 + dv * w1;
  f_term2 = mass * (a2 + grav) + dv * w2;
  thr = z0 * f_term0 + z1 * f_term1 + z2 * f_term2;
  if (1.0 + z2 < almost_zero) {
    return false;
  }
  tilt_den = sqrt(2.0 * (1.0 + z2));
  tilt0 = 0.5 * tilt_den;
  tilt1 = -z1 / tilt_den;
  tilt2 = z0 / tilt_den;
  c_half_psi = cos(0.5 * psi);
  s_half_psi = sin(0.5 * psi);
  quat(0) = tilt0 * c_half_psi;
  quat(1) = tilt1 * c_half_psi + tilt2 * s_half_psi;
  quat(2) = tilt2 * c_half_psi - tilt1 * s_half_psi;
  quat(3) = tilt0 * s_half_psi;
  c_psi = cos(psi);
  s_psi = sin(psi);
  omg_den = z2 + 1.0;
  omg_term = dz2 / omg_den;
  omg(0) = dz0 * s_psi - dz1 * c_psi - (z0 * s_psi - z1 * c_psi) * omg_term;
  omg(1) = dz0 * c_psi + dz1 * s_psi - (z0 * c_psi + z1 * s_psi) * omg_term;
  omg(2) = (z1 * dz0 - z0 * dz1) / omg_den + dpsi;

  return true;
}

// grav is the gravitional acceleration
// the coordinate should have upward z-axis
void Controller::minimumSingularityFlatWithDrag(
    const double mass, const double grav, const Eigen::Vector3d &vel,
    const Eigen::Vector3d &acc, const Eigen::Vector3d &jer, const double &yaw,
    const double &yawd, const Eigen::Quaterniond &att_est,
    Eigen::Quaterniond &att, Eigen::Vector3d &omg, double &thrust) const {

  // Drag effect parameters (Drag may cause larger tracking error in aggressive
  // flight during our tests) dv >= dh is required dv is the rotor drag effect
  // in vertical direction, typical value is 0.35 dh is the rotor drag effect in
  // horizontal direction, typical value is 0.25 cp is the second-order drag
  // effect, typical valye is 0.01 const double dh = 0.10; const double dv =
  // 0.23; const double cp = 0.01;
  const double dh = 0.00;
  const double dv = 0.00;
  const double cp = 0.00;

  // veps is a smnoothing constant, do not change it
  const double veps = 0.02; // ms^-s

  static Eigen::Vector3d omg_old(0.0, 0.0, 0.0);
  static double thrust_old =
      mass * (acc + grav * Eigen::Vector3d::UnitZ()).norm();

  Eigen::Vector4d quat;
  if (flatnessWithDrag(vel, acc, jer, yaw, yawd, thrust, quat, omg, mass, grav,
                       dh, dv, cp, veps)) {
    att = Eigen::Quaterniond(quat(0), quat(1), quat(2), quat(3));
    omg_old = omg;
    thrust_old = thrust;
  } else {
    ROS_WARN("Conor case: 1. Eactly inverted flight or 2. Unactuated falling");
    att = att_est;
    omg = omg_old;
    thrust = thrust_old;
  }

  return;
}
/************* Algorithm0 from Zhepei Wang, end ***************/

/************* Algorithm1 from the Zhepei Wang, start ***************/
quadrotor_msgs::Px4ctrlDebug Controller::update_alg1(const Desired_State_t &des,
                                                     const Odom_Data_t &odom,
                                                     const Imu_Data_t &imu,
                                                     Controller_Output_t &u,
                                                     double voltage) {
  // Check the given velocity is valid.
  if (des.v(2) < -3.0)
    ROS_WARN("[px4ctrl] Desired z-Velocity = %6.3fm/s, < -3.0m/s, which is "
             "dangerous since the drone will be unstable!",
             des.v(2));

  // Compute desired control commands
  const Eigen::Vector3d pid_error_accelerations =
      computePIDErrorAcc(odom, des, param);
  Eigen::Vector3d total_des_acc =
      computeLimitedTotalAcc(pid_error_accelerations, des.a);

  debug.fb_a_x = pid_error_accelerations(0); // debug
  debug.fb_a_y = pid_error_accelerations(1);
  debug.fb_a_z = pid_error_accelerations(2);
  debug.des_a_x = total_des_acc(0);
  debug.des_a_y = total_des_acc(1);
  debug.des_a_z = total_des_acc(2);

  u.thrust = computeDesiredCollectiveThrustSignal(odom.q, odom.v, total_des_acc,
                                                  param, voltage);

  Eigen::Quaterniond desired_attitude;
  computeFlatInput(total_des_acc, des.j, des.yaw, des.yaw_rate, odom.q,
                   desired_attitude, u.bodyrates);
  const Eigen::Vector3d feedback_bodyrates =
      computeFeedBackControlBodyrates(desired_attitude, odom.q, param);

  debug.des_q_w = desired_attitude.w(); // debug
  debug.des_q_x = desired_attitude.x();
  debug.des_q_y = desired_attitude.y();
  debug.des_q_z = desired_attitude.z();

  u.q = imu.q * odom.q.inverse() * desired_attitude; // Align with FCU frame
  u.bodyrates += feedback_bodyrates;

  // Used for thrust-accel mapping estimation
  timed_thrust.push(std::pair<ros::Time, double>(ros::Time::now(), u.thrust));
  while (timed_thrust.size() > 100)
    timed_thrust.pop();

  return debug; // debug
};

void Controller::normalizeWithGrad(const Eigen::Vector3d &x,
                                   const Eigen::Vector3d &xd,
                                   Eigen::Vector3d &xNor,
                                   Eigen::Vector3d &xNord) const {
  const double xSqrNorm = x.squaredNorm();
  const double xNorm = sqrt(xSqrNorm);
  xNor = x / xNorm;
  xNord = (xd - x * (x.dot(xd) / xSqrNorm)) / xNorm;
  return;
}

// grav is the gravitional acceleration
// the coordinate should have upward z-axis
void Controller::computeFlatInput(const Eigen::Vector3d &thr_acc,
                                  const Eigen::Vector3d &jer, const double &yaw,
                                  const double &yawd,
                                  const Eigen::Quaterniond &att_est,
                                  Eigen::Quaterniond &att,
                                  Eigen::Vector3d &omg) const {
  static Eigen::Vector3d omg_old(0.0, 0.0, 0.0);

  if (thr_acc.norm() < kMinNormalizedCollectiveAcc_) {
    att = att_est;
    omg.setConstant(0.0);
    // ROS_WARN("Conor case, thrust is too small, thr_acc.norm()=%f",
    // thr_acc.norm());
    return;
  } else {
    Eigen::Vector3d zb, zbd;
    normalizeWithGrad(thr_acc, jer, zb, zbd);
    double syaw = sin(yaw);
    double cyaw = cos(yaw);
    Eigen::Vector3d xc(cyaw, syaw, 0.0);
    Eigen::Vector3d xcd(-syaw * yawd, cyaw * yawd, 0.0);
    Eigen::Vector3d yc = zb.cross(xc);
    if (yc.norm() < kAlmostZeroValueThreshold_) {
      ROS_WARN("Conor case, pitch is close to 90 deg");
      att = att_est;
      omg = omg_old;
    } else {
      Eigen::Vector3d ycd = zbd.cross(xc) + zb.cross(xcd);
      Eigen::Vector3d yb, ybd;
      normalizeWithGrad(yc, ycd, yb, ybd);
      Eigen::Vector3d xb = yb.cross(zb);
      Eigen::Vector3d xbd = ybd.cross(zb) + yb.cross(zbd);
      omg(0) = (zb.dot(ybd) - yb.dot(zbd)) / 2.0;
      omg(1) = (xb.dot(zbd) - zb.dot(xbd)) / 2.0;
      omg(2) = (yb.dot(xbd) - xb.dot(ybd)) / 2.0;
      Eigen::Matrix3d rotM;
      rotM << xb, yb, zb;
      att = Eigen::Quaterniond(rotM);
      omg_old = omg;
    }
  }
  return;
}
/************* Algorithm1 from Zhepei Wang, end ***************/

/************* Algorithm from the rotor-drag paper, start ***************/
void Controller::update_alg2(const Desired_State_t &des,
                             const Odom_Data_t &odom, const Imu_Data_t &imu,
                             Controller_Output_t &u, double voltage) {
  // Check the given velocity is valid.
  if (des.v(2) < -3.0)
    ROS_WARN("[px4ctrl] Desired z-Velocity = %6.3fm/s, < -3.0m/s, which is "
             "dangerous since the drone will be unstable!",
             des.v(2));

  // Compute reference inputs that compensate for aerodynamic drag
  Eigen::Vector3d drag_acc = Eigen::Vector3d::Zero();
  computeAeroCompensatedReferenceInputs(des, odom, param, &u, &drag_acc);

  // Compute desired control commands
  const Eigen::Vector3d pid_error_accelerations =
      computePIDErrorAcc(odom, des, param);
  Eigen::Vector3d total_des_acc =
      computeLimitedTotalAcc(pid_error_accelerations, des.a, drag_acc);

  u.thrust = computeDesiredCollectiveThrustSignal(odom.q, odom.v, total_des_acc,
                                                  param, voltage);

  const Eigen::Quaterniond desired_attitude =
      computeDesiredAttitude(total_des_acc, des.yaw, odom.q);
  const Eigen::Vector3d feedback_bodyrates =
      computeFeedBackControlBodyrates(desired_attitude, odom.q, param);

  if (param.use_bodyrate_ctrl) {
    u.bodyrates += feedback_bodyrates;
    // u.bodyrates = imu.q * odom.q.inverse() * u.bodyrates;  // Align with FCU
    // frame
  } else {
    u.q = imu.q * odom.q.inverse() * desired_attitude; // Align with FCU frame
  }

  // Used for thrust-accel mapping estimation
  timed_thrust.push(std::pair<ros::Time, double>(ros::Time::now(), u.thrust));
  while (timed_thrust.size() > 100)
    timed_thrust.pop();
};

void Controller::computeAeroCompensatedReferenceInputs(
    const Desired_State_t &des, const Odom_Data_t &odom,
    const Parameter_t &param, Controller_Output_t *outputs,
    Eigen::Vector3d *drag_acc) const {

  const double dx = param.rt_drag.x;
  const double dy = param.rt_drag.y;
  const double dz = param.rt_drag.z;

  const Eigen::Quaterniond q_heading =
      Eigen::Quaterniond(Eigen::AngleAxisd(des.yaw, Eigen::Vector3d::UnitZ()));

  const Eigen::Vector3d x_C = q_heading * Eigen::Vector3d::UnitX();
  const Eigen::Vector3d y_C = q_heading * Eigen::Vector3d::UnitY();

  const Eigen::Vector3d alpha = des.a - Gravity + dx * des.v;
  const Eigen::Vector3d beta = des.a - Gravity + dy * des.v;
  const Eigen::Vector3d gamma = des.a - Gravity + dz * des.v;

  // Reference attitude
  const Eigen::Vector3d x_B_prototype = y_C.cross(alpha);
  const Eigen::Vector3d x_B =
      computeRobustBodyXAxis(x_B_prototype, x_C, y_C, odom.q);

  Eigen::Vector3d y_B = beta.cross(x_B);
  if (almostZero(y_B.norm())) {
    const Eigen::Vector3d z_B_estimated = odom.q * Eigen::Vector3d::UnitZ();
    y_B = z_B_estimated.cross(x_B);
    if (almostZero(y_B.norm())) {
      y_B = y_C;
    } else {
      y_B.normalize();
    }
  } else {
    y_B.normalize();
  }

  const Eigen::Vector3d z_B = x_B.cross(y_B);

  const Eigen::Matrix3d R_W_B_ref(
      (Eigen::Matrix3d() << x_B, y_B, z_B).finished());

  outputs->q = Eigen::Quaterniond(R_W_B_ref);

  // Reference thrust
  outputs->thrust = z_B.dot(gamma);

  // Rotor drag matrix
  const Eigen::Matrix3d D = Eigen::Vector3d(dx, dy, dz).asDiagonal();

  // Reference body rates
  const double B1 = outputs->thrust - (dz - dx) * z_B.dot(des.v);
  const double C1 = -(dx - dy) * y_B.dot(des.v);
  const double D1 = x_B.dot(des.j) + dx * x_B.dot(des.a);
  const double A2 = outputs->thrust + (dy - dz) * z_B.dot(des.v);
  const double C2 = (dx - dy) * x_B.dot(des.v);
  const double D2 = -y_B.dot(des.j) - dy * y_B.dot(des.a);
  const double B3 = -y_C.dot(z_B);
  const double C3 = (y_C.cross(z_B)).norm();
  const double D3 = des.yaw_rate * x_C.dot(x_B);

  const double denominator = B1 * C3 - B3 * C1;

  if (almostZero(denominator)) {
    outputs->bodyrates = Eigen::Vector3d::Zero();
  } else {
    // Compute body rates
    if (almostZero(A2)) {
      outputs->bodyrates.x() = 0.0;
    } else {
      outputs->bodyrates.x() =
          (-B1 * C2 * D3 + B1 * C3 * D2 - B3 * C1 * D2 + B3 * C2 * D1) /
          (A2 * denominator);
    }
    outputs->bodyrates.y() = (-C1 * D3 + C3 * D1) / denominator;
    outputs->bodyrates.z() = (B1 * D3 - B3 * D1) / denominator;
  }

  // Transform reference rates and derivatives into estimated body frame
  const Eigen::Matrix3d R_trans =
      odom.q.toRotationMatrix().transpose() * R_W_B_ref;
  const Eigen::Vector3d bodyrates_ref = outputs->bodyrates;

  outputs->bodyrates = R_trans * bodyrates_ref;

  // Drag accelerations
  *drag_acc = -1.0 * (R_W_B_ref * (D * (R_W_B_ref.transpose() * des.v)));
}

Eigen::Quaterniond
Controller::computeDesiredAttitude(const Eigen::Vector3d &des_acc,
                                   const double reference_heading,
                                   const Eigen::Quaterniond &est_q) const {
  const Eigen::Quaterniond q_heading = Eigen::Quaterniond(
      Eigen::AngleAxisd(reference_heading, Eigen::Vector3d::UnitZ()));

  // Compute desired orientation
  const Eigen::Vector3d x_C = q_heading * Eigen::Vector3d::UnitX();
  const Eigen::Vector3d y_C = q_heading * Eigen::Vector3d::UnitY();

  // Eigen::Vector3d des_acc2(-1.0,-1.0,-1.0);

  Eigen::Vector3d z_B;
  if (almostZero(des_acc.norm())) {
    // In case of free fall we keep the thrust direction to be the estimated one
    // This only works assuming that we are in this condition for a very short
    // time (otherwise attitude drifts)
    z_B = est_q * Eigen::Vector3d::UnitZ();
  } else {
    z_B = des_acc.normalized();
  }

  const Eigen::Vector3d x_B_prototype = y_C.cross(z_B);
  const Eigen::Vector3d x_B =
      computeRobustBodyXAxis(x_B_prototype, x_C, y_C, est_q);

  const Eigen::Vector3d y_B = (z_B.cross(x_B)).normalized();

  // From the computed desired body axes we can now compose a desired attitude
  const Eigen::Matrix3d R_W_B((Eigen::Matrix3d() << x_B, y_B, z_B).finished());

  const Eigen::Quaterniond desired_attitude(R_W_B);

  return desired_attitude;
}

Eigen::Vector3d Controller::computeRobustBodyXAxis(
    const Eigen::Vector3d &x_B_prototype, const Eigen::Vector3d &x_C,
    const Eigen::Vector3d &y_C, const Eigen::Quaterniond &est_q) const {
  Eigen::Vector3d x_B = x_B_prototype;

  // cout << "x_B.norm()=" << x_B.norm() << endl;

  if (almostZero(x_B.norm())) {
    // if cross(y_C, z_B) == 0, they are collinear =>
    // every x_B lies automatically in the x_C - z_C plane

    // Project estimated body x-axis into the x_C - z_C plane
    const Eigen::Vector3d x_B_estimated = est_q * Eigen::Vector3d::UnitX();
    const Eigen::Vector3d x_B_projected =
        x_B_estimated - (x_B_estimated.dot(y_C)) * y_C;
    if (almostZero(x_B_projected.norm())) {
      // Not too much intelligent stuff we can do in this case but it should
      // basically never occur
      x_B = x_C;
    } else {
      x_B = x_B_projected.normalized();
    }
  } else {
    x_B.normalize();
  }

  // if the quad is upside down, x_B will point in the "opposite" direction
  // of x_C => flip x_B (unfortunately also not the solution for our problems)
  if (x_B.dot(x_C) < 0.0) {
    x_B = -x_B;
    // std::cout << "CCCCCCCCCCCCCC" << std::endl;
  }

  return x_B;
}
/************* Algorithm from the rotor-drag paper, end ***************/

//* jr revised 加入 theta_cmd
/************* Algorithm3 from Algorithm1 and import joint_state_t, start
 * ***************/
quadrotor_msgs::Px4ctrlDebug
Controller::update_alg3(const Desired_State_t &des, const Odom_Data_t &odom,
                        const Joint_State_Data_t &js, const Imu_Data_t &imu,
                        const bool &in_cmd, Controller_Output_t &u,
                        double voltage) {
  // Check the given velocity is valid.
  if (des.v(2) < -3.0)
    ROS_WARN("[px4ctrl] Desired z-Velocity = %6.3fm/s, < -3.0m/s, which is "
             "dangerous since the drone will be unstable!",
             des.v(2));

  // Compute desired control commands
  const Eigen::Vector3d pid_error_accelerations =
      computePIDErrorAcc(odom, des, param);
  Eigen::Vector3d total_des_acc =
      computeLimitedTotalAcc(pid_error_accelerations, des.a);

  debug.fb_a_x = pid_error_accelerations(0); // debug
  debug.fb_a_y = pid_error_accelerations(1);
  debug.fb_a_z = pid_error_accelerations(2);
  debug.des_a_x = total_des_acc(0);
  debug.des_a_y = total_des_acc(1);
  debug.des_a_z = total_des_acc(2);

  //! ===== 2) L1 自适应扰动估计（世界坐标系）=====
  if (param.l1.enable) {
    applyL1Compensation(total_des_acc, odom, imu);
  }

  u.thrust = computeDesiredCollectiveThrustSignal(odom.q, odom.v, total_des_acc,
                                                  param, voltage);

  Eigen::Vector3d total_ref_acc =
      computeLimitedTotalAcc(Eigen::Vector3d::Zero(), des.a);
  Eigen::Quaterniond ref_attitude;
  computeFlatInput(total_ref_acc, des.j, des.yaw, des.yaw_rate, odom.q,
                   ref_attitude, u.bodyrates);

  Eigen::Quaterniond desired_attitude;
  computeFlatInput(total_des_acc, des.j, des.yaw, des.yaw_rate, odom.q,
                   desired_attitude, u.bodyrates);
  const Eigen::Vector3d feedback_bodyrates =
      computeFeedBackControlBodyrates(desired_attitude, odom.q, param);

  // debug
  debug.des_q_w = desired_attitude.w();
  debug.des_q_x = desired_attitude.x();
  debug.des_q_y = desired_attitude.y();
  debug.des_q_z = desired_attitude.z();

  Eigen::Vector3d e_pos;
  Eigen::Quaterniond e_quat;
  Eigen::Matrix4d T_e2b;
  rc_sdf_ptr->kine_ptr_->getEndPose(des.p, ref_attitude, des.theta, e_pos,
                                    e_quat, T_e2b);
  nav_msgs::Odometry end_pose_msg;
  end_pose_msg.header.stamp = ros::Time::now();
  end_pose_msg.header.frame_id = "world";
  end_pose_msg.pose.pose.position.x = e_pos(0);
  end_pose_msg.pose.pose.position.y = e_pos(1);
  end_pose_msg.pose.pose.position.z = e_pos(2);
  end_pose_msg.pose.pose.orientation.w = e_quat.w();
  end_pose_msg.pose.pose.orientation.x = e_quat.x();
  end_pose_msg.pose.pose.orientation.y = e_quat.y();
  end_pose_msg.pose.pose.orientation.z = e_quat.z();
  debug.end_pose_ref = end_pose_msg;

  u.q = imu.q * odom.q.inverse() * desired_attitude; // Align with FCU frame
  u.bodyrates += feedback_bodyrates;

  // TODO 舵机 PID 拿到 u.dtheta
  // 误差: e_q = q_ref - q, e_dq = dq_ref - dq
  // Eigen::Vector3d e_theta  = theta_ref_  - state.theta;
  // Eigen::Vector3d e_dq = dtheta_ref_ - state.dtheta;

  // 简单 PD：tau = Kp * e_q + Kd * e_dq
  // Eigen::Vector3d tau = params.arm_kp.cwiseProduct(e_q)
  //                     + params.arm_kd.cwiseProduct(e_dq);

  //* airgrasp
  if (in_cmd && param.arm_comp.enable) {
    // 1. 计算原始补偿增量
    Eigen::Vector3d wp_vec(param.arm_comp.wp_x, param.arm_comp.wp_y,
                           param.arm_comp.wp_z);
    Eigen::Vector3d wr_vec(param.arm_comp.wr_x, param.arm_comp.wr_y,
                           param.arm_comp.wr_z);

    Eigen::VectorXd delta_theta = calculateArmCompensation(
        des.p, desired_attitude, des.theta, odom.p, odom.q, wp_vec, wr_vec);

    // 2. 得到原始期望角度
    Eigen::VectorXd raw_theta = des.theta + delta_theta;
    clipTheta(raw_theta, param); // 先限幅

    // 3. 一阶低通滤波 (Exponential Smoothing)
    // 滤波公式: y(k) = alpha * x(k) + (1 - alpha) * y(k-1)
    double alpha = param.arm_comp.alpha; // 从配置文件获取滤波系数

    if (!theta_filter_.inited) {
      theta_filter_.last_theta = raw_theta;
      theta_filter_.inited = true;
    }

    u.theta = alpha * raw_theta + (1.0 - alpha) * theta_filter_.last_theta;
    theta_filter_.last_theta = u.theta; // 更新状态

    // Debug 输出
    debug.delta_theta.clear();
    for (int i = 0; i < u.theta.size(); i++)
      debug.delta_theta.push_back(u.theta(i));
  } else {
    u.theta = des.theta;
  }

  // 不补偿
  // u.theta = des.theta;

  // 4. (可选) 如果需要补偿速度，也可以根据 delta_theta/dt 做前馈
  u.dtheta = des.dtheta;

  // Used for thrust-accel mapping estimation
  timed_thrust.push(std::pair<ros::Time, double>(ros::Time::now(), u.thrust));
  while (timed_thrust.size() > 100)
    timed_thrust.pop();

  return debug; // debug
};

Eigen::Vector3d
Controller::computeFeedBackControlBodyrates(const Eigen::Quaterniond &des_q,
                                            const Eigen::Quaterniond &est_q,
                                            const Parameter_t &param) {
  // Compute the error quaternion
  const Eigen::Quaterniond q_e = est_q.inverse() * des_q;

  Eigen::AngleAxisd rotation_vector(q_e); // debug
  Eigen::Vector3d axis = rotation_vector.axis();
  debug.exec_err_axisang_x = axis(0);
  debug.exec_err_axisang_y = axis(1);
  debug.exec_err_axisang_z = axis(2);
  debug.exec_err_axisang_ang = rotation_vector.angle();

  // Compute desired body rates from control error
  Eigen::Vector3d bodyrates;

  if (q_e.w() >= 0) {
    bodyrates.x() = 2.0 * KAng(0) * q_e.x();
    bodyrates.y() = 2.0 * KAng(1) * q_e.y();
    bodyrates.z() = 2.0 * KAng(2) * q_e.z();
  } else {
    bodyrates.x() = -2.0 * KAng(0) * q_e.x();
    bodyrates.y() = -2.0 * KAng(1) * q_e.y();
    bodyrates.z() = -2.0 * KAng(2) * q_e.z();
  }

  if (bodyrates.x() > kMaxBodyratesFeedback_)
    bodyrates.x() = kMaxBodyratesFeedback_;
  if (bodyrates.x() < -kMaxBodyratesFeedback_)
    bodyrates.x() = -kMaxBodyratesFeedback_;
  if (bodyrates.y() > kMaxBodyratesFeedback_)
    bodyrates.y() = kMaxBodyratesFeedback_;
  if (bodyrates.y() < -kMaxBodyratesFeedback_)
    bodyrates.y() = -kMaxBodyratesFeedback_;
  if (bodyrates.z() > kMaxBodyratesFeedback_)
    bodyrates.z() = kMaxBodyratesFeedback_;
  if (bodyrates.z() < -kMaxBodyratesFeedback_)
    bodyrates.z() = -kMaxBodyratesFeedback_;

  // debug
  debug.fb_rate_x = bodyrates.x();
  debug.fb_rate_y = bodyrates.y();
  debug.fb_rate_z = bodyrates.z();

  return bodyrates;
}

Eigen::Vector3d Controller::computePIDErrorAcc(const Odom_Data_t &odom,
                                               const Desired_State_t &des,
                                               const Parameter_t &param) {
  // Compute the desired accelerations due to control errors in world frame
  // with a PID controller
  Eigen::Vector3d acc_error;

  // x acceleration
  double x_pos_error =
      std::isnan(des.p(0))
          ? 0.0
          : std::max(std::min(des.p(0) - odom.p(0), 1.0), -1.0);
  double x_vel_error = std::max(
      std::min((des.v(0) + Kp(0) * x_pos_error) - odom.v(0), 1.0), -1.0);
  acc_error(0) = Kv(0) * x_vel_error;

  // y acceleration
  double y_pos_error =
      std::isnan(des.p(1))
          ? 0.0
          : std::max(std::min(des.p(1) - odom.p(1), 1.0), -1.0);
  double y_vel_error = std::max(
      std::min((des.v(1) + Kp(1) * y_pos_error) - odom.v(1), 1.0), -1.0);
  acc_error(1) = Kv(1) * y_vel_error;

  // z acceleration
  double z_pos_error =
      std::isnan(des.p(2))
          ? 0.0
          : std::max(std::min(des.p(2) - odom.p(2), 1.0), -1.0);
  double z_vel_error = std::max(
      std::min((des.v(2) + Kp(2) * z_pos_error) - odom.v(2), 1.0), -1.0);
  acc_error(2) = Kv(2) * z_vel_error;

  debug.des_v_x = (des.v(0) + Kp(0) * x_pos_error); // debug
  debug.des_v_y = (des.v(1) + Kp(1) * y_pos_error);
  debug.des_v_z = (des.v(2) + Kp(2) * z_pos_error);

  return acc_error;
}

Eigen::Vector3d Controller::computeLimitedTotalAcc(
    const Eigen::Vector3d &PIDErrorAcc, const Eigen::Vector3d &ref_acc,
    const Eigen::Vector3d &drag_acc /*default = Eigen::Vector3d::Zero() */)
    const {
  Eigen::Vector3d total_acc;
  total_acc = PIDErrorAcc + ref_acc - Gravity - drag_acc;

  // Limit magnitude
  if (total_acc.norm() < kMinNormalizedCollectiveAcc_) {
    total_acc = total_acc.normalized() * kMinNormalizedCollectiveAcc_;
  }

  // Limit angle
  if (param.max_angle > 0) {
    double z_acc = total_acc.dot(Eigen::Vector3d::UnitZ());
    Eigen::Vector3d z_B = total_acc.normalized();
    if (z_acc < kMinNormalizedCollectiveAcc_) {
      z_acc = kMinNormalizedCollectiveAcc_; // Not allow too small z-force when
                                            // angle limit is enabled.
    }
    Eigen::Vector3d rot_axis = Eigen::Vector3d::UnitZ().cross(z_B).normalized();
    double rot_ang = std::acos(Eigen::Vector3d::UnitZ().dot(z_B) / (1 * 1));
    if (rot_ang > param.max_angle) // Exceed the angle limit
    {
      Eigen::Vector3d limited_z_B =
          Eigen::AngleAxisd(param.max_angle, rot_axis) *
          Eigen::Vector3d::UnitZ();
      total_acc = z_acc / std::cos(param.max_angle) * limited_z_B;
    }
  }

  return total_acc;
}

Eigen::Vector3d
Controller::computeLimitedAngularAcc(const Eigen::Vector3d candidate_bodyrate) {
  ros::Time t_now = ros::Time::now();
  if (last_ctrl_timestamp_ != ros::Time(0)) {
    double dura = (t_now - last_ctrl_timestamp_).toSec();
    double max_delta_bodyrate = kMaxAngularAcc_ * dura;
    Eigen::Vector3d bodyrate_out;

    if ((candidate_bodyrate(0) - last_bodyrate_(0)) > max_delta_bodyrate) {
      bodyrate_out(0) = last_bodyrate_(0) + max_delta_bodyrate;
    } else if ((candidate_bodyrate(0) - last_bodyrate_(0)) <
               -max_delta_bodyrate) {
      bodyrate_out(0) = last_bodyrate_(0) - max_delta_bodyrate;
    } else {
      bodyrate_out(0) = candidate_bodyrate(0);
    }

    if ((candidate_bodyrate(1) - last_bodyrate_(1)) > max_delta_bodyrate) {
      bodyrate_out(1) = last_bodyrate_(1) + max_delta_bodyrate;
    } else if ((candidate_bodyrate(1) - last_bodyrate_(1)) <
               -max_delta_bodyrate) {
      bodyrate_out(1) = last_bodyrate_(1) - max_delta_bodyrate;
    } else {
      bodyrate_out(1) = candidate_bodyrate(1);
    }

    if ((candidate_bodyrate(2) - last_bodyrate_(2)) > max_delta_bodyrate) {
      bodyrate_out(2) = last_bodyrate_(2) + max_delta_bodyrate;
    } else if ((candidate_bodyrate(2) - last_bodyrate_(2)) <
               -max_delta_bodyrate) {
      bodyrate_out(2) = last_bodyrate_(2) - max_delta_bodyrate;
    } else {
      bodyrate_out(2) = candidate_bodyrate(2);
    }

    last_ctrl_timestamp_ = t_now;
    last_bodyrate_ = bodyrate_out;

    return bodyrate_out;
  } else {
    last_ctrl_timestamp_ = t_now;
    last_bodyrate_ = candidate_bodyrate;
    return candidate_bodyrate;
  }
}

double Controller::computeDesiredCollectiveThrustSignal(
    const Eigen::Quaterniond &est_q, const Eigen::Vector3d &est_v,
    const Eigen::Vector3d &des_acc, const Parameter_t &param, double voltage) {

  double normalized_thrust;
  const Eigen::Vector3d body_z_axis = est_q * Eigen::Vector3d::UnitZ();
  double des_acc_norm = des_acc.dot(body_z_axis);
  // double des_acc_norm = des_acc.norm();
  if (des_acc_norm < kMinNormalizedCollectiveAcc_) {
    des_acc_norm = kMinNormalizedCollectiveAcc_;
  }

  // This compensates for an acceleration component in thrust direction due
  // to the square of the body-horizontal velocity.
  des_acc_norm -=
      param.rt_drag.k_thrust_horz * (pow(est_v.x(), 2.0) + pow(est_v.y(), 2.0));

  debug.des_thr = des_acc_norm; // debug

  if (param.thr_map.accurate_thrust_model) {
    normalized_thrust = thr_scale_compensate *
                        AccurateThrustAccMapping(des_acc_norm, voltage, param);
  } else {
    normalized_thrust = des_acc_norm / thr2acc;
  }

  return normalized_thrust;
}

double Controller::AccurateThrustAccMapping(const double des_acc_z,
                                            double voltage,
                                            const Parameter_t &param) const {
  if (voltage < param.low_voltage) {
    voltage = param.low_voltage;
    ROS_ERROR("Low voltage!");
  }
  if (voltage > 1.5 * param.low_voltage) {
    voltage = 1.5 * param.low_voltage;
  }

  // F=K1*Voltage^K2*(K3*u^2+(1-K3)*u)
  double a = param.thr_map.K3;
  double b = 1 - param.thr_map.K3;
  double c = -(param.mass * des_acc_z) /
             (param.thr_map.K1 * pow(voltage, param.thr_map.K2));
  double b2_4ac = pow(b, 2) - 4 * a * c;
  if (b2_4ac <= 0)
    b2_4ac = 0;
  double thrust = (-b + sqrt(b2_4ac)) / (2 * a);
  // if (thrust <= 0) thrust = 0; // This should be avoided before calling this
  // function
  return thrust;
}

bool Controller::almostZero(const double value) const {
  return fabs(value) < kAlmostZeroValueThreshold_;
}

bool Controller::almostZeroThrust(const double thrust_value) const {
  return fabs(thrust_value) < kAlmostZeroThrustThreshold_;
}

bool Controller::estimateThrustModel(const Eigen::Vector3d &est_a,
                                     const double voltage,
                                     const Eigen::Vector3d &est_v,
                                     const Parameter_t &param) {

  ros::Time t_now = ros::Time::now();
  while (timed_thrust.size() >= 1) {
    // Choose data before 35~45ms ago
    std::pair<ros::Time, double> t_t = timed_thrust.front();
    double time_passed = (t_now - t_t.first).toSec();
    if (time_passed > 0.045) // 45ms
    {
      // printf("continue, time_passed=%f\n", time_passed);
      timed_thrust.pop();
      continue;
    }
    if (time_passed < 0.035) // 35ms
    {
      // printf("skip, time_passed=%f\n", time_passed);
      return false;
    }

    /***********************************************************/
    /* Recursive least squares algorithm with vanishing memory */
    /***********************************************************/
    double thr = t_t.second;
    timed_thrust.pop();
    if (param.thr_map.accurate_thrust_model) {
      /**************************************************************************/
      /* Model: thr = thr_scale_compensate * AccurateThrustAccMapping(est_a(2))
       */
      /**************************************************************************/
      double thr_fb = AccurateThrustAccMapping(est_a(2), voltage, param);
      double gamma = 1 / (rho2 + thr_fb * P * thr_fb);
      double K = gamma * P * thr_fb;
      thr_scale_compensate =
          thr_scale_compensate + K * (thr - thr_fb * thr_scale_compensate);
      P = (1 - K * thr_fb) * P / rho2;
      // printf("%6.3f,%6.3f,%6.3f,%6.3f\n", thr_scale_compensate, gamma, K, P);
      // fflush(stdout);

      if (thr_scale_compensate > 1.15 || thr_scale_compensate < 0.85) {
        ROS_ERROR(
            "Thrust scale = %f, which shoule around 1. It means the thrust model is nolonger accurate. \
                  Re-calibrate the thrust model!",
            thr_scale_compensate);
        thr_scale_compensate =
            thr_scale_compensate > 1.15 ? 1.15 : thr_scale_compensate;
        thr_scale_compensate =
            thr_scale_compensate < 0.85 ? 0.85 : thr_scale_compensate;
      }

      debug.thr_scale_compensate = thr_scale_compensate; // debug
      debug.voltage = voltage;
      if (param.thr_map.print_val) {
        ROS_WARN("thr_scale_compensate = %f", thr_scale_compensate);
      }
    } else {
      /***********************************/
      /* Model: est_a(2) = thr2acc * thr */
      /***********************************/
      if (!param.thr_map.noisy_imu) {
        double gamma = 1 / (rho2 + thr * P * thr);
        double K = gamma * P * thr;
        thr2acc = thr2acc + K * (est_a(2) - thr * thr2acc);
        P = (1 - K * thr) * P / rho2;
        // printf("%6.3f,%6.3f,%6.3f,%6.3f\n", thr2acc, gamma, K, P);
        // fflush(stdout);
      } else // Strongly not recommended to use!!!
      {
        double K = 10 / param.ctrl_freq_max; // thr2acc changes 10 every second
                                             // when est_v(2) - des_v(2) = 1 m/s
        thr2acc = thr2acc + K * (est_v(2) - debug.des_v_z);
        // printf("%6.3f,%6.3f,%6.3f,%6.3f\n", thr2acc, K, est_v(2),
        // debug.des_v_z); fflush(stdout);
      }
      const double hover_percentage = param.gra / thr2acc;
      if (hover_percentage > 0.8 || hover_percentage < 0.1) {
        ROS_ERROR("Estimated hover_percentage >0.8 or <0.1! Perhaps the accel "
                  "vibration is too high!");
        thr2acc = hover_percentage > 0.8 ? param.gra / 0.8 : thr2acc;
        thr2acc = hover_percentage < 0.1 ? param.gra / 0.1 : thr2acc;
      }
      debug.hover_percentage = hover_percentage; // debug
      if (param.thr_map.print_val) {
        ROS_WARN("hover_percentage = %f", debug.hover_percentage);
      }
    }

    return true;
  }

  return false;
}

void Controller::resetThrustMapping(void) {
  thr2acc = param.gra / param.thr_map.hover_percentage;
  thr_scale_compensate = 1.0;
  P = 1e6;
}

Eigen::VectorXd Controller::calculateArmCompensation(
    const Eigen::Vector3d &des_p, const Eigen::Quaterniond &des_q,
    const Eigen::VectorXd &des_theta, const Eigen::Vector3d &real_p,
    const Eigen::Quaterniond &real_q, const Eigen::Vector3d &w_p,
    const Eigen::Vector3d &w_r) {

  auto kine = rc_sdf_ptr->kine_ptr_;
  int n = des_theta.size();
  const double damping = 1e-4;
  const double max_delta = 0.3;

  Eigen::Vector3d p_t, p_c;
  Eigen::Quaterniond q_t, q_c;
  Eigen::Matrix4d T_tmp;
  kine->getEndPose(des_p, des_q, des_theta, p_t, q_t, T_tmp);
  kine->getEndPose(real_p, real_q, des_theta, p_c, q_c, T_tmp);

  // 1. 调试：世界系误差
  Eigen::Vector3d world_pos_err = p_t - p_c;

  Eigen::VectorXd e_ee(6);
  e_ee.head<3>() = q_t.inverse() * world_pos_err;

  Eigen::Quaterniond q_err = q_c.inverse() * q_t;
  Eigen::Vector3d rot_vec = 2.0 * q_err.vec();
  if (q_err.w() < 0)
    rot_vec = -rot_vec;
  e_ee.tail<3>() = rot_vec;

  // 2. 调试：计算雅可比并观察其范数
  Eigen::MatrixXd J_ee = Eigen::MatrixXd::Zero(6, n);
  Eigen::Matrix3d R_t_inv = q_t.toRotationMatrix().transpose();
  Eigen::MatrixXd J_full =
      kine->getEndPosJacobianFull(real_p, real_q, des_theta);
  J_ee.topRows<3>() = R_t_inv * J_full.block(0, 7, 3, n);

  Eigen::Matrix3d R_drone = real_q.toRotationMatrix();
  auto links = kine->getLinks();
  for (int i = 0; i < n; ++i) {
    Eigen::Matrix4d T_prev_0;
    kine->getRelativeTransform(des_theta, i, 0, T_prev_0);
    Eigen::Vector3d axis_local = Eigen::Vector3d::Zero();
    axis_local(links[i + 1].ang_id) = 1.0;
    J_ee.block<3, 1>(3, i) =
        R_t_inv * R_drone * T_prev_0.block<3, 3>(0, 0) * axis_local;
  }

  Eigen::VectorXd w_combined(6);
  w_combined << w_p, w_r;
  Eigen::DiagonalMatrix<double, 6> W(w_combined);

  Eigen::MatrixXd Jw = W * J_ee;
  Eigen::VectorXd ew = W * e_ee;

  // 3. 求解
  Eigen::MatrixXd A =
      Jw.transpose() * Jw + damping * Eigen::MatrixXd::Identity(n, n);
  Eigen::VectorXd b = Jw.transpose() * ew;
  Eigen::VectorXd delta_theta = A.ldlt().solve(b);

  // === 核心调试输出 ===
  // ROS_INFO_STREAM_THROTTLE(0.5, "--- Arm Compensation Debug ---");
  // ROS_INFO_STREAM_THROTTLE(0.5, "World Pos Err Norm: " <<
  // world_pos_err.norm() << " m"); ROS_INFO_STREAM_THROTTLE(0.5, "EE Pos Err
  // (x,y,z): " << e_ee.head<3>().transpose()); ROS_INFO_STREAM_THROTTLE(0.5,
  // "EE Rot Err (r,p,y): " << e_ee.tail<3>().transpose());
  // ROS_INFO_STREAM_THROTTLE(0.5, "Weighted Error Norm: " << ew.norm());
  // ROS_INFO_STREAM_THROTTLE(0.5, "Jacobian Jw Norm: " << Jw.norm());
  // ROS_INFO_STREAM_THROTTLE(0.5, "Damping Term: " << damping);
  // ROS_INFO_STREAM_THROTTLE(0.5, "Final Delta Theta: " <<
  // delta_theta.transpose()); ROS_INFO_STREAM_THROTTLE(0.5,
  // "------------------------------");

  for (int i = 0; i < n; ++i) {
    delta_theta(i) = std::max(std::min(delta_theta(i), max_delta), -max_delta);
  }

  return delta_theta;
}

void Controller::applyL1Compensation(Eigen::Vector3d &total_des_acc,
                                     const Odom_Data_t &odom,
                                     const Imu_Data_t &imu) {
  if (!param.l1.enable)
    return;

  // 1. 计算实际线加速度（世界系）
  // 假设 imu.a 为机体系“真”线加速度（不含重力）
  Eigen::Matrix3d R_wb = odom.q.toRotationMatrix();
  Eigen::Vector3d a_world_meas = R_wb * imu.a;

  // 2. 测量加速度低通滤波
  if (!l1_.inited) {
    l1_.a_lpf = a_world_meas;
    l1_.inited = true;
  } else {
    l1_.a_lpf =
        param.l1.k_a_lpf * a_world_meas + (1.0 - param.l1.k_a_lpf) * l1_.a_lpf;
  }

  // 更新 Debug 数据
  debug.a_lpf_x = l1_.a_lpf.x();
  debug.a_lpf_y = l1_.a_lpf.y();
  debug.a_lpf_z = l1_.a_lpf.z();

  // 3. 计算残差 r = a_meas - a_cmd_nominal
  Eigen::Vector3d residual = l1_.a_lpf - total_des_acc;

  // 4. 适应律 (Adaptation Law)
  double dt = param.ctrl_dt;
  l1_.d_hat += dt * param.l1.k_adapt * (residual - l1_.d_hat);

  // 5. L1 核心：低通滤波 (Filter Lane)
  l1_.d_hat_filt += dt * param.l1.k_filt * (l1_.d_hat - l1_.d_hat_filt);

  // 6. 投影/饱和，避免估计过大
  for (int i = 0; i < 3; ++i) {
    l1_.d_hat(i) = clamp_val(l1_.d_hat(i), -param.l1.d_max, param.l1.d_max);
    l1_.d_hat_filt(i) =
        clamp_val(l1_.d_hat_filt(i), -param.l1.d_max, param.l1.d_max);
  }

  // 7. 生成 L1 补偿后的期望加速度 (反馈到控制回路)
  total_des_acc -= l1_.d_hat_filt;

  // 8. Debug 可视化
  debug.l1_res_x = residual.x();
  debug.l1_res_y = residual.y();
  debug.l1_res_z = residual.z();
  debug.l1_dhat_x = l1_.d_hat.x();
  debug.l1_dhat_y = l1_.d_hat.y();
  debug.l1_dhat_z = l1_.d_hat.z();
  debug.l1_dhatf_x = l1_.d_hat_filt.x();
  debug.l1_dhatf_y = l1_.d_hat_filt.y();
  debug.l1_dhatf_z = l1_.d_hat_filt.z();
  debug.des_a_l1_x = total_des_acc.x();
  debug.des_a_l1_y = total_des_acc.y();
  debug.des_a_l1_z = total_des_acc.z();
}

/**
 * @brief 对 theta 进行限幅，确保每个关节角在配置的最小值和最大值范围内
 * @param theta 需要限幅的关节角向量（会被原地修改）
 * @param param 包含 theta 限制范围的参数结构
 */
void Controller::clipTheta(Eigen::VectorXd &theta,
                           const Parameter_t &param) const {
  // 确保 theta 至少有 3 个元素
  if (theta.size() < 3) {
    ROS_WARN("[clipTheta] theta size is %ld, expected at least 3",
             theta.size());
    return;
  }

  // theta_0
  if (theta(0) < param.thetas.theta_0_min) {
    theta(0) = param.thetas.theta_0_min;
  } else if (theta(0) > param.thetas.theta_0_max) {
    theta(0) = param.thetas.theta_0_max;
  }

  // theta_1
  if (theta(1) < param.thetas.theta_1_min) {
    theta(1) = param.thetas.theta_1_min;
  } else if (theta(1) > param.thetas.theta_1_max) {
    theta(1) = param.thetas.theta_1_max;
  }

  // theta_2
  if (theta(2) < param.thetas.theta_2_min) {
    theta(2) = param.thetas.theta_2_min;
  } else if (theta(2) > param.thetas.theta_2_max) {
    theta(2) = param.thetas.theta_2_max;
  }
}
