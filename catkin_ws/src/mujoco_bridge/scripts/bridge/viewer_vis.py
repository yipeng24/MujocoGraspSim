"""
bridge/viewer_vis.py
====================
MuJoCo passive-viewer custom geometry rendering.

Draws (per frame, into user_scn):
  - Goal marker   : red sphere + yellow line from drone to goal
  - Flight trail  : blue fading spheres along past drone path
  - A* path       : green spheres + green capsule segments
  - Opt. trajectory: cyan spheres (every 3rd point)
"""

import numpy as np
import mujoco


# ── 正向运动学（与 ch_rc_sdf/params.yaml 运动链一致）────────────────────────
def _mat4(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T

def _Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def _Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def _Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def _fk_goal(pos, yaw, arm):
    """返回目标状态下 (p_body, p_j1, p_j2, p_ee) 世界系坐标。
    运动链：box0→box1 T=(-0.015,0.001,-0.096) Ry(th0)
            box1→box2 T=(0.15,-0.0012,0.0)   Ry(th1)
            box2→box3 T=(0.12,0.0,0.0006)    Rx(th2)
    """
    T0 = _mat4(_Rz(yaw), pos)
    T1 = T0 @ _mat4(_Ry(arm[0]), np.array([-0.015,  0.001, -0.096]))
    T2 = T1 @ _mat4(_Ry(arm[1]), np.array([ 0.150, -0.0012,  0.0  ]))
    T3 = T2 @ _mat4(_Rx(arm[2]), np.array([ 0.120,  0.0,     0.0006]))
    return pos.copy(), T1[:3, 3], T2[:3, 3], T3[:3, 3]


def _capsule_mat(a, b):
    """Return (rotation_matrix_flat, length) for a capsule aligned a→b."""
    diff = b - a
    ln   = float(np.linalg.norm(diff))
    if ln < 1e-6:
        return None, ln
    z   = diff / ln
    ref = np.array([0., 0., 1.])
    if abs(np.dot(z, ref)) > 0.99:
        ref = np.array([1., 0., 0.])
    x   = np.cross(ref, z);  x /= np.linalg.norm(x)
    y   = np.cross(z, x)
    mat = np.array([x[0], y[0], z[0],
                    x[1], y[1], z[1],
                    x[2], y[2], z[2]], dtype=np.float64)
    return mat, ln


class ViewerVis:
    """
    Stores path/trail buffers and renders custom geoms into user_scn each frame.
    Thread-safe: path buffers are updated under the provided lock.
    """

    def __init__(self, lock):
        self.lock        = lock
        self._trail      = []          # past drone positions
        self._trail_max  = 300
        self._astar_path = []          # A* waypoints
        self._opt_traj   = []          # optimized trajectory samples
        self._goal_pos   = None        # target position (np.array [x,y,z])
        self._goal_yaw   = 0.0         # target yaw
        self._goal_arm   = np.zeros(3) # target arm angles [th0, th1, th2]

    # ── Buffer updates (called from ROS callbacks) ────────────────────────

    def set_goal_state(self, pos, yaw, arm):
        """设置目标位置+臂角（由 goal picker 触发）。"""
        with self.lock:
            self._goal_pos = pos.copy()
            self._goal_yaw = float(yaw)
            self._goal_arm = arm.copy()

    def update_goal_arm(self, arm):
        """仅更新幽灵的臂角（键盘调整时调用，位置不变）。"""
        with self.lock:
            if self._goal_pos is not None:
                self._goal_arm = arm.copy()

    def clear_goal_state(self):
        """隐藏幽灵。"""
        with self.lock:
            self._goal_pos = None

    def update_trail(self, drone_pos):
        """Append current drone position to the flight trail."""
        self._trail.append(drone_pos.copy())
        if len(self._trail) > self._trail_max:
            self._trail.pop(0)

    def set_astar_path(self, path_msg):
        """nav_msgs/Path callback: update A* path buffer."""
        pts = [np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z])
               for p in path_msg.poses]
        with self.lock:
            self._astar_path = pts

    def set_opt_traj(self, path_msg):
        """nav_msgs/Path callback: update optimized trajectory buffer."""
        pts = [np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z])
               for p in path_msg.poses]
        with self.lock:
            self._opt_traj = pts

    # ── Main render call ──────────────────────────────────────────────────

    def render(self, scn, drone_pos, goal_pos):
        """
        Write custom geoms into scn.geoms starting at index 0.
        Sets scn.ngeom when done.  Call viewer.sync() afterwards.
        """
        idx = 0

        idx = self._render_goal(scn, idx, drone_pos, goal_pos)
        idx = self._render_goal_ghost(scn, idx)
        idx = self._render_trail(scn, idx)
        idx = self._render_astar(scn, idx)
        idx = self._render_opt_traj(scn, idx)

        scn.ngeom = idx

    # ── Per-layer helpers ─────────────────────────────────────────────────

    def _render_goal_ghost(self, scn, idx):
        """在目标位置渲染半透明"幽灵"无人机+机械臂（FK 正向运动学）。"""
        with self.lock:
            if self._goal_pos is None:
                return idx
            pos = self._goal_pos.copy()
            yaw = self._goal_yaw
            arm = self._goal_arm.copy()

        p0, p1, p2, p3 = _fk_goal(pos, yaw, arm)

        # — 机身圆盘（cylinder，Z轴朝上，带偏航旋转）
        if idx < scn.maxgeom:
            cy, sy = np.cos(yaw), np.sin(yaw)
            mat_disk = np.array([cy, sy, 0.,
                                 -sy, cy, 0.,
                                  0.,  0., 1.], dtype=np.float64)
            mujoco.mjv_initGeom(
                scn.geoms[idx], mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.array([0.17, 0.02, 0.]),
                p0, mat_disk,
                np.array([0.2, 0.85, 0.2, 0.40]))
            idx += 1

        # — 机械臂各段（capsule）：机身→关节1→关节2→末端
        link_segs = [
            (p0, p1, np.array([1.0, 0.55, 0.0, 0.75]), 0.013),
            (p1, p2, np.array([1.0, 0.75, 0.0, 0.75]), 0.011),
            (p2, p3, np.array([0.9, 0.90, 0.0, 0.75]), 0.009),
        ]
        for a, b, rgba, rad in link_segs:
            if idx >= scn.maxgeom:
                break
            mat, ln = _capsule_mat(a, b)
            if mat is not None and ln > 0.005:
                mujoco.mjv_initGeom(
                    scn.geoms[idx], mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.array([rad, ln / 2, 0.]),
                    (a + b) / 2, mat, rgba)
                idx += 1

        # — 末端执行器（红色球）
        if idx < scn.maxgeom:
            mujoco.mjv_initGeom(
                scn.geoms[idx], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.045, 0., 0.]),
                p3, np.eye(3).flatten(),
                np.array([1.0, 0.2, 0.2, 0.90]))
            idx += 1

        return idx

    def _render_goal(self, scn, idx, drone_pos, goal_pos):
        if goal_pos is None or idx >= scn.maxgeom:
            return idx

        # Red sphere at goal
        mujoco.mjv_initGeom(
            scn.geoms[idx], mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.12, 0., 0.]), goal_pos,
            np.eye(3).flatten(), np.array([1.0, 0.2, 0.2, 0.9]))
        idx += 1

        # Yellow capsule: drone → goal
        if idx < scn.maxgeom:
            diff   = goal_pos - drone_pos
            length = float(np.linalg.norm(diff))
            if length > 0.05:
                mat, _ = _capsule_mat(drone_pos, goal_pos)
                if mat is not None:
                    mujoco.mjv_initGeom(
                        scn.geoms[idx], mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.array([0.012, length / 2, 0.]),
                        (drone_pos + goal_pos) / 2, mat,
                        np.array([1.0, 0.85, 0.0, 0.6]))
                    idx += 1
        return idx

    def _render_trail(self, scn, idx):
        trail_snap = list(self._trail)
        sampled    = trail_snap[::5]
        n_trail    = max(len(sampled), 1)
        for i, pt in enumerate(sampled):
            if idx >= scn.maxgeom:
                break
            frac = (i + 1) / n_trail
            r    = 0.012 + 0.022 * frac
            mujoco.mjv_initGeom(
                scn.geoms[idx], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([r, 0., 0.]), pt,
                np.eye(3).flatten(),
                np.array([0.3, 0.6, 1.0, 0.25 + 0.55 * frac]))
            idx += 1
        return idx

    def _render_astar(self, scn, idx):
        with self.lock:
            astar_snap = list(self._astar_path)

        for i, pt in enumerate(astar_snap):
            if idx >= scn.maxgeom:
                break
            # Green sphere at waypoint
            mujoco.mjv_initGeom(
                scn.geoms[idx], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.06, 0., 0.]), pt,
                np.eye(3).flatten(), np.array([0.2, 0.9, 0.2, 0.9]))
            idx += 1
            # Green capsule to next waypoint
            if i < len(astar_snap) - 1 and idx < scn.maxgeom:
                mat, ln = _capsule_mat(pt, astar_snap[i + 1])
                if mat is not None:
                    mujoco.mjv_initGeom(
                        scn.geoms[idx], mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.array([0.015, ln / 2, 0.]),
                        (pt + astar_snap[i + 1]) / 2, mat,
                        np.array([0.2, 0.9, 0.2, 0.6]))
                    idx += 1
        return idx

    def _render_opt_traj(self, scn, idx):
        with self.lock:
            traj_snap = list(self._opt_traj)

        for pt in traj_snap[::3]:   # every 3rd point for density
            if idx >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                scn.geoms[idx], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.04, 0., 0.]), pt,
                np.eye(3).flatten(), np.array([0.0, 0.9, 0.9, 0.9]))
            idx += 1
        return idx
