#!/usr/bin/env python3
"""
mujoco_goal_picker.py
=====================
2D 俯视地图窗口，双击选择飞行目标点。

操作方式:
  双击地图         → 设置目标点（无人机飞到该 XY，保持当前高度）
  滚轮上/下        → 放大 / 缩小
  中键拖拽         → 平移地图（无限范围）
  H 键             → 就地悬停

发布: /position_cmd (quadrotor_msgs/PositionCommand) at 20Hz
订阅: /odom (nav_msgs/Odometry)
"""

import threading
import tkinter as tk
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from geometry_msgs.msg import PoseStamped


# ── 场景障碍物（与 scene_drone.xml 一致）────────────────────────────────
# MuJoCo box size = 半轴长；cylinder size[0] = 半径
OBSTACLES = [
    ('circle', {'cx':  1.2,  'cy':  0.8,  'r':  0.12,
                'color': '#9AA0AD', 'label': 'Pillar'}),
    ('rect',   {'cx': -0.8,  'cy': -0.6,  'w': 0.80, 'h': 0.60,
                'color': '#B88050', 'label': 'Table'}),
]

CANVAS_SIZE  = 560    # 画布像素
INIT_PX_PER_M = 80.0  # 初始缩放：80 像素/米（视野约 ±3.5m）


class GoalPickerApp:
    def __init__(self):
        rospy.init_node('mujoco_goal_picker', anonymous=True)

        # planning_mode=true → 发布 PoseStamped 到 /move_base_simple/goal
        # planning_mode=false → 发布 PositionCommand 到 /position_cmd（直接控制）
        self._planning_mode = rospy.get_param('~planning_mode', False)
        self._last_goal_pub_time = rospy.Time(0)   # 限速：规划模式下 1Hz

        self._lock      = threading.Lock()
        self._drone_pos = np.zeros(3)
        self._drone_yaw = 0.0
        self._has_odom  = False
        self._goal_pos  = None
        self._goal_yaw  = 0.0

        rospy.Subscriber('/odom', Odometry, self._odom_cb, queue_size=1)
        self._cmd_pub  = rospy.Publisher('/position_cmd',  PositionCommand, queue_size=1)
        self._goal_pub = rospy.Publisher('/goal_preview', PoseStamped,     queue_size=1)

        mode_str = 'planning (PoseStamped→/goal_preview, CONFIRM→/move_base_simple/goal)' \
                   if self._planning_mode else 'direct (PositionCommand→/position_cmd)'
        rospy.loginfo(f'[goal_picker] mode: {mode_str}')

        # ── 视图状态（支持缩放 + 平移）──────────────────────────────────
        self._px_per_m  = INIT_PX_PER_M   # 缩放：像素/米
        self._view_cx   = 0.0             # 视图中心世界坐标 X
        self._view_cy   = 0.0             # 视图中心世界坐标 Y
        self._pan_start = None            # 中键拖拽起始点 (canvas_x, canvas_y)
        self._pan_origin = None           # 拖拽起始时的 view center

        # ── Tkinter 窗口 ─────────────────────────────────────────────────
        self._root = tk.Tk()
        mode_title = '【规划模式】' if self._planning_mode else '【直飞模式】'
        self._root.title(f'Goal Picker {mode_title}  |  双击→预览目标  ENTER→确认飞行  滚轮缩放')
        self._root.resizable(False, False)

        tk.Label(
            self._root,
            text='双击→预览幽灵   ENTER键确认飞行   滚轮→缩放   中键拖→平移   H→悬停',
            font=('Helvetica', 6), bg='#2A2A2A', fg='#EAEAEA', pady=4
        ).pack(fill='x')

        self._canvas = tk.Canvas(
            self._root, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg='#F4F4F4', cursor='crosshair', highlightthickness=0)
        self._canvas.pack()

        # 鼠标绑定
        self._canvas.bind('<Double-Button-1>', self._on_double_click)
        self._canvas.bind('<Button-4>',        self._on_scroll_up)    # Linux 滚轮上
        self._canvas.bind('<Button-5>',        self._on_scroll_down)  # Linux 滚轮下
        self._canvas.bind('<MouseWheel>',      self._on_mousewheel)   # Windows/Mac
        self._canvas.bind('<Button-2>',        self._on_pan_start)    # 中键按下
        self._canvas.bind('<B2-Motion>',       self._on_pan_move)     # 中键拖拽
        self._canvas.bind('<ButtonRelease-2>', self._on_pan_end)      # 中键松开
        self._root.bind('<h>', lambda _: self._set_hover())
        self._root.bind('<H>', lambda _: self._set_hover())

        self._status_var = tk.StringVar(value='等待起飞...')
        tk.Label(
            self._root, textvariable=self._status_var,
            font=('Helvetica', 10), bg='#1E1E1E', fg='#88AADD', pady=3
        ).pack(fill='x')

        # 坐标显示（随鼠标移动更新）
        self._coord_var = tk.StringVar(value='')
        tk.Label(
            self._root, textvariable=self._coord_var,
            font=('Helvetica', 9), bg='#1A1A1A', fg='#888', pady=2
        ).pack(fill='x')
        self._canvas.bind('<Motion>', self._on_mouse_move)

    # ────────────────────────────────────────────────────────────────────
    # 坐标转换（动态缩放 + 平移）
    # ────────────────────────────────────────────────────────────────────

    def _w2c(self, wx, wy):
        """世界坐标 → 画布像素。"""
        cx = (wx - self._view_cx) * self._px_per_m + CANVAS_SIZE / 2
        cy = -(wy - self._view_cy) * self._px_per_m + CANVAS_SIZE / 2
        return cx, cy

    def _c2w(self, cx, cy):
        """画布像素 → 世界坐标。"""
        wx = (cx - CANVAS_SIZE / 2) / self._px_per_m + self._view_cx
        wy = -(cy - CANVAS_SIZE / 2) / self._px_per_m + self._view_cy
        return wx, wy

    # ────────────────────────────────────────────────────────────────────
    # ROS 回调
    # ────────────────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        with self._lock:
            self._drone_pos[:] = [p.x, p.y, p.z]
            self._drone_yaw    = float(np.arctan2(siny, cosy))
            self._has_odom     = True

    # ────────────────────────────────────────────────────────────────────
    # 鼠标事件
    # ────────────────────────────────────────────────────────────────────

    def _on_double_click(self, event):
        wx, wy = self._c2w(event.x, event.y)
        with self._lock:
            if not self._has_odom:
                self._status_var.set('请先触发起飞，等待无人机悬停')
                return
            dz  = float(self._drone_pos[2])
            yaw = self._drone_yaw
            self._goal_pos = np.array([wx, wy, max(0.2, dz)])
            self._goal_yaw = yaw
        action = '调好臂角后按 ENTER 确认' if self._planning_mode else '飞行中...'
        self._status_var.set(
            f'预览目标: ({wx:.2f}, {wy:.2f}, {self._goal_pos[2]:.2f}) m  → {action}')
        rospy.loginfo(
            f'[goal_picker] 目标: ({wx:.2f}, {wy:.2f}, {self._goal_pos[2]:.2f})')

    def _on_scroll_up(self, event):
        self._zoom(event.x, event.y, factor=1.2)

    def _on_scroll_down(self, event):
        self._zoom(event.x, event.y, factor=1.0 / 1.2)

    def _on_mousewheel(self, event):
        factor = 1.2 if event.delta > 0 else 1.0 / 1.2
        self._zoom(event.x, event.y, factor)

    def _zoom(self, cx, cy, factor):
        """以鼠标位置为中心缩放。"""
        wx, wy = self._c2w(cx, cy)
        self._px_per_m = max(5.0, min(500.0, self._px_per_m * factor))
        # 保持鼠标下的世界点不动
        new_cx = wx - (cx - CANVAS_SIZE / 2) / self._px_per_m
        new_cy = wy + (cy - CANVAS_SIZE / 2) / self._px_per_m
        self._view_cx = new_cx
        self._view_cy = new_cy
        self._redraw_all()

    def _on_pan_start(self, event):
        self._pan_start  = (event.x, event.y)
        self._pan_origin = (self._view_cx, self._view_cy)

    def _on_pan_move(self, event):
        if self._pan_start is None:
            return
        dx = (event.x - self._pan_start[0]) / self._px_per_m
        dy = (event.y - self._pan_start[1]) / self._px_per_m
        self._view_cx = self._pan_origin[0] - dx
        self._view_cy = self._pan_origin[1] + dy
        self._redraw_all()

    def _on_pan_end(self, event):
        self._pan_start  = None
        self._pan_origin = None

    def _on_mouse_move(self, event):
        wx, wy = self._c2w(event.x, event.y)
        self._coord_var.set(f'鼠标位置: ({wx:.2f}, {wy:.2f}) m')

    # ────────────────────────────────────────────────────────────────────
    # 目标控制
    # ────────────────────────────────────────────────────────────────────

    def _set_hover(self):
        with self._lock:
            if not self._has_odom:
                return
            pos = self._drone_pos.copy()
            yaw = self._drone_yaw
            self._goal_pos = pos.copy()
            self._goal_yaw = yaw
        self._status_var.set(
            f'就地悬停: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m')
        rospy.loginfo('[goal_picker] 就地悬停')

    def _publish_goal(self):
        with self._lock:
            goal = self._goal_pos.copy() if self._goal_pos is not None else None
            yaw  = self._goal_yaw
        if goal is None:
            return

        now = rospy.Time.now()

        if self._planning_mode:
            # 规划模式：发布 /goal_preview 供 ghost 显示，限速 5Hz
            # 不发 /move_base_simple/goal —— 由 joy_to_position_cmd CONFIRM 触发
            if (now - self._last_goal_pub_time).to_sec() < 0.2:
                return
            self._last_goal_pub_time = now
            import math
            ps = PoseStamped()
            ps.header.stamp    = now
            ps.header.frame_id = 'world'
            ps.pose.position.x = float(goal[0])
            ps.pose.position.y = float(goal[1])
            ps.pose.position.z = float(goal[2])
            ps.pose.orientation.w = float(math.cos(yaw / 2.0))
            ps.pose.orientation.z = float(math.sin(yaw / 2.0))
            self._goal_pub.publish(ps)
        else:
            # 直接控制模式：20Hz 持续发布 PositionCommand
            cmd = PositionCommand()
            cmd.header.stamp    = now
            cmd.header.frame_id = 'world'
            cmd.position.x = float(goal[0])
            cmd.position.y = float(goal[1])
            cmd.position.z = float(goal[2])
            cmd.yaw         = float(yaw)
            cmd.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
            self._cmd_pub.publish(cmd)

    # ────────────────────────────────────────────────────────────────────
    # 绘制
    # ────────────────────────────────────────────────────────────────────

    def _redraw_all(self):
        """缩放/平移后重绘所有静态元素。"""
        self._canvas.delete('static')
        self._draw_static()

    def _draw_static(self):
        c = self._canvas
        s = self._px_per_m   # 缩放因子

        # 网格（按当前缩放决定间距）
        grid_step = 1.0
        if s > 150:
            grid_step = 0.25
        elif s > 60:
            grid_step = 0.5

        # 计算当前视图覆盖的世界坐标范围
        half_w = CANVAS_SIZE / 2 / s
        wx_min = self._view_cx - half_w - grid_step
        wx_max = self._view_cx + half_w + grid_step
        wy_min = self._view_cy - half_w - grid_step
        wy_max = self._view_cy + half_w + grid_step

        vx = np.arange(np.floor(wx_min / grid_step) * grid_step,
                       wx_max, grid_step)
        vy = np.arange(np.floor(wy_min / grid_step) * grid_step,
                       wy_max, grid_step)

        for v in vx:
            x1, y1 = self._w2c(v, wy_min)
            x2, y2 = self._w2c(v, wy_max)
            color = '#BBBBBB' if abs(v) < 1e-6 else '#DCDCDC'
            c.create_line(x1, y1, x2, y2, fill=color, tags='static')
            if s > 30:
                lx, ly = self._w2c(v, self._view_cy)
                c.create_text(lx, ly + 8, text=f'{v:.1f}',
                              font=('Helvetica', 7), fill='#AAAAAA', tags='static')
        for v in vy:
            x1, y1 = self._w2c(wx_min, v)
            x2, y2 = self._w2c(wx_max, v)
            color = '#BBBBBB' if abs(v) < 1e-6 else '#DCDCDC'
            c.create_line(x1, y1, x2, y2, fill=color, tags='static')

        # 坐标轴
        ox, oy   = self._w2c(0, 0)
        ax1, ay1 = self._w2c(0.5, 0)
        ax2, ay2 = self._w2c(0, 0.5)
        c.create_line(ox, oy, ax1, ay1, fill='#CC2222', width=2,
                      arrow='last', tags='static')
        c.create_line(ox, oy, ax2, ay2, fill='#22AA22', width=2,
                      arrow='last', tags='static')
        c.create_text(ax1+12, ay1, text='+X', fill='#CC2222',
                      font=('Helvetica', 9, 'bold'), tags='static')
        c.create_text(ax2, ay2-12, text='+Y', fill='#22AA22',
                      font=('Helvetica', 9, 'bold'), tags='static')

        # 障碍物
        for otype, p in OBSTACLES:
            if otype == 'circle':
                cx_, cy_ = self._w2c(p['cx'], p['cy'])
                r_px = p['r'] * s
                c.create_oval(cx_-r_px, cy_-r_px, cx_+r_px, cy_+r_px,
                              fill=p['color'], outline='#444', width=1, tags='static')
                if r_px > 6:
                    c.create_text(cx_, cy_-r_px-7, text=p['label'],
                                  font=('Helvetica', 8), fill='#444', tags='static')
            elif otype == 'rect':
                cx_, cy_ = self._w2c(p['cx'], p['cy'])
                hw = p['w'] / 2 * s
                hh = p['h'] / 2 * s
                c.create_rectangle(cx_-hw, cy_-hh, cx_+hw, cy_+hh,
                                   fill=p['color'], outline='#444', width=1, tags='static')
                if hw > 15:
                    c.create_text(cx_, cy_, text=p['label'],
                                  font=('Helvetica', 8, 'bold'), fill='white', tags='static')

        # 缩放比例尺
        bar_m   = 1.0
        bar_px  = bar_m * s
        bx1     = CANVAS_SIZE - 20 - bar_px
        bx2     = CANVAS_SIZE - 20
        by      = CANVAS_SIZE - 15
        c.create_line(bx1, by, bx2, by, fill='#555', width=2, tags='static')
        c.create_line(bx1, by-4, bx1, by+4, fill='#555', width=2, tags='static')
        c.create_line(bx2, by-4, bx2, by+4, fill='#555', width=2, tags='static')
        c.create_text((bx1+bx2)/2, by-8, text=f'{bar_m:.1f}m',
                      font=('Helvetica', 8), fill='#555', tags='static')

    def _redraw_dynamic(self):
        self._canvas.delete('dynamic')
        c = self._canvas

        with self._lock:
            pos  = self._drone_pos.copy()
            yaw  = self._drone_yaw
            goal = self._goal_pos.copy() if self._goal_pos is not None else None
            has  = self._has_odom

        if not has:
            return

        dx, dy = self._w2c(pos[0], pos[1])

        # 虚线：无人机 → 目标
        if goal is not None:
            gx, gy = self._w2c(goal[0], goal[1])
            c.create_line(dx, dy, gx, gy,
                          fill='#FF9999', width=1, dash=(5, 4), tags='dynamic')
            # 目标十字圆
            R = 11
            c.create_oval(gx-R, gy-R, gx+R, gy+R,
                          outline='#DD0000', width=2, tags='dynamic')
            c.create_line(gx-R, gy, gx+R, gy, fill='#DD0000', width=2, tags='dynamic')
            c.create_line(gx, gy-R, gx, gy+R, fill='#DD0000', width=2, tags='dynamic')
            c.create_text(gx, gy+R+9,
                          text=f'({goal[0]:.1f},{goal[1]:.1f})',
                          font=('Helvetica', 7), fill='#DD0000', tags='dynamic')

        # 无人机：蓝圆 + 朝向箭头
        R = 9
        c.create_oval(dx-R, dy-R, dx+R, dy+R,
                      fill='#3377FF', outline='#1144BB', width=2, tags='dynamic')
        L = 16
        ex = dx + L * np.cos(yaw)
        ey = dy - L * np.sin(yaw)
        c.create_line(dx, dy, ex, ey,
                      fill='white', width=2, arrow='last', tags='dynamic')
        c.create_text(dx, dy+R+9,
                      text=f'({pos[0]:.1f},{pos[1]:.1f})',
                      font=('Helvetica', 7), fill='#3377FF', tags='dynamic')

    # ────────────────────────────────────────────────────────────────────
    # 主循环
    # ────────────────────────────────────────────────────────────────────

    def _tick(self):
        if not rospy.is_shutdown():
            self._redraw_dynamic()
            self._publish_goal()
            self._root.after(50, self._tick)
        else:
            self._root.quit()

    def run(self):
        self._draw_static()
        threading.Thread(target=rospy.spin, daemon=True).start()
        self._root.after(100, self._tick)
        try:
            self._root.mainloop()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    app = GoalPickerApp()
    app.run()
