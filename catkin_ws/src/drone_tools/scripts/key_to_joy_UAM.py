#!/usr/bin/env python3
"""
key_to_joy_UAM.py  (drone_tools / MuJoCo sim version)
======================================================
pygame 键盘窗口 → 发布 /joy_free (sensor_msgs/Joy)

键位说明:
  W/S/A/D      前/后/左/右 (drone body frame)
  Q/E          上/下
  ←/→         偏航
  ↑/↓         机械臂 pitch (arm joint 0)
  U/J          机械臂 pitch2 (arm joint 1)
  M/N          机械臂 roll  (arm joint 2)
  O/P          夹爪开/关
  5            MODE  — 切换 MANUAL ↔ PLANNING 模式
  6            GEAR  — 就地悬停 (hold position)
  '            LOCK
  BACKSPACE    CANCEL
  ENTER        CONFIRM

订阅: /ctrl_mode (std_msgs/String) → 在窗口底部显示当前模式
发布: /joy_free  (sensor_msgs/Joy)
"""

import threading
import pygame
from pygame.locals import *

import rospy
import rospkg
from sensor_msgs.msg import Joy
from std_msgs.msg import String

# ── Joy 轴/按钮索引（与 joy_to_position_cmd.py / input.h 一致）──────────
CH_FORWARD    = 0
CH_LEFT       = 1
CH_UP         = 2
CH_YAW        = 4
CH_PITCH      = 5
CH_PITCH2     = 6
CH_ROLL       = 7

BUTTON_CANCEL     = 0
BUTTON_CONFIRM    = 1
BUTTON_GRAB_OPEN  = 2
BUTTON_GRAB_CLOSE = 3
BUTTON_MODE       = 4   # key 5 → MANUAL ↔ PLANNING
BUTTON_GEAR       = 5   # key 6 → hover in place
BUTTON_LOCK       = 6

IMG_W, IMG_H   = 648, 361
STATUS_BAR_H   = 50
WINDOW_H       = IMG_H + STATUS_BAR_H


def main():
    pygame.init()
    screen = pygame.display.set_mode((IMG_W, WINDOW_H))
    pygame.display.set_caption('Drone Keyboard Control  |  5=MODE  6=GEAR(Hover)')

    rospack = rospkg.RosPack()
    img_path = rospack.get_path('drone_tools') + '/files/keyboard_UAM.png'
    img = pygame.image.load(img_path).convert()

    font_big  = pygame.font.SysFont('monospace', 18, bold=True)
    font_small = pygame.font.SysFont('monospace', 13)


    rospy.init_node('key2joy_node_free')
    joy_pub = rospy.Publisher('/joy_free', Joy, queue_size=10)
    rate    = rospy.Rate(50)

    #  joy_to_position_cmd 发布
    _state      = {'mode': 'MANUAL', 'locked': False, 'hovering': False, 'fsm': '---'}
    _state_lock = threading.Lock()

    def _mode_cb(msg):
        with _state_lock:
    
            parts = msg.data.split('|')
            _state['mode']   = parts[0]
            _state['locked'] = (len(parts) > 1 and parts[1] == 'LOCKED')

    def _fsm_cb(msg):
        with _state_lock:
            _state['fsm'] = msg.data

    rospy.Subscriber('/ctrl_mode', String, _mode_cb, queue_size=1)
    rospy.Subscriber('/planning/fsm_state', String, _fsm_cb, queue_size=1)

    
    joy_ = Joy()
    joy_.header.frame_id = 'map'
    joy_.header.stamp    = rospy.Time.now()
    joy_.axes    = [0.0] * 8
    joy_.buttons = [0]   * 11

    vel0   = 0.5
    omega0 = 1.0

    have_first_event = False

    while not rospy.is_shutdown():
        rate.sleep()

  
        screen.blit(img, (0, 0))

        with _state_lock:
            mode_str = _state['mode']
            locked   = _state['locked']
            hovering = _state['hovering']
            fsm_str  = _state['fsm']

        
        bar_rect  = pygame.Rect(0, IMG_H, IMG_W, STATUS_BAR_H)
        bar_color = (30, 90, 30) if mode_str == 'MANUAL' else (100, 60, 10)
        pygame.draw.rect(screen, bar_color, bar_rect)

        mode_color = (80, 255, 80) if mode_str == 'MANUAL' else (255, 165, 40)
        label = f'MODE: {mode_str}'
        if locked:
            label += '  [LOCKED]'
        mode_surf = font_big.render(label, True, mode_color)
        screen.blit(mode_surf, (10, IMG_H + 4))

        fsm_colors = {'IDLE': (130, 130, 130), 'HOVER': (100, 200, 255),
                      'GOAL': (255, 220, 50),  '---': (100, 100, 100)}
        fsm_color = fsm_colors.get(fsm_str, (255, 100, 100))
        fsm_surf  = font_big.render(f'PLANNER: {fsm_str}', True, fsm_color)
        screen.blit(fsm_surf, (IMG_W - fsm_surf.get_width() - 10, IMG_H + 4))

        hint = "5=MODE  6=GEAR(Hover)  '=LOCK  ENTER=起飞  BS=降落"
        hint_surf = font_small.render(hint, True, (180, 180, 180))
        screen.blit(hint_surf, (10, IMG_H + 32))

        pygame.display.flip()

   
        joy_.axes    = [0.0] * 8
        joy_.buttons = [0]   * 11


        for event in pygame.event.get():
            if event.type == QUIT:
                rospy.signal_shutdown('window closed')
                return

            if event.type == KEYDOWN:
                vel   = vel0
                omega = omega0

                # 位置控制
                if event.key == pygame.K_w:     joy_.axes[CH_FORWARD] =  vel
                if event.key == pygame.K_s:     joy_.axes[CH_FORWARD] = -vel
                if event.key == pygame.K_a:     joy_.axes[CH_LEFT]    =  vel
                if event.key == pygame.K_d:     joy_.axes[CH_LEFT]    = -vel
                if event.key == pygame.K_q:     joy_.axes[CH_UP]      =  vel
                if event.key == pygame.K_e:     joy_.axes[CH_UP]      = -vel
                if event.key == pygame.K_LEFT:  joy_.axes[CH_YAW]     =  omega
                if event.key == pygame.K_RIGHT: joy_.axes[CH_YAW]     = -omega

                # 机械臂
                if event.key == pygame.K_DOWN:  joy_.axes[CH_PITCH]  =  omega
                if event.key == pygame.K_UP:    joy_.axes[CH_PITCH]  = -omega
                if event.key == pygame.K_j:     joy_.axes[CH_PITCH2] =  omega
                if event.key == pygame.K_u:     joy_.axes[CH_PITCH2] = -omega
                if event.key == pygame.K_m:     joy_.axes[CH_ROLL]   =  omega
                if event.key == pygame.K_n:     joy_.axes[CH_ROLL]   = -omega

                # 按钮（单次触发）
                if event.key == pygame.K_RETURN:    joy_.buttons[BUTTON_CONFIRM]    = 1
                if event.key == pygame.K_BACKSPACE:  joy_.buttons[BUTTON_CANCEL]    = 1
                if event.key == pygame.K_QUOTE:      joy_.buttons[BUTTON_LOCK]      = 1
                if event.key == pygame.K_o:          joy_.buttons[BUTTON_GRAB_OPEN] = 1
                if event.key == pygame.K_p:          joy_.buttons[BUTTON_GRAB_CLOSE]= 1
                if event.key == pygame.K_5:          joy_.buttons[BUTTON_MODE]      = 1
                if event.key == pygame.K_6:
                    joy_.buttons[BUTTON_GEAR] = 1
                    with _state_lock:
                        _state['hovering'] = True

                # 有移动输入时退出悬停状态
                _move_keys = (pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d,
                              pygame.K_q, pygame.K_e, pygame.K_LEFT, pygame.K_RIGHT)
                if event.key in _move_keys:
                    with _state_lock:
                        _state['hovering'] = False

                joy_pub.publish(joy_)
                have_first_event = True

            elif event.type == KEYUP:
                # 松键时轴归零并发布（按钮不需要，已在 KEYDOWN 发一次）
                if event.key == pygame.K_w:     joy_.axes[CH_FORWARD] = 0.0
                if event.key == pygame.K_s:     joy_.axes[CH_FORWARD] = 0.0
                if event.key == pygame.K_a:     joy_.axes[CH_LEFT]    = 0.0
                if event.key == pygame.K_d:     joy_.axes[CH_LEFT]    = 0.0
                if event.key == pygame.K_q:     joy_.axes[CH_UP]      = 0.0
                if event.key == pygame.K_e:     joy_.axes[CH_UP]      = 0.0
                if event.key == pygame.K_LEFT:  joy_.axes[CH_YAW]     = 0.0
                if event.key == pygame.K_RIGHT: joy_.axes[CH_YAW]     = 0.0
                if event.key in (pygame.K_DOWN, pygame.K_UP):
                    joy_.axes[CH_PITCH]  = 0.0
                if event.key in (pygame.K_j, pygame.K_u):
                    joy_.axes[CH_PITCH2] = 0.0
                if event.key in (pygame.K_m, pygame.K_n):
                    joy_.axes[CH_ROLL]   = 0.0

                joy_pub.publish(joy_)
                have_first_event = True

        if not have_first_event:
            joy_pub.publish(joy_)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
