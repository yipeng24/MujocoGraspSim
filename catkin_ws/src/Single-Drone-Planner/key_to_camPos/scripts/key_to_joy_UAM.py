#!/usr/bin/env python

import pygame
from pygame.locals import *
import time
import sys
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
import rospkg

CH_FORWARD = 0
CH_LEFT = 1
CH_UP = 2
CH_YAW = 4
CH_PITCH = 5
CH_PITCH2 = 6
CH_ROLL = 7


BUTTON_CANCEL = 0
BUTTON_CONFIRM = 1
BUTTON_LOCK = 6
BUTTON_GRAB_OPEN = 2
BUTTON_GRAB_CLOSE = 3
BUTTON_MODE = 4
BUTTON_GEAR = 5

def main():

    # initialize pygame to get keyboard event
    pygame.init()
    window_size = Rect(0, 0, 648, 361)
    screen = pygame.display.set_mode(window_size.size)

    # get an instance of RosPack with the default search paths
    rospack = rospkg.RosPack()

    # list all packages, equivalent to rospack list
    rospack.list() 

    # get the file path for rospy_tutorials
    img_path = rospack.get_path('key_to_camPos') + "/files/keyboard_UAM.png"

    # print("img_path:",img_path)
    img = pygame.image.load(img_path)

    # initialize ros publisher
    joy_pub = rospy.Publisher('/joy_free', Joy, queue_size=10)
    rospy.init_node('key2joy_node_free')
    rate = rospy.Rate(50)
    
    # init joy message
    joy_ = Joy()    
    joy_.header.frame_id = 'map'
    joy_.header.stamp = rospy.Time.now()
    for i in range(8):
      joy_.axes.append(0.0)
    for i in range(11):
      joy_.buttons.append(0)
    vel0 = 0.5
    omega0 = 1.0

    have_first_event = False
    while not rospy.is_shutdown():
        rate.sleep()
        screen.blit(img, (1,1))
        pygame.display.flip()
        # reset message axes
        for i in range(8):
            joy_.axes[i] = 0.0

        # reset buttons
        for i in range(11):
            joy_.buttons[i] = 0
            
        for event in pygame.event.get():

            if event.type == KEYDOWN:
                vel = vel0
                omega = omega0
                # position control
                if event.key == pygame.K_w:
                    print('forward')
                    joy_.axes[CH_FORWARD] = vel
                if event.key == pygame.K_s:
                    print('backward')
                    joy_.axes[CH_FORWARD] = -vel
                if event.key == pygame.K_a:
                    print('left')
                    joy_.axes[CH_LEFT] = vel
                if event.key == pygame.K_d:
                    print('right')
                    joy_.axes[CH_LEFT] = -vel
                if event.key == pygame.K_q:
                    print('up')
                    joy_.axes[CH_UP] = vel
                if event.key == pygame.K_e:
                    print('down')
                    joy_.axes[CH_UP] = -vel
                if event.key == pygame.K_LEFT:
                    print('turn left')
                    joy_.axes[CH_YAW] = omega
                if event.key == pygame.K_RIGHT:
                    print('turn right')
                    joy_.axes[CH_YAW] = -omega

                # arm
                if event.key == pygame.K_DOWN:
                    print('[Joy] Arm  pitch +')
                    joy_.axes[CH_PITCH] = omega
                if event.key == pygame.K_j:
                    print('[Joy] Arm  pitch +')
                    joy_.axes[CH_PITCH2] = omega
                if event.key == pygame.K_m:
                    print('[Joy] Arm  roll +')
                    joy_.axes[CH_ROLL] = omega

                if event.key == pygame.K_UP:
                    print('[Joy] Arm  pitch -')
                    joy_.axes[CH_PITCH] = -omega
                if event.key == pygame.K_u:
                    print('[Joy] Arm  pitch -')
                    joy_.axes[CH_PITCH2] = -omega
                if event.key == pygame.K_n:
                    print('[Joy] Arm  roll -')
                    joy_.axes[CH_ROLL] = -omega
                
                # buttons
                if event.key == pygame.K_RETURN:
                    print('[JOY] CONFIRM')
                    joy_.buttons[BUTTON_CONFIRM] = 1

                if event.key == pygame.K_QUOTE:
                    print('[JOY] LOCAK')
                    joy_.buttons[BUTTON_LOCK] = 1

                if event.key == pygame.K_BACKSPACE:
                    print('[JOY] CANCEL')
                    joy_.buttons[BUTTON_CANCEL] = 1

                if event.key == pygame.K_o:
                    print('[JOY] GRAB OPEN')
                    joy_.buttons[BUTTON_GRAB_OPEN] = 1

                if event.key == pygame.K_p:
                    print('[JOY] GRAB CLOSE')
                    joy_.buttons[BUTTON_GRAB_CLOSE] = 1

                if event.key == pygame.K_5:
                    print('[JOY] Mode')
                    joy_.buttons[BUTTON_MODE] = 1

                if event.key == pygame.K_6:
                    print('[JOY] GEAR')
                    joy_.buttons[BUTTON_GEAR] = 1


		        # Publish
                joy_pub.publish(joy_)
                have_first_event = True

            # when keyup, reset velcity
            elif event.type == pygame.KEYUP:
                vel = 0.0
                omega = 0.0
                if event.key == pygame.K_w:
                    # print('forward')
                    joy_.axes[CH_FORWARD] = vel
                if event.key == pygame.K_s:
                    # print('backward')
                    joy_.axes[CH_FORWARD] = -vel
                if event.key == pygame.K_a:
                    # print('left')
                    joy_.axes[CH_LEFT] = vel
                if event.key == pygame.K_d:
                    # print('right')
                    joy_.axes[CH_LEFT] = -vel
                if event.key == pygame.K_q:
                    # print('up')
                    joy_.axes[CH_UP] = vel
                if event.key == pygame.K_e:
                    # print('down')
                    joy_.axes[CH_UP] = -vel
                if event.key == pygame.K_LEFT:
                    # print('turn left')
                    joy_.axes[CH_YAW] = omega
                if event.key == pygame.K_RIGHT:
                    # print('turn right')
                    joy_.axes[CH_YAW] = -omega

                # arm
                if event.key == pygame.K_DOWN:
                    print('[Joy] Arm  pitch +')
                    joy_.axes[CH_PITCH] = omega
                if event.key == pygame.K_j:
                    print('[Joy] Arm  pitch2 +')
                    joy_.axes[CH_PITCH2] = omega
                if event.key == pygame.K_m:
                    print('[Joy] Arm  roll +')
                    joy_.axes[CH_ROLL] = omega

                if event.key == pygame.K_UP:
                    print('[Joy] Arm  pitch -')
                    joy_.axes[CH_PITCH] = -omega
                if event.key == pygame.K_u:
                    print('[Joy] Arm  pitch2 -')
                    joy_.axes[CH_PITCH2] = -omega
                if event.key == pygame.K_n:
                    print('[Joy] Arm  roll -')
                    joy_.axes[CH_ROLL] = -omega


                joy_pub.publish(joy_)
                have_first_event = True

        if not have_first_event:
            joy_pub.publish(joy_)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
