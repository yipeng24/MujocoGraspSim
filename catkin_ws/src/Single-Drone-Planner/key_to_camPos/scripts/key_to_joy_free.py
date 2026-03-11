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
CH_PITCH = 3
CH_YAW = 4

def main():

    # initialize pygame to get keyboard event
    pygame.init()
    window_size = Rect(0, 0, 697, 319)
    screen = pygame.display.set_mode(window_size.size)

    # get an instance of RosPack with the default search paths
    rospack = rospkg.RosPack()

    # list all packages, equivalent to rospack list
    rospack.list() 

    # get the file path for rospy_tutorials
    img_path = rospack.get_path('key_to_camPos') + "/files/keyboard.jpg"

    print("img_path:",img_path)
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
    vel0 = 1.0
    omega0 = 1.0
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
                # z control
                if event.key == pygame.K_UP:
                    print('head up')
                    joy_.axes[CH_PITCH] = -omega
                if event.key == pygame.K_DOWN:
                    print('head down')
                    joy_.axes[CH_PITCH] = omega
                if event.key == pygame.K_LEFT:
                    print('turn left')
                    joy_.axes[CH_YAW] = omega
                if event.key == pygame.K_RIGHT:
                    print('turn right')
                    joy_.axes[CH_YAW] = -omega
                    
		#Publish
                joy_pub.publish(joy_)

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
                # z control
                if event.key == pygame.K_UP:
                    # print('up')
                    joy_.axes[CH_PITCH] = omega
                if event.key == pygame.K_DOWN:
                    # print('down')
                    joy_.axes[CH_PITCH] = -omega
                if event.key == pygame.K_LEFT:
                    # print('turn left')
                    joy_.axes[CH_YAW] = omega
                if event.key == pygame.K_RIGHT:
                    # print('turn right')
                    joy_.axes[CH_YAW] = -omega
                joy_pub.publish(joy_)



if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
