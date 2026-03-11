#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt

def vis_single_image(img,title):
    if type(img) is not np.ndarray:
        img = img.detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    im = ax.imshow(img, cmap='jet', vmin = 0, vmax = 0.25)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    ax.axis('off')
    plt.show()
                        
class ColorbarNode:
    def __init__(self):
        rospy.init_node('colorbar_node', anonymous=True)
        # self.img_sub = rospy.Subscriber('/sdf_map/grid_CR_img', Image, self.img_callback)
        self.img_sub = rospy.Subscriber('/loss_img', Image, self.img_callback)
        self.bridge = CvBridge()

        rospy.spin()

    def img_callback(self, msg):
        depth_cv2 = self.bridge.imgmsg_to_cv2(msg, "32FC1")# 0-65.535
        depth_np0 = np.array(depth_cv2, dtype=np.float32)
        vis_single_image(depth_np0, 'depth')

if __name__ == '__main__':
    try:
        goal_to_odom_publisher = ColorbarNode()
    except rospy.ROSInterruptException:
        pass