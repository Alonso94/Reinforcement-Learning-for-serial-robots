#! usr/bin/env
import gym
import rospy
from gym import error, spaces, utils
from gym.utils import seeding
from gazebo_msgs.srv import GetModelState, GetJointProperties, GetLinkState
from gazebo_msgs.msg import ModelState
from std_msgs.msg import Float64
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from math import radians
import numpy as np
import cv2
import glob
import os
from cv_bridge import CvBridge, CvBridgeError

metric = lambda a1, a2: ((a1.x - a2.x)**2 + (a1.y - a2.y)**2 + (a1.z - a2.z)**2)**(1/2)

bridge = CvBridge()

class CrumbCameraEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        rospy.init_node('arm_start')
        self.arm = [rospy.Publisher('arm_'+str(i+1)+'_joint/command', Float64, queue_size=10) for i in range(5)]
        self.gripper = rospy.Publisher('gripper_1_joint/command', Float64, queue_size=10)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.resetworld = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.joint_state = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)
        self.link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        self.aim = self.model_state('box', 'link_box').pose.position
        #self.aim = self.box_state('box', 'link_box').pose.position
        self.unpause()
        self.action_dim=4
        self.obs_dim=(3, 640, 480)
        low = -np.pi/2.0 * np.ones(4)
        high = np.pi/2.0 * np.ones(4)
        self.action_space = spaces.Box(low, high)
        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

    def _reset(self):
        for i in range(5):
            self.arm[i].publish(0.0)
            rospy.sleep(1)
        self.gripper.publish(2.0)
        self.resetworld()
        _, state = self.get_state()
        jpg_state = self.render('human')
        jpg_state=self.proc_img(jpg_state)
        print(jpg_state.shape)
        return jpg_state
    
    def _seed(self, seed = None):
        print('12')

    def _step(self, action):
        """action = (joint, step)"""
        gripper = self.get_link_pose('gripper_1_link')
        r1 = metric(gripper, self.aim)//0.001*0.001
        _, state = self.get_state()
        self.arm[action[0]].publish(state[action[0]]+action[1])
        rospy.sleep(1.5)
        gripper = self.get_link_pose('gripper_1_link')
        vec = self.get_link_pose('wrist_1_link')
        gripper_2_link = self.get_link_pose('gripper_2_link')
        gripper.x = gripper.x*0.5 + gripper_2_link.x*0.5 + (gripper.x - vec.x)*(0.02/0.11450014)
        gripper.y = gripper.y*0.5 + gripper_2_link.y*0.5 + (gripper.y - vec.y)*(0.02/0.11450014)
        gripper.z = gripper.z*0.5 + gripper_2_link.z*0.5 + (gripper.z - vec.z)*(0.02/0.11450014)
        r2 = metric(gripper, self.aim)//0.001*0.001
        reward = 0
        _, state = self.get_state()
        jpg_state = self.render('human') 
        jpg_state=self.proc_img(jpg_state)
        print(jpg_state.shape)
        reward = r1 - r2
        done = False
        if reward < 0:
            reward = reward * 2
        if reward == 0:
            reward = -1
        if (r2//0.025 == 0):
            done = True
            reward = 100 
        return jpg_state, reward, done

    def _render(self, mode='human', close=False):
        latest_img = rospy.wait_for_message("crumb_camera/image_raw", Image, timeout=3)
        img = bridge.imgmsg_to_cv2(latest_img, "bgr8")
        return img.T

    def get_state(self):
        return self.model_state('box', 'base_footprint').pose.position, [self.joint_state('arm_'+str(i+1)+'_joint').position[0] for i in range(5)]

    def get_link_pose(self, link):
        return self.link_state(link, 'base_footprint').link_state.pose.position
    
    def proc_img(self, img):
        result = img.T
        gray = cv2.cvtColor(img.T, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.01 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        result[markers == -1] = [255, 0, 0]
        result[:,:,1] = 50*markers
        return result.T








	

