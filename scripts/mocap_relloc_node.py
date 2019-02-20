#!/usr/bin/env python

import collections
import copy
import threading
import sys

import numpy as np

import rospy
import rostopic

import actionlib
from std_msgs.msg import Bool, Duration
import tf2_ros
import tf2_geometry_msgs
from tf2_geometry_msgs import transform_to_kdl
import tf_conversions as tfc
import PyKDL as kdl

from geometry_msgs.msg import Pose, PoseStamped, TransformStamped

from volaly_msgs.msg import EmptyAction, EmptyFeedback, EmptyResult

class MocapRellocNode:
    def __init__(self):
        action_ns = rospy.get_param('~action_ns', '')
        robot_name = rospy.get_param('~robot_name')
        self.lock = threading.RLock()

        self.publish_rate = rospy.get_param('~publish_rate', 50)

        sec = rospy.get_param('~tf_exp_time', 90.0)
        self.tf_exp_time = rospy.Duration(sec)

        if self.tf_exp_time <= rospy.Duration(sys.float_info.epsilon):
            rospy.logwarn("tf_exp_time is set to 0.0, the relloc transform will never expire!")

        self.human_frame = rospy.get_param('~human_frame_id', 'human_footprint')
        self.robot_root_frame = rospy.get_param('~robot_root_frame', robot_name + '/odom')

        pointing_ray_topic = rospy.get_param('~pointing_ray_topic', 'pointing_ray')
        self.sub_pointing_ray = rospy.Subscriber(pointing_ray_topic, PoseStamped, self.pointing_ray_cb)
        self.pointing_ray_msg = None

        self.ray_direction_frame = rospy.get_param('~ray_direction_frame', 'pointer')

        human_pose_topic = rospy.get_param('~human_pose_topic', '/optitrack/head')
        self.sub_human_pose = rospy.Subscriber(human_pose_topic, PoseStamped, self.human_pose_cb)
        self.human_pose_msg = None

        self.cached_yaw = 0.0

        self.reset_state()

        self.tf_buff = tf2_ros.Buffer()
        self.tf_ls = tf2_ros.TransformListener(self.tf_buff)
        self.tf_br = tf2_ros.TransformBroadcaster()

        self.mocap_relloc_server = actionlib.SimpleActionServer(action_ns + '/relloc_action', EmptyAction, self.execute_relloc, False)
        self.mocap_relloc_server.start()

    def kdl_to_transform(self, k):
        t = TransformStamped()
        t.transform.translation.x = k.p.x()
        t.transform.translation.y = k.p.y()
        t.transform.translation.z = k.p.z()
        (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w) = k.M.GetQuaternion()

        return t

    def reset_state(self):
        with self.lock:
            self.cached_tf = None
            self.estimated = False
            self.tf_expired = True

    def human_pose_cb(self, msg):
        self.human_pose_msg = msg

    def pointing_ray_cb(self, msg):
        self.pointing_ray_msg = msg

    def calculate_pose(self, ignore_heading = False):
        if not self.human_pose_msg or not self.pointing_ray_msg:
            if not self.human_pose_msg:
                rospy.logerr('Cannot relloc: user\'s MOCAP pose is not known')
            if not self.pointing_ray_msg:
                rospy.logerr('Cannot relloc: pointing ray is not known')

            return None

        # Project human pose on the ground and adjust pointing ray
        tmp_h_f = tfc.fromMsg(self.human_pose_msg.pose)
        tmp_ray_f = tfc.fromMsg(self.pointing_ray_msg.pose)
        _,_,yaw = tmp_h_f.M.GetRPY()
        _,_,ray_yaw = tmp_ray_f.M.GetRPY()

        ######### The bug that cost me a finger being cut by the drone blades #########################################
        # human_f = kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, ray_yaw - yaw), kdl.Vector(tmp_h_f.p.x(), tmp_h_f.p.y(), 0.0))
        ###############################################################################################################

        if ignore_heading:
            human_f = kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, self.cached_yaw), kdl.Vector(tmp_h_f.p.x(), tmp_h_f.p.y(), 0.0))
        else:
            human_f = kdl.Frame(kdl.Rotation.RPY(0.0, 0.0, yaw), kdl.Vector(tmp_h_f.p.x(), tmp_h_f.p.y(), 0.0))
            self.cached_yaw = yaw

        t = self.kdl_to_transform(human_f)
        t.header = self.human_pose_msg.header
        t.child_frame_id = self.human_frame

        return t

    def execute_relloc(self, goal):
        t = self.calculate_pose()
        self.cached_tf = t

        if not t:
            self.mocap_relloc_server.preempt_request = True
            # self.mocap_relloc_server.set_aborted(text = 'Residual error is too high: {}'.format(res_err))
            self.mocap_relloc_server.set_aborted()
            return

        if self.mocap_relloc_server.is_preempt_requested():
            self.mocap_relloc_server.set_preempted()
            rospy.logwarn('Relloc action has been preempted')

        self.mocap_relloc_server.set_succeeded(result=EmptyResult())

    def run(self):
        loop_rate = rospy.Rate(self.publish_rate)

        while not rospy.is_shutdown():
            try:
                now = rospy.Time.now()
                if self.cached_tf:
                    if self.tf_exp_time > rospy.Duration(sys.float_info.epsilon):
                        self.tf_expired = now > (self.cached_tf.header.stamp + self.tf_exp_time)
                    else:
                        self.tf_expired = False

                    if not self.tf_expired:
                        # t = copy.deepcopy(self.cached_tf)
                        t = self.calculate_pose(ignore_heading = True)
                        # Update stamp to keep tf alive
                        t.header.stamp = now
                        self.tf_br.sendTransform(t)

                loop_rate.sleep()

            except rospy.ROSException, e:
                if e.message == 'ROS time moved backwards':
                    rospy.logwarn("Saw a negative time change. Resetting internal state...")
                    self.reset_state()

if __name__ == '__main__':
    rospy.init_node('mocap_relloc')

    mocap_relloc = MocapRellocNode()

    try:
        mocap_relloc.run()
    except rospy.ROSInterruptException:
        rospy.logdebug('Exiting')
        pass
