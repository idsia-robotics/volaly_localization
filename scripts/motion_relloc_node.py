#!/usr/bin/env python

import collections
import copy
import threading

import numpy as np

import rospy
import rostopic

import actionlib
from std_msgs.msg import Bool, Duration
import tf2_ros
import tf2_geometry_msgs
import tf_conversions as tfc
import PyKDL as kdl

from geometry_msgs.msg import Pose, PoseStamped, TransformStamped

from volaly_msgs.msg import MotionRellocAction, MotionRellocFeedback, MotionRellocResult
from volaly_msgs.msg import MotionRellocContAction, MotionRellocContFeedback, MotionRellocContResult
from volaly_msgs.msg import GoToAction, GoToGoal
from volaly_msgs.msg import FollowShapeAction, FollowShapeGoal
from volaly_msgs.msg import FollowMeAction, FollowMeGoal

from volaly_localization import relloclib

class MotionRellocNode:
    def __init__(self):
        action_ns = rospy.get_param('~action_ns', '')
        robot_name = rospy.get_param('~robot_name')
        self.lock = threading.RLock()

        self.publish_rate = rospy.get_param('~publish_rate', 50)

        sec = rospy.get_param('~tf_exp_time', 90.0)
        self.tf_exp_time = rospy.Duration(sec)

        self.human_frame = rospy.get_param('human_frame_id', 'human_footprint')
        self.robot_root_frame = rospy.get_param('robot_root_frame', robot_name + '/odom')
        drone_goto_action_ns = rospy.get_param('~drone_goto_action_ns', '/' + robot_name + '/goto_action')
        drone_shape_action_ns = rospy.get_param('~drone_followpath_action_ns', '/' + robot_name + '/followshape_action')
        drone_followme_action_ns = rospy.get_param('~drone_followme_action_ns', '/' + robot_name + '/followme_action')

        self.ray_origin_frame = rospy.get_param('ray_origin_frame', 'eyes')
        self.ray_direction_frame = rospy.get_param('ray_direction_frame', 'pointer')
        self.ray_inverse = rospy.get_param('ray_inverse', False)

        pose_topic = rospy.get_param('~robot_pose_topic', '/' + robot_name + '/odom/pose/pose')
        pose_topic_class, pose_real_topic, pose_eval_func = rostopic.get_topic_class(pose_topic)
        self.robot_pose_msg_eval = pose_eval_func

        ############ FIXME ############
        trigger_topic = rospy.get_param('~trigger_topic', '/' + robot_name + '/joy/buttons[6]')
        trigger_topic_class, trigger_real_topic, trigger_eval_func = rostopic.get_topic_class(trigger_topic)
        self.trigger_msg_eval = trigger_eval_func
        # self.trigger_sub = rospy.Subscriber(trigger_real_topic, trigger_topic_class, self.trigger_cb)
        self.trigger_val = None
        self.last_trigger_val = None
        ###############################

        self.timewindow = rospy.get_param('~timewindow', 5.0)
        self.sync_freq = rospy.get_param('~freq', 20.0)
        self.sample_size = rospy.get_param('~sample_size', self.sync_freq * 3.0)

        self.residual_threshold = np.radians(rospy.get_param('~residual_threshold_deg', 3.0))

        self.robot_motion_span_min = rospy.get_param('~robot_motion_span_min', 0.20) # 20 cm

        if self.timewindow and self.sync_freq:
            self.queue_size = int(self.timewindow * self.sync_freq); # 5.0 seconds at 5 Hz
            rospy.loginfo('Max queue size: {}'.format(self.queue_size))
            if self.sample_size > self.queue_size:
                rospy.loginfo('sample_size [{}] is bigger than queue_size [{}]. Setting sample_size = queue_size'.format(self.sample_size, self.queue_size))
                self.sample_size = self.queue_size
        else:
            rospy.logwarn('Either timewindow or queue_size is set to 0. Using unbound queue.')
            self.queue_size = None
        self.deque = collections.deque(maxlen = self.queue_size)

        self.relloc_deque = collections.deque(maxlen = self.sync_freq * 1.0) # Use 1s of relloc data to trigger selection

        self.robot_pose_msg = None
        self.robot_sub = rospy.Subscriber(pose_real_topic, pose_topic_class, self.robot_pose_cb)

        self.is_valid_pub = rospy.Publisher('is_relloc_valid', Bool, queue_size = 10)

        self.initial_guess = np.array([0, 0, 0, 0])
        self.reset_state()

        self.tf_buff = tf2_ros.Buffer()
        self.tf_ls = tf2_ros.TransformListener(self.tf_buff)
        self.tf_br = tf2_ros.TransformBroadcaster()

        self.drone_goto_client = actionlib.SimpleActionClient(drone_goto_action_ns, GoToAction)
        rospy.loginfo('Waiting for ' + drone_goto_action_ns)
        self.drone_goto_client.wait_for_server()

        # self.drone_shape_client = actionlib.SimpleActionClient(drone_shape_action_ns, FollowShapeAction)
        # rospy.loginfo('Waiting for ' + drone_shape_action_ns)
        # self.drone_shape_client.wait_for_server()

        self.drone_followme_client = actionlib.SimpleActionClient(drone_followme_action_ns, FollowMeAction)
        rospy.loginfo('Waiting for ' + drone_followme_action_ns)
        self.drone_followme_client.wait_for_server()

        self.relloc_server = actionlib.SimpleActionServer(action_ns + '/relloc_action', MotionRellocContAction, self.execute_relloc, False)
        self.relloc_server.start()

        self.relloc_cont_server = actionlib.SimpleActionServer(action_ns + '/relloc_cont_action', MotionRellocContAction, self.execute_relloc_cont, False)
        self.relloc_cont_server.start()

    def reset_state(self):
        with self.lock:
            self.deque.clear()
            self.estimated_tf = self.initial_guess
            self.cached_tf = None
            self.estimated = False
            self.tf_expired = True

    def robot_pose_cb(self, msg):
        self.robot_pose_msg = msg
        self.topic_sync_cb(self.robot_pose_msg)

    def execute_relloc(self, goal):
        loop_rate = rospy.Rate(10) # 50 Hz

        if not self.robot_pose_msg:
            self.relloc_server.preempt_request = True
            self.relloc_server.set_aborted()
            return

        if self.relloc_cont_server.is_active():
            self.relloc_cont_server.preempt_request = True
            while not rospy.is_shutdown() or self.relloc_cont_server.is_active():
                loop_rate.sleep()

        self.trigger_cb(True)
        start_stamp = rospy.Time.now()

        time_thresh = rospy.Duration(3.0)
        res_thresh = np.radians(10.0)

        ts = None

        self.deque.clear()

        while not rospy.is_shutdown():
            if self.relloc_server.is_preempt_requested():
                # Trigger: True -> False (resets state)
                self.trigger_cb(False)

                self.relloc_server.set_preempted()
                rospy.logwarn('Relloc action has been preempted')
                break

            # Copy the queue
            cur_deque = list(self.deque)
            cur_length = len(cur_deque)

            # Do we have enough data yet?
            if cur_length < self.queue_size:
                loop_rate.sleep()
                continue

            # Take last 3 seconds out of 5
            cur_deque = cur_deque[-self.sample_size : ]
            t, res_err = self.estimate_pose(cur_deque, constraints = False)

            self.cached_tf = t

            pose = PoseStamped()
            pose.header = t.header
            pose.pose.position.x = t.transform.translation.x
            pose.pose.position.y = t.transform.translation.y
            pose.pose.position.z = t.transform.translation.z
            pose.pose.orientation.x = t.transform.rotation.x
            pose.pose.orientation.y = t.transform.rotation.y
            pose.pose.orientation.z = t.transform.rotation.z
            pose.pose.orientation.w = t.transform.rotation.w

            feedback = MotionRellocContFeedback()
            feedback.estimate = pose
            feedback.residual_error = res_err # in rad
            self.relloc_server.publish_feedback(feedback)

            if res_err < res_thresh:
                # Trigger
                self.relloc_server.set_succeeded(result=MotionRellocContResult(pose))
                break
            else:
                # Trigger: True -> False (resets state)
                self.trigger_cb(False)

                rospy.logwarn('Residual error is too high: {}'.format(res_err))

                self.relloc_server.set_aborted(text = 'Residual error is too high: {}'.format(res_err))
                return

    def execute_relloc_cont(self, goal):
        loop_rate = rospy.Rate(10) # 50 Hz

        if not self.robot_pose_msg:
            self.relloc_cont_server.preempt_request = True
            self.relloc_cont_server.set_aborted()
            return

        if self.relloc_server.is_active():
            self.relloc_server.preempt_request = True
            while self.relloc_server.is_active():
                loop_rate.sleep()

        self.trigger_cb(True)
        start_stamp = rospy.Time.now()

        time_thresh = rospy.Duration(3.0)
        # res_thresh = 0.05
        res_thresh = np.radians(10.0)

        ts = None

        self.deque.clear()

        while not rospy.is_shutdown():
            if self.relloc_cont_server.is_preempt_requested():
                # Trigger: True -> False (resets state)
                self.trigger_cb(False)

                self.relloc_cont_server.set_preempted()
                rospy.logwarn('RellocCont action has been preempted')
                break

            # Copy the queue
            cur_deque = list(self.deque)
            cur_length = len(cur_deque)

            # Do we have enough data yet?
            if cur_length < self.sample_size:
                continue

            t, res_err = self.estimate_pose(cur_deque)

            # if not (t and res_err):
            #     continue

            self.cached_tf = t

            pose = PoseStamped()
            pose.header = t.header
            pose.pose.position.x = t.transform.translation.x
            pose.pose.position.y = t.transform.translation.y
            pose.pose.position.z = t.transform.translation.z
            pose.pose.orientation.x = t.transform.rotation.x
            pose.pose.orientation.y = t.transform.rotation.y
            pose.pose.orientation.z = t.transform.rotation.z
            pose.pose.orientation.w = t.transform.rotation.w

            if res_err < res_thresh:
                if ts is None:
                    ts = rospy.Time.now()
                if rospy.Time.now() > ts + time_thresh:
                    # rospy.loginfo('Estimated pose: {}'.format(pose))

                    # Trigger
                    self.relloc_cont_server.set_succeeded(result=MotionRellocContResult(pose))
                    break
            else:
                ts = None


            feedback = MotionRellocContFeedback()
            feedback.estimate = pose
            feedback.residual_error = res_err # in rad
            self.relloc_cont_server.publish_feedback(feedback)

            loop_rate.sleep()

    def trigger_cb(self, msg):
        self.last_trigger_val = self.trigger_val
        self.trigger_val = msg

        # if self.trigger_msg_eval:
        #     self.trigger_val = True if self.trigger_msg_eval(msg) else False
        # else:
        #     self.trigger_val = False

        if self.last_trigger_val != self.trigger_val:
            # rospy.logwarn(self.trigger_val)
            pass

        if not self.last_trigger_val and self.trigger_val:
            rospy.loginfo('Collecting new data')
            self.reset_state()

    def estimate_pose(self, pts, constraints = True):
        ########### FIXME ##############
        np.random.seed(0)
        ################################

        data = np.array(pts)

        sample_size = self.sample_size
        if sample_size > data.shape[0]:
            sample_size = data.shape[0]

        idx = np.random.choice(data.shape[0], size=sample_size, replace=False)
        tmp = data[idx, :]
        p = np.transpose(tmp[:, 0:3])
        qc = np.transpose(tmp[:, 3:6])
        qv = np.transpose(tmp[:, 6:9])

        # rospy.loginfo('{} {} {}'.format(p, qc, qv))

        max_p = np.max(p, axis=1)
        min_p = np.min(p, axis=1)
        motion_span = np.linalg.norm(max_p - min_p)
        # if motion_span > self.robot_motion_span_min:
        #     return None, None
        #     # rospy.logwarn('Robot motion span: %3.4f', motion_span)

        rospy.loginfo('Estimating pose. Using {} of total {} data points'.format(sample_size, data.shape[0]))

        if constraints:
            res, maxerr = relloclib.estimate_pose(p, qc, qv, self.estimated_tf)
        else:
            res, maxerr = relloclib.estimate_pose_no_constraints(p, qc, qv, self.estimated_tf)
        self.estimated_tf = res.x

        rospy.loginfo("Average angular error (residual) in deg: {:.2f}".format(np.rad2deg(res.fun)))
        rospy.loginfo("Maximum angular error in deg: {:.2f}".format(np.rad2deg(maxerr)))
        rospy.loginfo("Recovered transform (tx, ty, tz, rotz): {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(
                    res.x[0],
                    res.x[1],
                    res.x[2],
                    np.rad2deg(res.x[3])))

        est_quat = kdl.Rotation.RPY(0.0, 0.0, res.x[3]).GetQuaternion()
        est_tran = res.x[:3]

        t = TransformStamped()

        t.header.frame_id = self.human_frame
        t.child_frame_id = self.robot_root_frame
        t.header.stamp = rospy.Time.now()
        t.transform.translation.x = est_tran[0]
        t.transform.translation.y = est_tran[1]
        t.transform.translation.z = est_tran[2]
        t.transform.rotation.x = est_quat[0]
        t.transform.rotation.y = est_quat[1]
        t.transform.rotation.z = est_quat[2]
        t.transform.rotation.w = est_quat[3]

        self.estimated = True

        return t, res.fun
        # return t, maxerr

    def topic_sync_cb(self, robot_pose_msg):
        if not self.trigger_val:
            return

        if self.robot_pose_msg_eval:
            # Extract relevant fields from the message
            rpose = self.robot_pose_msg_eval(robot_pose_msg)
        else:
            rpose = robot_pose_msg

        robot_pos = None

        if isinstance(rpose, Pose):
            robot_pos = (rpose.position.x, rpose.position.y, rpose.position.z)
        else:
            rospy.logerr('Wrong topic/field type for robot pose: {}/{}. Should be geometry_msgs/Pose'
                .format(type(rpose).__module__.split('.')[0], type(rpose).__name__))
            return

        try:
            origin_tf = self.tf_buff.lookup_transform(self.human_frame, self.ray_origin_frame, rospy.Time())#robot_pose_msg.header.stamp)
            ray_tf = self.tf_buff.lookup_transform(self.human_frame, self.ray_direction_frame, rospy.Time())#robot_pose_msg.header.stamp)
            if self.ray_inverse:
                unit_vector = kdl.Vector(-1.0, 0.0, 0.0)
            else:
                unit_vector = kdl.Vector(1.0, 0.0, 0.0)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException), e:
            rospy.logerr(e)
            return

        orig = (origin_tf.transform.translation.x,
                origin_tf.transform.translation.y,
                origin_tf.transform.translation.z)

        quat = (ray_tf.transform.rotation.x,
                ray_tf.transform.rotation.y,
                ray_tf.transform.rotation.z,
                ray_tf.transform.rotation.w)

        frame = tfc.fromTf((orig, quat))

        p = list(robot_pos)
        q = [frame.p.x(), frame.p.y(), frame.p.z()]
        # Rotate unit vector in the direction of pointing
        v = frame.M * unit_vector

        q.extend([v.x(), v.y(), v.z()])
        p.extend(q)

        with self.lock:
            self.deque.append(p)

    def run(self):
        loop_rate = rospy.Rate(self.publish_rate)

        while not rospy.is_shutdown():
            try:
                now = rospy.Time.now()
                if self.cached_tf:
                    self.tf_expired = now > (self.cached_tf.header.stamp + self.tf_exp_time)

                    if not self.tf_expired:
                        t = copy.deepcopy(self.cached_tf)
                        # Update stamp to keep tf alive
                        t.header.stamp = now
                        self.tf_br.sendTransform(t)

                self.is_valid_pub.publish(not self.tf_expired)

                loop_rate.sleep()

            except rospy.ROSException, e:
                if e.message == 'ROS time moved backwards':
                    rospy.logwarn("Saw a negative time change. Resetting internal state...")
                    self.reset_state()

if __name__ == '__main__':
    rospy.init_node('motion_relloc')

    motion_relloc = MotionRellocNode()

    try:
        motion_relloc.run()
    except rospy.ROSInterruptException:
        rospy.logdebug('Exiting')
        pass
