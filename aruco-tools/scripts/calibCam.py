#!/usr/bin/env python

import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from tf.transformations import quaternion_about_axis, quaternion_matrix, quaternion_from_matrix
import tf2_ros
from tf2_geometry_msgs import PoseStamped
from geometry_msgs.msg import TransformStamped, Transform, Vector3, Pose, Point, Quaternion
from std_msgs.msg import Header
import yaml
import argparse
import os

class ArucoDetector:
    def __init__(self, image_topic, camera_info_topic, marker_length, camera_frame_id, camera_base_id, yaml_file, pub_live):
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.poses = {}

        self.marker_length = marker_length
        self.camera_frame_id = camera_frame_id
        self.camera_base_id = camera_base_id
        self.yaml_file = yaml_file
        self.pub_live = pub_live
        if(self.pub_live):
            self.broadcaster = tf2_ros.StaticTransformBroadcaster()

        # Subscribers
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback)
        #NOTE: Adjust these two parameters for your printed marker
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        self.parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)
        self.marker_points = np.array([[-self.marker_length / 2, self.marker_length / 2, 0],
                              [self.marker_length / 2, self.marker_length / 2, 0],
                              [self.marker_length / 2, -self.marker_length / 2, 0],
                              [-self.marker_length / 2, -self.marker_length / 2, 0]], dtype=np.float32)

        # TF Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.load_existing_poses()

    def camera_info_callback(self, data):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(data.K).reshape(3, 3)
            self.dist_coeffs = np.array(data.D)

    def image_callback(self, data):
        if self.camera_matrix is None or self.dist_coeffs is None:
            rospy.loginfo("Waiting for camera calibration data...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)
            aruco.drawDetectedMarkers(cv_image, corners, ids)
            
            if np.all(ids is not None):
                for i in range(0, len(ids)):
                    _, rvec, tvec = cv2.solvePnP(self.marker_points, corners[i], self.camera_matrix, self.dist_coeffs, None, None, False, cv2.SOLVEPNP_IPPE_SQUARE)
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)

                    # Construct the transformation matrix
                    axis = rvec.flatten() / np.linalg.norm(rvec)
                    angle = np.linalg.norm(rvec)
                    q = quaternion_about_axis(angle, axis)
                    tvec = tvec.flatten()

                    # Create the transform for the detected marker
                    tag_pose = PoseStamped(Header(stamp=rospy.Time.now(), frame_id=self.camera_frame_id),
                                            Pose(Point(tvec[0], tvec[1], tvec[2]), Quaternion(q[0], q[1], q[2], q[3])))
                    
                    # Transform to the base_link frame
                    try:
                        tag_cam_base = self.tf_buffer.transform(tag_pose, self.camera_base_id, rospy.Duration(1))
                        q = np.array([tag_cam_base.pose.orientation.x, tag_cam_base.pose.orientation.y, tag_cam_base.pose.orientation.z, tag_cam_base.pose.orientation.w])
                        t_mat = quaternion_matrix(q)
                        t_mat[:3,3] = [tag_cam_base.pose.position.x, tag_cam_base.pose.position.y, tag_cam_base.pose.position.z]
                        t_mat = np.linalg.inv(t_mat)
                        cam_pose = PoseStamped(Header(stamp=rospy.Time.now(), frame_id=f"aruco_{ids[i][0]}"),
                                               Pose(Point(*t_mat[:3,3]),Quaternion(*quaternion_from_matrix(t_mat))))
                        
                        cam_pose_base = self.tf_buffer.transform(cam_pose, "base_link", rospy.Duration(1))
                        if self.pub_live:
                            transform = TransformStamped(cam_pose_base.header, self.camera_base_id,
                                                         Transform(Vector3(cam_pose_base.pose.position.x, cam_pose_base.pose.position.y, cam_pose_base.pose.position.z), 
                                                                   cam_pose_base.pose.orientation))
                            self.broadcaster.sendTransform(transform)
                        else:
                            self.poses[self.camera_base_id] = {
                                'parent': 'base_link',
                                'position': {
                                    'x': cam_pose_base.pose.position.x,
                                    'y': cam_pose_base.pose.position.y,
                                    'z': cam_pose_base.pose.position.z
                                },
                                'orientation': {
                                    'x': cam_pose_base.pose.orientation.x,
                                    'y': cam_pose_base.pose.orientation.y,
                                    'z': cam_pose_base.pose.orientation.z,
                                    'w': cam_pose_base.pose.orientation.w
                                }
                            }
                        # rospy.loginfo(f"Pose for cam: {self.poses[self.camera_base_id]}")
                        rospy.loginfo_once("Aruco detected!")
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                        rospy.logerr(f"Failed to transform to base_link: {e}")

            # cv2.imshow("Live Image", cv_image)
            # cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(e)

    def load_existing_poses(self):
        if os.path.exists(self.yaml_file):
            with open(self.yaml_file, 'r') as file:
                existing_poses = yaml.safe_load(file)
                if existing_poses is not None:
                    self.poses.update(existing_poses)
                    rospy.loginfo(f"Loaded existing poses from {self.yaml_file}")

    def save_poses_to_yaml(self):
        with open(self.yaml_file, 'w') as file:
            yaml.dump(self.poses, file, default_flow_style=False)
        rospy.loginfo(f"Saved poses to {self.yaml_file}")
        rospy.loginfo(self.poses)

def main():
    import sys
    rospy.init_node('aruco_detector', anonymous=True)

    parser = argparse.ArgumentParser(description='Aruco Detector and Pose Saver')
    parser.add_argument('image_topic', type=str, help='The image topic')
    parser.add_argument('camera_info_topic', type=str, help='The camera info topic')
    parser.add_argument('marker_length', type=float, help='The marker length')
    parser.add_argument('camera_frame_id', type=str, help='The camera frame ID')
    parser.add_argument('camera_base_id', type=str, help='The camera frame ID')
    parser.add_argument('yaml_file', type=str, help='The YAML file to save poses')
    parser.add_argument('pub_live', type=bool, help='Publish live pose of cam instead of storing it')

    args, _ = parser.parse_known_args()
    args.pub_live= args.pub_live=="True"

    aruco_detector = ArucoDetector(args.image_topic, args.camera_info_topic, args.marker_length, args.camera_frame_id, args.camera_base_id, args.yaml_file, args.pub_live)

    # Wait for user input to save the poses to a YAML file
    if(not args.pub_live):
        rospy.loginfo("Press Ctrl+C to save the pose to a YAML file.")
    rospy.spin()

    # Save the poses to a YAML file when the node is shut down
    if(not args.pub_live):
        aruco_detector.save_poses_to_yaml()

if __name__ == '__main__':
    main()
