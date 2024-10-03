#!/usr/bin/env python3

import rospy
import tf2_ros
import tf_conversions
import cv2
import cv2.aruco as aruco
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped

class CharucoDiamondDetector:
    def __init__(self):
        rospy.init_node('charuco_diamond_detector', anonymous=True)
        
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Subscribe to camera image and camera info topics
        self.image_sub = rospy.Subscriber("/rgb/image_raw", Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber("/rgb/camera_info", CameraInfo, self.camera_info_callback)
        
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
        
        # Placeholder for camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None

    def camera_info_callback(self, data):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(data.K).reshape(3, 3)
        if self.dist_coeffs is None:
            self.dist_coeffs = np.array(data.D)

    def image_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, self.dictionary)
        
        if len(corners) > 0:
            aruco.drawDetectedMarkers(cv_image, corners, ids)
            # Parameters for diamond detection: square marker length and marker ids
            diamondCorners, diamondIds = aruco.detectCharucoDiamond(gray, corners, ids, 0.03625/0.02721)
            if diamondIds is not None:
                # Assuming single diamond detection, for multiple diamonds, iterate through diamondCorners and diamondIds
                aruco.drawDetectedDiamonds(cv_image, diamondCorners, diamondIds)
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(diamondCorners, 0.03625, self.camera_matrix, self.dist_coeffs)  # Adjust marker size
                if rvec is not None and tvec is not None:
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec[0], tvec[0], 0.1)  # Adjust axis length as needed
                    self.broadcast_tf(rvec[0], tvec[0])
        
        cv2.imshow("Charuco Diamond", cv2.resize(cv_image,(1280,720)))
        cv2.waitKey(3)
    
    def broadcast_tf(self, rvec, tvec):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "rgb_camera_link"
        t.child_frame_id = "charuco_diamond"
        
        # Convert rvec (Rodrigues) to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Convert rotation matrix to quaternion
        quaternion = tf_conversions.transformations.quaternion_from_matrix(np.vstack((np.hstack((rotation_matrix, [[0], [0], [0]])), [0, 0, 0, 1])))
        
        t.transform.translation.x = tvec[0][0]
        t.transform.translation.y = tvec[0][1]
        t.transform.translation.z = tvec[0][2]
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]
        
        self.tf_broadcaster.sendTransform(t)

if __name__ == '__main__':
    try:
        detector = CharucoDiamondDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
