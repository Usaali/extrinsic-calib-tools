#!/usr/bin/env python

import rospy
import tf2_ros
import yaml
from geometry_msgs.msg import TransformStamped
import argparse

def load_poses_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

def publish_static_transforms(poses):
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    transforms = []

    for tag_id, pose in poses.items():
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = pose['parent']
        transform.child_frame_id = f"{tag_id}"

        transform.transform.translation.x = pose['position']['x']
        transform.transform.translation.y = pose['position']['y']
        transform.transform.translation.z = pose['position']['z']

        transform.transform.rotation.x = pose['orientation']['x']
        transform.transform.rotation.y = pose['orientation']['y']
        transform.transform.rotation.z = pose['orientation']['z']
        transform.transform.rotation.w = pose['orientation']['w']

        transforms.append(transform)

    broadcaster.sendTransform(transforms)

if __name__ == '__main__':
    rospy.init_node('aruco-pose-publisher')

    parser = argparse.ArgumentParser(description='Aruco Pose Publisher')
    parser.add_argument('yaml_file', type=str, help='The YAML file to save poses')
    args, _ = parser.parse_known_args()

    try:
        poses = load_poses_from_yaml(args.yaml_file)
        publish_static_transforms(poses)
        rospy.loginfo("Published static transforms for ArUco tags.")
    except Exception as e:
        rospy.logerr(f"Failed to publish static transforms: {e}")

    rospy.spin()
