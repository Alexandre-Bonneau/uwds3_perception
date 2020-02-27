#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
import numpy as np
from tf.transformations import translation_matrix, quaternion_matrix
from tf.transformations import translation_from_matrix, quaternion_from_matrix
from visualization_msgs.msg import Marker, MarkerArray
from uwds3_msgs.msg import SceneNode, PrimitiveShape, SceneChangesStamped
from geometry_msgs.msg import Vector3
from std_msgs.msg import ColorRGBA


class ShapeVisualizerNode(object):
    def __init__(self):
        self.tracks_topic = rospy.get_param("~tracks_topic", "tracks")
        self.marker_id_map = {}
        self.tracks = {}
        self.track_marker = {}
        self.last_marker_id = 0
        self.markers_publisher = rospy.Publisher(self.tracks_topic+"_viz", MarkerArray, queue_size=1)
        self.track_subscriber = rospy.Subscriber(self.tracks_topic, SceneChangesStamped, self.observation_callback, queue_size=1)
        rospy.loginfo("[visualizer] Shape visualizer ready !")

    def observation_callback(self, tracks_msg):
        markers_msg = MarkerArray()
        self.last_marker_id = 0
        for track in tracks_msg.changes.nodes:
            if track.has_shape is True:
                for shape_idx, shape in enumerate(track.shapes):
                    marker = Marker()
                    self.last_marker_id += 1
                    marker.id = self.last_marker_id
                    marker.ns = track.id
                    marker.action = Marker.MODIFY
                    marker.header = tracks_msg.header

                    position = [track.pose_stamped.pose.pose.position.x,
                                track.pose_stamped.pose.pose.position.y,
                                track.pose_stamped.pose.pose.position.z]
                    orientation = [track.pose_stamped.pose.pose.orientation.x,
                                   track.pose_stamped.pose.pose.orientation.y,
                                   track.pose_stamped.pose.pose.orientation.z,
                                   track.pose_stamped.pose.pose.orientation.w]
                    world_transform = np.dot(translation_matrix(position), quaternion_matrix(orientation))
                    position = [shape.pose.position.x,
                                shape.pose.position.y,
                                shape.pose.position.z]
                    orientation = [shape.pose.orientation.x,
                                   shape.pose.orientation.y,
                                   shape.pose.orientation.z,
                                   shape.pose.orientation.w]
                    shape_transform = np.dot(translation_matrix(position), quaternion_matrix(orientation))
                    shape_world_transform = np.dot(world_transform, shape_transform)
                    position = translation_from_matrix(shape_world_transform)
                    orientation = quaternion_from_matrix(shape_world_transform)

                    marker.pose.position.x = position[0]
                    marker.pose.position.y = position[1]
                    marker.pose.position.z = position[2]

                    marker.pose.orientation.x = orientation[0]
                    marker.pose.orientation.y = orientation[1]
                    marker.pose.orientation.z = orientation[2]
                    marker.pose.orientation.w = orientation[3]

                    if shape.type == PrimitiveShape.CYLINDER:
                        marker.type = Marker.CYLINDER
                        marker.scale = Vector3(x=shape.dimensions[0],
                                               y=shape.dimensions[0],
                                               z=shape.dimensions[1])
                    elif shape.type == PrimitiveShape.SPHERE:
                        marker.type = Marker.SPHERE
                        marker.scale = Vector3(x=shape.dimensions[0],
                                               y=shape.dimensions[0],
                                               z=shape.dimensions[0])
                    elif shape.type == PrimitiveShape.BOX:
                        marker.type = Marker.CUBE
                        marker.scale = Vector3(x=shape.dimensions[0],
                                               y=shape.dimensions[1],
                                               z=shape.dimensions[2])
                    elif shape.type == PrimitiveShape.MESH:
                        if shape.mesh_resource != "":
                            marker.type = Marker.MESH_RESOURCE
                            marker.mesh_resource = shape.mesh_resource
                            marker.mesh_use_embedded_materials = True
                        else:
                            marker.type = Marker.TRIANGLE_LIST
                            marker.points = shape.vertices
                        marker.scale = Vector3(x=1.0, y=1.0, z=1.0)
                    else:
                        raise NotImplementedError("Shape not implemented")
                    marker.color = shape.color
                    marker.color.a = 0.75
                    marker.lifetime = rospy.Duration(0.25)
                    markers_msg.markers.append(marker)
        self.markers_publisher.publish(markers_msg)

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()

if __name__ == '__main__':
    rospy.init_node("shape_visualizer", anonymous=False)
    ros_node = ShapeVisualizerNode().run()
