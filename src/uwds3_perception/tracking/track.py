import rospy
import cv2
import numpy as np
import uuid
import uwds3_msgs
import rospy
from tf.transformations import euler_matrix
from .single_object_tracker import SingleObjectTracker
from pyuwds3.types.scene_node import SceneNode
from pyuwds3.types.bbox_stable import BoundingBoxStable
from pyuwds3.types.camera import HumanCamera
from pyuwds3.types.vector.vector6d import Vector6D
from pyuwds3.types.vector.vector6d_stable import Vector6DStable


class TrackState:
    """Represents the track states"""
    TENTATIVE = 1
    CONFIRMED = 2
    OCCLUDED = 3
    DELETED = 4


class Track(SceneNode):
    """Represents a track in both image and world space"""

    def __init__(self,
                 detection,
                 n_init,
                 max_disappeared,
                 max_age):
        """Track constructor"""

        self.n_init = n_init
        self.max_disappeared = max_disappeared
        self.max_age = max_age

        self.hits = 1
        self.age = 1

        self.id = str(uuid.uuid4()).replace("-", "")

        self.bbox = BoundingBoxStable(detection.bbox.xmin,
                                      detection.bbox.ymin,
                                      detection.bbox.xmax,
                                      detection.bbox.ymax)

        self.label = detection.label

        if self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED
        else:
            self.state = TrackState.TENTATIVE

        self.pose = None

        self.shapes = []

        self.tracker = SingleObjectTracker()

        if self.label == "face":
            self.camera = HumanCamera()
        else:
            self.camera = None

        self.features = detection.features

        self.expiration_duration = 1.0

    def update(self, detection):
        """Updates the track's bbox"""
        self.bbox.update(detection.bbox.xmin,
                         detection.bbox.ymin,
                         detection.bbox.xmax,
                         detection.bbox.ymax,
                         depth=detection.bbox.depth)
        for name, features in detection.features.items():
            self.features[name] = features
        self.age = 0
        self.hits += 1
        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED
        if self.state == TrackState.OCCLUDED:
            self.state = TrackState.CONFIRMED

    def update_pose(self, position, rotation=None):
        """ """
        if self.pose is None:
            if rotation is None:
                self.pose = Vector6DStable(x=position.x,
                                           y=position.y,
                                           z=position.z)
            else:
                self.pose = Vector6DStable(x=position.x,
                                           y=position.y,
                                           z=position.z,
                                           rx=rotation.x,
                                           ry=rotation.y,
                                           rz=rotation.z)
        else:
            self.pose.pos.update(position.x, position.y, position.z)
            if rotation is not None:
                self.pose.rot.update(rotation.x, rotation.y, rotation.z)

    def predict_bbox(self):
        """Predict the bbox location based on motion model (kalman tracker)"""
        self.bbox.predict()

    def mark_missed(self):
        """Mark the track missed"""
        self.age += 1
        if self.state == TrackState.TENTATIVE:
            if self.age > self.n_init:
                self.state = TrackState.DELETED
        if self.state == TrackState.CONFIRMED:
            if self.age > self.max_disappeared:
                self.state = TrackState.OCCLUDED
        elif self.state == TrackState.OCCLUDED:
            if self.age > self.max_age:
                self.state = TrackState.DELETED

    def is_perceived(self):
        """Returns True if the track is perceived"""
        if not self.to_delete():
            return self.state != TrackState.OCCLUDED
        else:
            return False

    def is_confirmed(self):
        """Returns True if the track is confirmed"""
        return self.state == TrackState.CONFIRMED

    def is_occluded(self):
        """Returns True if the track is occluded"""
        return self.state == TrackState.OCCLUDED

    def to_delete(self):
        """Returns True if the track is deleted"""
        return self.state == TrackState.DELETED

    def is_tentative(self):
        return self.state == TrackState.TENTATIVE

    # def project_into(self, camera_track):
    #     """Returns the 2D bbox in the given camera plane"""
    #     if self.is_located() and camera_track.is_located():
    #         if self.has_shape() and camera_track.has_camera():
    #             success, tf_sensor = camera_track.transform()
    #             success, tf_track = self.pose.transform()
    #             tf_project = np.dot(np.linalg.inv(tf_sensor), tf_track)
    #             camera_matrix = camera_track.camera.camera_matrix()
    #             fx = camera_matrix[0][0]
    #             fy = camera_matrix[1][1]
    #             cx, cy = camera_track.camera.center()
    #             z = tf_project[2]
    #             w = (self.shape.width() * fx/z)
    #             h = (self.shape.height() * fy/z)
    #             x = (tf_project[0] * fx/z)+cx
    #             y = (tf_project[1] * fy/z)+cy
    #             xmin = x - w/2.0
    #             ymin = y - h/2.0
    #             xmax = x + w/2.0
    #             ymax = y + h/2.0
    #             if xmax < 0:
    #                 return False, None
    #             if ymax < 0:
    #                 return False, None
    #             if xmin > camera_track.camera.width:
    #                 return False, None
    #             if ymin > camera_track.camera.height:
    #                 return False, None
    #             if xmin < 0:
    #                 xmin = 0
    #             if ymin < 0:
    #                 ymin = 0
    #             if xmax > camera_track.camera.width:
    #                 xmax = camera_track.camera.width
    #             if ymax > camera_track.camera.height:
    #                 ymax = camera_track.camera.height
    #             return True, BoundingBox(xmin, ymin, xmax, ymax)
    #     return False, None
