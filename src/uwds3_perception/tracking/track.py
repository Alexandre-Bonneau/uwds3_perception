import rospy
import cv2
import numpy as np
import uuid
import uwds3_msgs
from uwds3_perception.types.features import Features
from uwds3_perception.types.bbox import BoundingBox, BoundingBoxStabilized
from uwds3_perception.types.camera import Camera, HumanCamera
from uwds3_perception.types.vector import Vector6DStabilized
from tf.transformations import translation_matrix, euler_matrix
from tf.transformations import translation_from_matrix, quaternion_from_matrix
from .single_object_tracker import SingleObjectTracker


class TrackState:
    """Represents the track states"""

    TENTATIVE = 1
    CONFIRMED = 2
    OCCLUDED = 3
    DELETED = 4


class Track(object):
    """Represents a track in both image and world space"""

    def __init__(self,
                 detection,
                 n_init,
                 max_disappeared,
                 max_age,
                 tracker_type):
        """Track constructor"""

        self.n_init = n_init
        self.max_disappeared = max_disappeared
        self.max_age = max_age

        self.hits = 1
        self.age = 1

        self.uuid = str(uuid.uuid4()).replace("-", "")
        self.bbox = detection.bbox
        self.bbox = BoundingBoxStabilized(detection.bbox.xmin,
                                          detection.bbox.ymin,
                                          detection.bbox.xmax,
                                          detection.bbox.ymax)
        self.label = detection.label
        self.state = TrackState.TENTATIVE

        self.tracker = None
        self.filter = None

        self.tracking_features = None

        self.pose = None

        self.shape = None

        if self.label == "face":
            self.camera = HumanCamera()
        else:
            self.camera = None

        self.features = detection.features
        if tracker_type is not None:
            self.tracker = SingleObjectTracker(tracker_type)

    def update(self, detection):
        """Updates the track's bbox"""
        self.bbox.update(detection.bbox.xmin,
                         detection.bbox.ymin,
                         detection.bbox.xmax,
                         detection.bbox.ymax)
        self.features = detection.features
        self.age = 0
        self.hits += 1
        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED
        if self.state == TrackState.OCCLUDED:
            self.state = TrackState.CONFIRMED

    def update_pose(self, position, rotation=None):
        if self.pose is None:
            self.pose = Vector6DStabilized()
            self.pose.position.from_array(position)
            if rotation is not None:
                self.pose.rotation.from_array(rotation)
        else:
            self.pose.position.update(position)
            if rotation is not None:
                self.pose.rotation.update(rotation)

    def predict_bbox(self):
        """Predict the bbox location based on motion model (kalman tracker)"""
        self.bbox.predict()
        self.age += 1
        if self.age > self.max_disappeared:
            self.state = TrackState.OCCLUDED
        if self.state == TrackState.OCCLUDED:
            if self.age > self.max_age:
                self.state = TrackState.DELETED

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
        if not self.is_deleted():
            return self.state != TrackState.OCCLUDED
        else:
            return False

    def is_confirmed(self):
        """Returns True if the track is confirmed"""
        return self.state == TrackState.CONFIRMED

    def is_occluded(self):
        """Returns True if the track is occluded"""
        return self.state == TrackState.OCCLUDED

    def is_deleted(self):
        """Returns True if the track is deleted"""
        return self.state == TrackState.DELETED

    def is_tentative(self):
        return self.state == TrackState.TENTATIVE

    def is_located(self):
        return self.pose is not None

    def has_shape(self):
        return self.shape is not None

    def has_camera(self):
        return self.camera is not None

    def project_into(self, camera_track):
        """Returns the 2D bbox in the given camera plane"""
        if self.is_located() and camera_track.is_located():
            if self.has_shape() and camera_track.has_camera():
                success, tf_sensor = camera_track.transform()
                success, tf_track = self.pose.transform()
                tf_project = np.dot(np.linalg.inv(tf_sensor), tf_track)
                camera_matrix = camera_track.camera.camera_matrix()
                fx = camera_matrix[0][0]
                fy = camera_matrix[1][1]
                cx, cy = camera_track.camera.center()
                z = tf_project[2]
                w = (self.shape.width() * fx/z)
                h = (self.shape.height() * fy/z)
                x = (tf_project[0] * fx/z)+cx
                y = (tf_project[1] * fy/z)+cy
                xmin = x - w/2.0
                ymin = y - h/2.0
                xmax = x + w/2.0
                ymax = y + h/2.0
                if xmax < 0:
                    return False, None
                if ymax < 0:
                    return False, None
                if xmin > camera_track.camera.width:
                    return False, None
                if ymin > camera_track.camera.height:
                    return False, None
                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax > camera_track.camera.width:
                    xmax = camera_track.camera.width
                if ymax > camera_track.camera.height:
                    ymax = camera_track.camera.height
                return True, BoundingBox(xmin, ymin, xmax, ymax)
        return False, None

    def draw(self, image, color, thickness=1):
        """Draws the track"""
        if self.is_confirmed():
            track_color = (0, 200, 0, 0)
            text_color = (50, 50, 50)
        else:
            if self.is_occluded():
                track_color = (0, 0, 200, 0.3)
                text_color = (250, 250, 250)
            else:
                track_color = (200, 0, 0, 0.3)
                text_color = (250, 250, 250)

        if self.is_confirmed():
            self.bbox.draw(image, track_color, 2)
            cv2.rectangle(image, (self.bbox.xmin, self.bbox.ymax-20),
                                 (self.bbox.xmax, self.bbox.ymax), (200, 200, 200), -1)
            self.bbox.draw(image, track_color, 2)
            self.bbox.draw(image, text_color, 1)

            cv2.putText(image,
                        "{}".format(self.uuid[:6]),
                        (self.bbox.xmax-60, self.bbox.ymax-8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_color,
                        1)
            cv2.putText(image,
                        self.label,
                        (self.bbox.xmin+5, self.bbox.ymax-8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_color,
                        1)
            if "facial_landmarks" in self.features:
                self.features["facial_landmarks"].draw(image, track_color, thickness)
        elif self.is_occluded():
            self.bbox.draw(image, track_color, 1)

    def to_msg(self, header, expiration_duration=1.0):
        """Converts into a ROS message"""
        msg = uwds3_msgs.msg.SceneNode()
        msg.id = self.label+"_"+self.uuid
        msg.label = self.label

        if self.is_located():
            msg.is_located = True
            q = self.pose.quaternion()
            msg.pose_stamped.header = header
            msg.pose_stamped.pose.pose.position.x = self.pose.position.x
            msg.pose_stamped.pose.pose.position.y = self.pose.position.y
            msg.pose_stamped.pose.pose.position.z = self.pose.position.z
            msg.pose_stamped.pose.pose.orientation.x = q[0]
            msg.pose_stamped.pose.pose.orientation.y = q[1]
            msg.pose_stamped.pose.pose.orientation.z = q[2]
            msg.pose_stamped.pose.pose.orientation.w = q[3]

        for features in self.features.values():
            msg.features.append(features.to_msg())

        if self.has_camera():
            msg.has_camera = True
            msg.camera.info.header = header
            msg.camera.info.header.frame_id = msg.id
            msg.camera = self.camera.to_msg()

        if self.has_shape():
            msg.shape = self.shape.to_msg()

        msg.last_update = header.stamp
        msg.expiration_duration = rospy.Duration(expiration_duration)
        return msg
