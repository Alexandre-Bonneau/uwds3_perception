import cv2
import rospy
import numpy as np
import geometry_msgs
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from uwds3_msgs.msg import SceneNodeArrayStamped, SceneNode
from cv_bridge import CvBridge
from tf import transformations
import tf2_ros
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from .detection.opencv_dnn_detector import OpenCVDNNDetector
from .tracking.tracker import Tracker
from .estimation.depth_estimator import DepthEstimator
from .tracking.linear_assignment import iou_distance


class Uwds3Perception(object):
    def __init__(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.tf_broadcaster = TransformBroadcaster()

        self.rgb_image_topic = rospy.get_param("~rgb_image_topic", "/camera/rgb/image_raw")
        self.rgb_camera_info_topic = rospy.get_param("~rgb_camera_info_topic", "/camera/rgb/camera_info")

        self.depth_image_topic = rospy.get_param("~depth_image_topic", "/camera/depth/image_raw")
        self.depth_camera_info_topic = rospy.get_param("~depth_camera_info_topic", "/camera/depth/camera_info")

        self.base_frame_id = rospy.get_param("~base_frame_id", "base_link")
        self.global_frame_id = rospy.get_param("~global_frame_id", "map")

        self.bridge = CvBridge()

        rospy.loginfo("[perception] Subscribing to /{} topic...".format(self.depth_camera_info_topic))
        self.camera_info = None
        self.camera_frame_id = None
        self.camera_info_subscriber = rospy.Subscriber(self.depth_camera_info_topic, CameraInfo, self.camera_info_callback)

        self.detector_model_filename = rospy.get_param("~detector_model_filename", "")
        self.detector_weights_filename = rospy.get_param("~detector_weights_filename", "")
        self.detector_config_filename = rospy.get_param("~detector_config_filename", "")

        self.face_detector_model_filename = rospy.get_param("~face_detector_model_filename", "")
        self.face_detector_weights_filename = rospy.get_param("~face_detector_weights_filename", "")
        self.face_detector_config_filename = rospy.get_param("~face_detector_config_filename", "")

        self.body_parts = ["person", "face", "right_hand", "left_hand"]

        self.detector = OpenCVDNNDetector(self.detector_model_filename,
                                          self.detector_weights_filename,
                                          self.detector_config_filename,
                                          300)

        self.face_detector = OpenCVDNNDetector(self.face_detector_model_filename,
                                               self.face_detector_weights_filename,
                                               self.face_detector_config_filename,
                                               300)

        self.n_frame = rospy.get_param("~n_frame", 2)
        self.frame_count = 0

        self.only_human = rospy.get_param("~only_human", False)

        self.use_depth = rospy.get_param("~use_depth", False)

        self.shape_predictor_config_filename = rospy.get_param("~shape_predictor_config_filename", "")

        self.tracker = Tracker(iou_distance, n_init=10, min_distance=0.8, max_disappeared=4, max_age=15)

        self.tracks_publisher = rospy.Publisher("uwds3_perception/tracks", SceneNodeArrayStamped, queue_size=1)

        self.visualization_publisher = rospy.Publisher("uwds3_perception/visualization", Image, queue_size=1)

        if self.use_depth is True:
            rospy.loginfo("[perception] Subscribing to /{} topic...".format(self.rgb_image_topic))
            self.rgb_image_sub = message_filters.Subscriber(self.rgb_image_topic, Image)
            rospy.loginfo("[perception] Subscribing to /{} topic...".format(self.depth_image_topic))
            self.depth_image_sub = message_filters.Subscriber(self.depth_image_topic, Image)

            self.sync = message_filters.ApproximateTimeSynchronizer([self.rgb_image_sub, self.depth_image_sub], 10, 0.1, allow_headerless=True)
            self.sync.registerCallback(self.observation_callback)

        else:
            rospy.loginfo("[perception] Subscribing to /{} topic...".format(self.rgb_image_topic))
            self.rgb_image_sub = rospy.Subscriber(self.rgb_image_topic, Image, self.observation_callback, queue_size=1)


    def camera_info_callback(self, msg):
        if self.camera_info is None:
            rospy.loginfo("[perception] Camera info received !")
        self.camera_info = msg
        self.camera_frame_id = msg.header.frame_id
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.dist_coeffs = np.array(msg.D)

    def observation_callback(self, rgb_image_msg, depth_image_msg=None):
        if self.camera_info is not None:
            perception_timer = cv2.getTickCount()
            bgr_image = self.bridge.imgmsg_to_cv2(rgb_image_msg)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            if depth_image_msg is not None:
                depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg)
            viz_frame = rgb_image

            detection_timer = cv2.getTickCount()
            detections = []
            if self.frame_count % self.n_frame == 0:
                detections = self.detector.detect(rgb_image)
            if self.frame_count % self.n_frame == 1:
                detections = self.face_detector.detect(rgb_image)
            self.frame_count += 1
            detection_fps = cv2.getTickFrequency() / (cv2.getTickCount() - detection_timer)

            tracking_timer = cv2.getTickCount()
            if self.only_human is False:
                tracks = self.tracker.update(rgb_image, detections, self.camera_matrix, self.dist_coeffs)
            else:
                detections = [d for d in detections if d.class_label in self.body_parts]
            tracks = self.tracker.update(rgb_image, detections, self.camera_matrix, self.dist_coeffs, depth_image=depth_image)

            tracking_fps = cv2.getTickFrequency() / (cv2.getTickCount() - tracking_timer)
            perception_fps = cv2.getTickFrequency() / (cv2.getTickCount() - perception_timer)
            detection_fps_str = "Detection fps : {:0.4f}hz".format(detection_fps)
            tracking_fps_str = "Tracking and pose estimation fps : {:0.4f}hz".format(tracking_fps)
            perception_fps_str = "Perception fps : {:0.4f}hz".format(perception_fps)

            cv2.putText(viz_frame, "nb detection/tracks : {}/{}".format(len(detections), len(tracks)), (5, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(viz_frame, detection_fps_str, (5, 45),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(viz_frame, tracking_fps_str, (5, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(viz_frame, perception_fps_str, (5, 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            entity_array = SceneNodeArrayStamped()
            entity_array.header = rgb_image_msg.header

            for track in tracks:
                draw_track(viz_frame, track, self.camera_matrix, self.dist_coeffs)
                if track.is_confirmed():
                    #print track.translation, track.rotation
                    if track.rotation is not None and track.translation is not None:
                        #print("broadcast tf {}".format(track.uuid))
                        transform = geometry_msgs.msg.TransformStamped()
                        transform.header = rgb_image_msg.header
                        transform.header.frame_id = self.camera_frame_id
                        transform.child_frame_id = track.class_label+"_"+track.uuid[:6]
                        transform.transform.translation.x = track.translation[0]
                        transform.transform.translation.y = track.translation[1]
                        transform.transform.translation.z = track.translation[2]
                        q_rot = transformations.quaternion_from_euler(track.rotation[0], track.rotation[1], track.rotation[2], "rxyz")
                        transform.transform.rotation.x = q_rot[0]
                        transform.transform.rotation.y = q_rot[1]
                        transform.transform.rotation.z = q_rot[2]
                        transform.transform.rotation.w = q_rot[3]
                        self.tf_broadcaster.sendTransform(transform)

                    entity = SceneNode()
                    entity.label = track.class_label
                    if track.class_label == "person":
                        entity.class = "Human"
                    elif track.class_label in self.body_parts:
                        entity.class = "HumanBodyPart"
                    else:
                        entity.class = "Thing"
                    entity.id = track.class_label+"_"+track.uuid
                    if track.translation is not None and track.rotation is not None:
                        entity.is_located = True
                        entity.position.header.frame_id = self.camera_frame_id
                        entity.position.pose.pose.position.x = track.translation[0]
                        entity.position.pose.pose.position.y = track.translation[1]
                        entity.position.pose.pose.position.z = track.translation[2]
                        entity.position.pose.pose.orientation.x = q_rot[0]
                        entity.position.pose.pose.orientation.y = q_rot[1]
                        entity.position.pose.pose.orientation.z = q_rot[2]
                        entity.position.pose.pose.orientation.w = q_rot[3]
                    else:
                        entity.is_located = False
                    entity.has_shape = False
                    entity.last_update = rgb_image_msg.header.stamp
                    entity.expiration_time = rgb_image_msg.header.stamp + rospy.Duration(3.0)
                    entity_array.nodes.append(entity)

            viz_img_msg = self.bridge.cv2_to_imgmsg(viz_frame)
            self.tracks_publisher.publish(entity_array)
            self.visualization_publisher.publish(viz_img_msg)


def draw_track(rgb_image, track, camera_matrix, dist_coeffs):
    if track.is_confirmed():
        tl_corner = (int(track.bbox.left()), int(track.bbox.top()))
        br_corner = (int(track.bbox.right()), int(track.bbox.bottom()))
        if track.rotation is not None and track.translation is not None:
            cv2.drawFrameAxes(rgb_image, camera_matrix, dist_coeffs, np.array(track.rotation).reshape((3,1)), np.array(track.translation).reshape(3,1), 0.03)
        if track.class_label == "face":
            if "facial_landmarks" in track.properties:
                for (x, y) in track.properties["facial_landmarks"]:
                    cv2.circle(rgb_image, (x, y), 1, (0, 255, 0), -1)
            else:
                cv2.circle(rgb_image, (track.bbox.center().x, track.bbox.center().y), 2, (0, 255, 0), -1)
        else:
            cv2.circle(rgb_image, (track.bbox.center().x, track.bbox.center().y), 2, (0, 255, 0), -1)
        cv2.putText(rgb_image, track.uuid[:6], (tl_corner[0]+5, tl_corner[1]+25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 0, 0), 2)
        cv2.putText(rgb_image, track.class_label, (tl_corner[0]+5, tl_corner[1]+45),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 0, 0), 2)
        cv2.rectangle(rgb_image, tl_corner, br_corner, (255, 255, 0), 2)

def get_last_transform_from_tf2(self, source_frame, target_frame):
        try:
            trans = self.tf_buffer.lookup_transform(source_frame, target_frame, rospy.Time(0))
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z

            rx = trans.transform.rotation.x
            ry = trans.transform.rotation.y
            rz = trans.transform.rotation.z
            rw = trans.transform.rotation.w

            return True, [x, y, z], [rx, ry, rz, rw]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return False, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
