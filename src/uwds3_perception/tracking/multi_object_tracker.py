import numpy as np
import rospy
from pyuwds3.bbox_metrics import iou, overlap, centroid
from .linear_assignment import LinearAssignment
from .single_object_tracker import SingleObjectTracker
from scipy.spatial.distance import euclidean, cosine
from .track import Track


def iou_cost(detection, track):
    """Returns the iou cost"""
    return abs(1 - iou(detection.bbox, track.bbox))


def overlap_cost(detection, track):
    """Returns the overlap cost"""
    return 1 - overlap(detection.bbox, track.bbox)


def centroid_cost(detection, track):
    """Returns the centroid cost"""
    return centroid(detection.bbox, track.bbox)


def color_cost(detection, track):
    """Returns the color cost"""
    return euclidean(detection.features["color"].data,
                     track.features["color"].data)


def face_cost(detection, track):
    """Returns the face cost"""
    return euclidean(detection.features["facial_description"].data,
                     track.features["facial_description"].data)


class MultiObjectTracker(object):
    """Represents the multi object tracker"""
    def __init__(self,
                 geometric_metric,
                 features_metric,
                 max_distance_geom,
                 max_distance_feat,
                 n_init,
                 max_disappeared,
                 max_age):

        self.n_init = n_init
        self.max_disappeared = max_disappeared
        self.max_age = max_age
        self.features_metric = features_metric
        self.max_distance_feat = max_distance_feat
        self.tracks = []
        self.geometric_assignment = LinearAssignment(geometric_metric, max_distance=max_distance_geom)
        self.features_assignment = LinearAssignment(features_metric, max_distance=max_distance_feat)

    def update(self, rgb_image, detections):
        """Updates the tracker"""
        # First we try to assign the detections to the tracks by using a geometric assignment (centroid or iou)
        if len(detections) > 0:
            first_matches, unmatched_detections, unmatched_tracks = self.geometric_assignment.match(self.tracks, detections)

            # Then we try to assign de detections to the tracks that didn't match based on the features
            if len(unmatched_tracks) > 0 and len(unmatched_detections) > 0:
                trks = [self.tracks[t] for t in unmatched_tracks]
                dets = [detections[d] for d in unmatched_detections]

                second_matches, remaining_detections, remaining_tracks = self.features_assignment.match(trks, dets)
                matches = list(first_matches)+list(second_matches)
            else:
                matches = first_matches
                remaining_tracks = unmatched_tracks
                remaining_detections = unmatched_detections

            for detection_indice, track_indice in matches:
                self.tracks[track_indice].update(detections[detection_indice])
        else:
            remaining_tracks = np.arange(len(self.tracks))
            remaining_detections = []

        for track_indice in remaining_tracks:
            self.tracks[track_indice].mark_missed()
            if self.tracks[track_indice].is_confirmed() and not self.tracks[track_indice].is_occluded():
                self.tracks[track_indice].predict_bbox()

        for detection_indice in remaining_detections:
            self.start_track(rgb_image, detections[detection_indice])

        self.tracks = [t for t in self.tracks if not t.to_delete()]

        return self.tracks

    def start_track(self, rgb_image, detection):
        """Start to track a detection"""
        self.tracks.append(Track(rgb_image,
                                 detection,
                                 self.n_init,
                                 self.max_disappeared,
                                 self.max_age,
                                 self.features_metric,
                                 self.max_distance_feat))
        return len(self.tracks)-1
