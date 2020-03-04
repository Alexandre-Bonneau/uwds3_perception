import cv2
import numpy as np
from pyuwds3.types.detection import Detection


class SingleObjectTracker(object):
    """Single object tracker based on opencv MEDIANFLOW tracker"""
    def __init__(self):
        """ """
        self.medianflow_tracker = None
        self.object_label = None

    def update(self, rgb_image, object):
        """ """
        self.age = 0
        self.medianflow_tracker = cv2.TrackerMedianFlow_create()
        self.object_label = object.label
        xmin = object.bbox.xmin
        ymin = object.bbox.ymin
        w = object.bbox.width()
        h = object.bbox.height()
        bbox = (xmin, ymin, w, h)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        self.medianflow_tracker.init(bgr_image, bbox)

    def predict(self, rgb_image):
        """ """
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        success, bbox = self.medianflow_tracker.update(bgr_image)
        xmin, ymin, width, height = bbox
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        prediction = Detection(xmin, ymin, xmin+width, ymin+height, self.object_label, 1.0)
        return success, prediction
