import cv2
import uwds3_msgs.msg
from pyuwds3.types.shape.sphere import Sphere
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np

K = 6


class ShapeEstimator(object):
    """ """
    def estimate(self, rgb_image, objects_tracks, camera_matrix, dist_coeffs):
        """ """
        for o in objects_tracks:
            if o.bbox.depth is not None:
                if o.label != "person":
                    if not o.has_shape():
                        shape = o.bbox.cylinder(camera_matrix, dist_coeffs)
                        if o.label == "face":
                            shape = Sphere(shape.width()*2.0)
                        shape.pose.pos.x = .0
                        shape.pose.pos.y = .0
                        shape.pose.pos.z = .0
                        shape.color = self.__compute_dominant_color(rgb_image, o.bbox)
                        o.shapes.append(shape)
                else:
                    shape = o.bbox.cylinder(camera_matrix, dist_coeffs)
                    shape.pose.pos.x = .0
                    shape.pose.pos.y = .0
                    shape.pose.pos.z = .0
                    if not o.has_shape():
                        shape.color = self.__compute_dominant_color(rgb_image, o.bbox)
                        shape.w = 0.60
                        o.shapes.append(shape)
                    else:
                        o.shapes[0].w = 0.60
                        o.shapes[0].h = shape.h

    def __compute_dominant_color(self, rgb_image, bbox):
        xmin = int(bbox.xmin)
        ymin = int(bbox.ymin)
        h = int(bbox.height())
        w = int(bbox.width())
        cropped_image = rgb_image[ymin:ymin+h, xmin:xmin+w].copy()
        np_pixels = cropped_image.shape[0] * cropped_image.shape[1]
        cropped_image = cropped_image.reshape((np_pixels, 3))
        clt = KMeans(n_clusters=K)
        labels = clt.fit_predict(cropped_image)
        label_counts = Counter(labels)
        dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]/255.0
        color = np.ones(4)
        color[0] = dominant_color[0]
        color[1] = dominant_color[1]
        color[2] = dominant_color[2]
        return color
