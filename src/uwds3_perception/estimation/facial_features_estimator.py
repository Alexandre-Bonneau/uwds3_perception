import cv2
import numpy as np
from pyuwds3.types.features import Features
from .face_frontalizer_estimator import FaceFrontalizerEstimator
from uwds3_perception.estimation.face_alignement_estimator import FaceAlignementEstimator

class FacialFeaturesEstimator(object):
    """Represents the facial description estimator"""
    def __init__(self, face_3d_model_filename, embedding_model_filename, frontalize=False):
        """FacialFeaturesEstimator constructor"""
        self.name = "facial_description"
        self.dimensions = (128, 0)
        self.face_alignement_estimator = FaceAlignementEstimator()
        self.model = cv2.dnn.readNetFromTorch(embedding_model_filename)
        if frontalize is True:
            ref_filename = "../../../data/face_frontalizer/ref3d.pkl"
            self.frontalizer = FaceFrontalizerEstimator(ref_filename)
        else:
            self.frontalizer = None

    def estimate(self, rgb_image, faces, camera_matrix=None, dist_coeffs=None):
        """Extracts the facial description features"""
        cropped_imgs = []
        for f in faces:
            if "facial_description" not in f.features:
                xmin = int(f.bbox.xmin)
                ymin = int(f.bbox.ymin)
                w = int(f.bbox.width())
                h = int(f.bbox.height())
                if w > 27 and h > 27:
                    cropped_imgs.append(self.face_alignement_estimator.align(rgb_image,f))

                    if self.frontalizer is not None:
                        bool,_,frontalized_img= self.frontalizer.estimate(rgb_image,f)
                        if bool:
                            cropped_imgs.append(frontalized_img)
                        else:
                            assert(False), "Error in Frontaliser"
        if len(cropped_imgs) > 0:
            blob = cv2.dnn.blobFromImages(cropped_imgs,
                                          1.0 / 255,
                                          (96, 96),
                                          (0, 0, 0),
                                          swapRB=False,
                                          crop=False)
            self.model.setInput(blob)
            for f, features in zip(faces, self.model.forward()):
                f.features[self.name] = Features(self.name, self.dimensions, np.array(features), float(h)/rgb_image.shape[0])
