import cv2
import numpy as np
from pyuwds3.types.landmarks import FacialLandmarks
from pyuwds3.types.landmarks import FacialLandmarks68Index
from pyuwds3.types.vector.vector2d import Vector2D
from pyuwds3.types.features import Features
import dlib
DESIRED_LEFT_EYE=(0.285, 0.38)

class FaceAlignementEstimator(object):
	def __init__(self):
		# store the facial landmark predictor, desired output left
		# eye position, and desired output face width + height
		# self.landmarkestimator = landmarkestimator

		self.desiredFaceWidth = 256
		self.desiredFaceHeight = 256
		self.index = FacialLandmarks68Index()

		self.dlib_detector = dlib.get_frontal_face_detector()

		# if the desired face height is None, set it to be the
		# desired face width (normal behavior)



	def align(self, rgb_image, face):
		self.desiredFaceWidth = int(face.bbox.width())
		self.desiredFaceHeight = int(face.bbox.height())
		facial_landmark = face.features["facial_landmarks"]
		(lStart, lEnd) = (self.index.FIRST_LEFT_EYE_POINTS,self.index.LAST_LEFT_EYE_POINTS)
		(rStart, rEnd) = (self.index.FIRST_RIGHT_EYE_POINTS,self.index.LAST_RIGHT_EYE_POINTS)
		leftEyePts = map( facial_landmark.get_point, range(lStart,lEnd +1))
		rightEyePts = map( facial_landmark.get_point, range(rStart,rEnd +1))

		# compute the center of mass for each eye
		leftEyeCenter = reduce(Vector2D.__add__,leftEyePts).to_array()/len(leftEyePts)
		rightEyeCenter = reduce(Vector2D.__add__,rightEyePts).to_array()/len(rightEyePts)

		# compute the angle between the eye centroids
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]
		angle = np.degrees(np.arctan2(dY, dX)) - 180
		# compute the desired right eye x-coordinate based on the
		# desired x-coordinate of the left eye
		desiredRightEyeX = 1.0 - DESIRED_LEFT_EYE[0]

		# determine the scale of the new resulting image by taking
		# the ratio of the distance between eyes in the *current*
		# image to the ratio of distance between eyes in the
		# *desired* image
		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desiredDist = (desiredRightEyeX - DESIRED_LEFT_EYE[0])
		desiredDist *= self.desiredFaceWidth
		scale = desiredDist / dist

		# compute center (x, y)-coordinates (i.e., the median point)
		# between the two eyes in the input image
		eyesCenter = ((face.bbox.xmin + face.bbox.xmax)/2,(face.bbox.ymin + face.bbox.ymax)/2)
		# grab the rotation matrix for rotating and scaling the face
		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

		# update the translation component of the matrix
		tX = self.desiredFaceWidth * 0.5
		tY = self.desiredFaceHeight * DESIRED_LEFT_EYE[1]
		M[0, 2] += (tX - eyesCenter[0])
		M[1, 2] += (tY - eyesCenter[1])

		# apply the affine transformation
		(w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
		output = cv2.warpAffine(rgb_image, M, (w, h),
			flags=cv2.INTER_CUBIC)

		# return the aligned face

		# output
		facedets = self.dlib_detector(output,1)
		h,w,_ = output.shape
		if len(facedets)>0:

			xmin = max (0,facedets[0].left())
			ymin = max (0,facedets[0].top())
			xmax = min (w, facedets[0].right())
			ymax = min (h, facedets[0].bottom())
			output = output[xmin:xmax,ymin:ymax]
		else:
			output = output[0:int(h*0.8),0:int(w)]
		return output
