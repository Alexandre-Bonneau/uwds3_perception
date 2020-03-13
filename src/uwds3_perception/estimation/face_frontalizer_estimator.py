import matplotlib.pyplot as plt
import cv2
import dlib
import numpy as np
import copy
from scipy import ndimage
import scipy.misc as sm
import pickle as pkl
from uwds3_perception.estimation.facial_landmarks_estimator import FacialLandmarksEstimator
from uwds3_perception.recognition.knn_assignement import KNNLoader
from uwds3_perception.detection.opencv_dnn_detector import OpenCVDNNDetector
WD = 250
HT = 250
ACC_CONST = 800
ref_filename = "../../../data/face_frontalizer/ref3d.pkl"
test = "../../../data/face_frontalizer/test.jpg"
def plot3d(p3ds):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(p3ds[:,0],p3ds[:,1], p3ds[:,2])
    plt.show()


class FaceFrontalizerEstimator(object):
    """Python implementation of :
    Tal Hassner, Shai Harel*, Eran Paz* and Roee Enbar,
    Effective Face Frontalization in Unconstrained Images,
    IEEE Conf. on Computer Vision and Pattern Recognition (CVPR 2015)

    modified from https://github.com/ChrisYang/facefrontalisation"""
    def __init__(self, ref_3d_model_filename):
        #self.points_3d = np.load(face_3d_model_filename)
        with open(ref_3d_model_filename) as f:
            ref = pkl.load(f)
            self.refU  = ref['refU']
            self.A = ref['outA']
            self.refxy = ref['ref_XY']
            self.p3d = ref['p3d']
            self.refimg = ref['refimg']
    def get_headpose(self,p2d):
        assert(len(p2d) == len(self.p3d))
        p3_ = np.reshape(self.p3d,(-1,3,1)).astype(np.float)
        p2_ = np.reshape(p2d,(-1,2,1)).astype(np.float)
        distCoeffs = np.zeros((5,1))
        succ,rvec,tvec = cv2.solvePnP(p3_,p2_, self.A, distCoeffs)
        if not succ:
            print('There is something wrong, please check.')
            return None
        else:
            matx = cv2.Rodrigues(rvec)

            ProjM_ = self.A.dot(np.insert(matx[0],3,tvec.T,axis=1))
            return rvec,tvec,ProjM_

    def estimate(self, rgb_image, face):
        if "facial_landmarks" in face.features:
            PATH_face_model = '../../../data/face_frontalizer/face_shape.dat'
            md_face = dlib.shape_predictor(PATH_face_model)
            face_det = dlib.get_frontal_face_detector()
            facedets = face_det(rgb_image,1)
            # print(det)
            # if(len(facedets) == 0):
            #     cv2.imshow('image',rgb_image)
            #     cv2.waitKey(0)
            #     det = dlib.rectangle(face.bbox.xmin,face.bbox.ymin,face.bbox.xmax,face.bbox.ymax)
            #
            # else:
            det = facedets[0]

            # print(det)
            # print(face.bbox.xmin)
            # print(facedets)
            # print("fghjklmefrtyuio")
            shape = md_face(rgb_image,det)
            p2d_ = np.asarray([(shape.part(n).x, shape.part(n).y,) for n in range(shape.num_parts)], np.float32)


            #p2d_ = face.features["facial_landmarks"].data
            # p2d_ = p2d
            facebb = [face.bbox.xmin, face.bbox.ymin, face.bbox.width(), face.bbox.height()]
            w = facebb[2]
            h = facebb[3]
            fb_ = np.clip([[facebb[0] - w, facebb[1] - h],[facebb[0] + 2 * w, facebb[1] + 2 * h]], [0,0], [rgb_image.shape[1], rgb_image.shape[0]])
            img = rgb_image[fb_[0][1]:fb_[1][1], fb_[0][0]:fb_[1][0],:]
            p2d = copy.deepcopy(p2d_)
            p2d[:,0] = (p2d_[:,0] - fb_[0][0]) * float(WD) / float(img.shape[1])
            p2d[:,1] = (p2d_[:,1] - fb_[0][1])  * float(HT) / float(img.shape[0])
            img = cv2.resize(img, (WD,HT))
            #finished rescaling

            tem3d = np.reshape(self.refU,(-1,3),order='F')
            bgids = tem3d[:,1] < 0# excluding background 3d points
            # plot3d(tem3d)
            # print tem3d.shape
            ref3dface = np.insert(tem3d, 3, np.ones(len(tem3d)),axis=1).T
            ProjM = self.get_headpose(p2d)[2]
            proj3d = ProjM.dot(ref3dface)
            proj3d[0] /= proj3d[2]
            proj3d[1] /= proj3d[2]
            proj2dtmp = proj3d[0:2]
            #The 3D reference is projected to the 2D region by the estimated pose
            #The check the projection lies in the image or not
            vlids = np.logical_and(np.logical_and(proj2dtmp[0] > 0, proj2dtmp[1] > 0),
                                   np.logical_and(proj2dtmp[0] < img.shape[1] - 1,  proj2dtmp[1] < img.shape[0] - 1))
            vlids = np.logical_and(vlids, bgids)
            proj2d_valid = proj2dtmp[:,vlids]

            sp_  = self.refU.shape[0:2]
            synth_front = np.zeros(sp_,np.float)
            inds = np.ravel_multi_index(np.round(proj2d_valid).astype(int),(img.shape[1], img.shape[0]),order = 'F')
            unqeles, unqinds, inverids, conts  = np.unique(inds, return_index=True, return_inverse=True, return_counts=True)
            tmp_ = synth_front.flatten()
            tmp_[vlids] = conts[inverids].astype(np.float)
            synth_front = tmp_.reshape(synth_front.shape,order='F')
            synth_front = cv2.GaussianBlur(synth_front, (17,17), 30).astype(np.float)

            rawfrontal = np.zeros((self.refU.shape[0],self.refU.shape[1], 3))
            for k in range(3):
                z = img[:,:,k]
                intervalues = ndimage.map_coordinates(img[:,:,k].T,proj2d_valid,order=3,mode='nearest')
                tmp_  = rawfrontal[:,:,k].flatten()
                tmp_[vlids] = intervalues
                rawfrontal[:,:,k] = tmp_.reshape(self.refU.shape[0:2],order='F')

            mline = synth_front.shape[1]/2
            sumleft = np.sum(synth_front[:,0:mline])
            sumright = np.sum(synth_front[:,mline:])
            sum_diff = sumleft - sumright
            print sum_diff
            if np.abs(sum_diff) > ACC_CONST:
                weights = np.zeros(sp_)
                if sum_diff > ACC_CONST:
                    weights[:,mline:] = 1.
                else:
                    weights[:,0:mline] = 1.
                weights = cv2.GaussianBlur(weights, (33,33), 60.5).astype(np.float)
                synth_front /= np.max(synth_front)
                weight_take_from_org = 1 / np.exp(1 + synth_front)
                weight_take_from_sym = 1 - weight_take_from_org
                weight_take_from_org = weight_take_from_org * np.fliplr(weights)
                weight_take_from_sym = weight_take_from_sym * np.fliplr(weights)
                weights = np.tile(weights,(1,3)).reshape((weights.shape[0],weights.shape[1],3),order='F')
                weight_take_from_org = np.tile(weight_take_from_org,(1,3)).reshape((weight_take_from_org.shape[0],weight_take_from_org.shape[1],3),order='F')
                weight_take_from_sym = np.tile(weight_take_from_sym,(1,3)).reshape((weight_take_from_sym.shape[0],weight_take_from_sym.shape[1],3),order='F')
                denominator = weights + weight_take_from_org + weight_take_from_sym
                frontal_sym = (rawfrontal * weights + rawfrontal * weight_take_from_org + np.fliplr(rawfrontal) * weight_take_from_sym) / denominator
            else:
                frontal_sym = rawfrontal
            if(len(facedets)==0):
                cv2.imshow('image',frontal_sym)
                cv2.waitKey(0)

            return True,rawfrontal, np.round(frontal_sym).astype(np.uint8)
        return False, None, None

if __name__ == '__main__':
    detector_model = "../../../models/detection/opencv_face_detector_uint8.pb"
    detector_model_txt = "../../../models/detection/opencv_face_detector.pbtxt"
    embedding_model_file = "../../../models/features/nn4.small2.v1.t7"
    #detector_model_test = "../../../models/detection/ssd_mobilenet_v2_coco_2018_03_29.pb"
    detector_config_filename = "../../../config/detection/face_config.yaml"
    face_3d_model_filename = "../../../config/estimation/face_3d_model.npy"
    shape_predictor_config_filename= "../../../models/estimation/shape_predictor_68_face_landmarks.dat"

    face_detector = OpenCVDNNDetector(detector_model, detector_model_txt,detector_config_filename,300)
    image = cv2.imread(test)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face = face_detector.detect(rgb_image)
    ffe = FaceFrontalizerEstimator(ref_filename)
    facial_landmarks =FacialLandmarksEstimator(shape_predictor_config_filename)
    facial_landmarks.estimate(rgb_image,face)
    # print(face[0].BoundingBox)


    # fronter = frontalizer('ref3d.pkl')
    # img = plt.imread(test)

    b,i,fe = ffe.estimate(rgb_image,face[0])
    # aaa=p2d
    #
    # bbb=face[0].features["facial_landmarks"].data
    if b :
        sm.toimage(np.round(fe).astype(np.uint8)).show()
