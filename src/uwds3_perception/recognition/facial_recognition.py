import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cosine
from uwds3_perception.detection.opencv_dnn_detector import OpenCVDNNDetector
from uwds3_perception.estimation.facial_features_estimator import FacialFeaturesEstimator
from uwds3_perception.detection.face_detector import FaceDetector
from pyuwds3.types.features import Features
import numpy.random as rng
import time
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Lambda
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.utils import shuffle
from keras import backend as K

class OpenFaceRecognition(object):
    def __init__(self,
                 #input_shape,
                 detector_model_filename,
                 detector_weights_filename,
                 detector_config_filename,
                 frontalize=False,
                 metric_distance=euclidean):
        self.face_detector = OpenCVDNNDetector(detector_model_filename,
                                               detector_weights_filename,
                                               detector_config_filename,
                                               300)

        self.detector_model_filename = detector_model_filename
        self.facial_features_estimator = FacialFeaturesEstimator( face_3d_model_filename,embedding_model_file)
        #self.input_shape = input_shape
        self.detector_weights_filename = detector_weights_filename
        self.frontalize = frontalize
        self.metric_distance = metric_distance

    def extract(self, rgb_image):
        # print rgb_image.shape

        face_list = self.face_detector.detect(rgb_image)
        if len(face_list) == 0:
            print("no image found for extraction")
        else:
            self.facial_features_estimator.estimate(rgb_image,face_list,self.frontalize)
            name = self.facial_features_estimator.name
            return face_list[0].features[name]


    def predict(self, rgb_image_1, rgb_image_2):
        feature1 = self.extract(rgb_image_1)
        feature2 = self.extract(rgb_image_2)
        return(1- self.metric_distance(feature1.to_array(),
                                    feature2.to_array()))



class FacialRecognitionDataLoader(object):
    def __init__(self, train_directory_path, val_directory_path):
        print("Start loading the dataset:\r\n'{}'\r\n'{}'".format(train_directory_path, val_directory_path))
        self.X_train, self.Y_train, self.train_classes = self.load_dataset(train_directory_path)
        self.X_val, self.Y_val, self.val_classes = self.load_dataset(val_directory_path)
        print("Training categories ({} different):".format(len(self.train_classes.keys())))
        print("{}\r\n".format(self.train_classes.keys()))
        print("Validation categories ({} different from training):".format(len(self.val_classes.keys())))
        print("{}\r\n".format(self.val_classes.keys()))

    def load_dataset(self, data_directory_path, n=0):
        X_data = []
        Y_data = []
        individual_dict = {}

        for c in os.listdir(data_directory_path):
            individual_dict[n] = c
            print(data_directory_path)
            individual_path = os.path.join(data_directory_path, c)
            individual_images = []
            for snapshot_file in os.listdir(individual_path):
                image_path = os.path.join(individual_path, snapshot_file)
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                individual_images.append(rgb_image)
                Y_data.append(n)
            try:
                X_data.append(np.stack(individual_images))
            except ValueError as e:
                print("Exception occured: {}".format(e))
            n += 1


        X_data = np.stack(X_data)
        Y_data = np.vstack(Y_data)
        # print(X_data.shape)
        # print(Y_data.shape)
        return X_data, Y_data, individual_dict

    def test_recognition(self, model, N_way, trials, mode="val", verbose=True):
        """
        Tests average N way recognition accuracy of the embedding net over k trials
        """
        n_correct = 0
        if verbose:
            print("Evaluating model {} on {} random  way recognition tasks...".format(trials, N_way))
        for i in range(trials):
            # print( "{} / {} trial".format(i,trials) )
            true_person,support_set, targets = self.make_recognition_task(N_way, mode=mode)
            probs = []
            for i in support_set:
                probs.append(model.predict(true_person,i))
            if np.argmax(probs) == np.argmax(targets):
                n_correct += 1
        percent_correct = (100.0 * n_correct / trials)
        if verbose:
            print("Got an average of {}% {} way recognition accuracy".format(percent_correct, N_way))
        return percent_correct


    def evaluate(self, model, N_way=20, trials=50, verbose=True):
        ways = np.arange(1, N_way+1)
        train_accs = []

        for N in ways:
            train_accs.append(self.test_recognition(model, N, trials, mode="train", verbose=verbose))

        plt.plot(ways, train_accs, "b", label="")
        plt.xlabel("Number of possible classes in one-shot tasks")
        plt.ylabel("% Accuracy")
        plt.title("One-Shot Learning Performance")
        plt.legend(loc='center left')
        plt.axis([1, N_way, 0, 100])
        plt.show()

    def make_recognition_task(self, N_way, mode="val", person=None):
        """
        Creates pairs of test image, support set for testing N way learning.
        """

        if mode == 'train':
            X = self.X_train
            persons = self.train_classes
        else:
            X = self.X_val
            persons = self.val_classes

        n_classes, n_examples, w, h,_ = X.shape

        if person is not None: # if person is specified,
            true_person = person
        else: # if no class specified just pick a bunch of random
            true_person = np.random.randint(n_classes)

        ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))
        indices = rng.choice((range(true_person) + range(true_person+1, n_classes)),N_way-1)
        support_set = [X[true_person][ex2]]
        for i in indices:
            support_set.append(X[i][rng.choice(n_examples)])
        targets = np.zeros((N_way,))
        targets[0] = 1
        targets, support_set = shuffle(targets, support_set)
        return X[true_person][ex1], support_set, targets

if __name__ == '__main__':
    detector_model = "../../../models/detection/opencv_face_detector_uint8.pb"
    detector_model_txt = "../../../models/detection/opencv_face_detector.pbtxt"
    embedding_model_file = "../../../models/features/nn4.small2.v1.t7"
    #detector_model_test = "../../../models/detection/ssd_mobilenet_v2_coco_2018_03_29.pb"
    detector_config_filename = "../../../config/detection/face_config.yaml"
    face_3d_model_filename = "../../../config/estimation/face_3d_model.npy"


frdl = FacialRecognitionDataLoader("../../../src/uwds3_perception/recognition/snapshots",
"../../../src/uwds3_perception/recognition/snapshots")
#frdl.load_dataset("../../../src/uwds3_perception/recognition/snapshots/")

ofd = OpenFaceRecognition(detector_model, detector_model_txt,detector_config_filename)
#frdl.test_recognition(ofd,4,50)
frdl.evaluate(ofd,4)
