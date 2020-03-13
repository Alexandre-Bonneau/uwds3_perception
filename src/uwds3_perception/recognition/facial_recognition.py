import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from uwds3_perception.detection.opencv_dnn_detector import OpenCVDNNDetector
from uwds3_perception.estimation.facial_features_estimator import FacialFeaturesEstimator
from uwds3_perception.estimation.face_alignement_estimator import FaceAlignementEstimator
from uwds3_perception.estimation.facial_landmarks_estimator import FacialLandmarksEstimator
from uwds3_perception.detection.face_detector import FaceDetector
from uwds3_perception.recognition.knn_assignement import KNearestNeighborsAssignement
from uwds3_perception.recognition.knn_assignement import KNNLoader
import numpy.random as rng
from sklearn.utils import shuffle


class FacialRecognition(object):
    def __init__(self, knn_model_filename):
        loader = KNNLoader()
        self.knn = loader.load(knn_model_filename)
    def recognize(self, face_tracks):
        for track in face_tracks:
            _, track.description, _ = self.knn.predict(track.features["facial_description"].to_array())


class OpenFaceRecognition(object):
    def __init__(self,
                 detector_model_filename,
                 detector_weights_filename,
                 detector_config_filename,
                 face_3d_model_filename,
                 embedding_model_file,
                 shape_predictor_config_filename,
                 frontalize=False,
                 metric_distance=euclidean):
        self.face_detector = OpenCVDNNDetector(detector_model_filename,
                                               detector_weights_filename,
                                               detector_config_filename,
                                               300)

        self.detector_model_filename = detector_model_filename
        self.facial_landmarks_estimator= FacialLandmarksEstimator(shape_predictor_config_filename)
        self.face_alignement_estimator = FaceAlignementEstimator()
        self.facial_features_estimator = FacialFeaturesEstimator( face_3d_model_filename,embedding_model_file,frontalize)
        #self.input_shape = input_shape
        self.detector_weights_filename = detector_weights_filename
        self.frontalize = frontalize
        self.metric_distance = metric_distance

    def extract(self, rgb_image):
        face_list = self.face_detector.detect(rgb_image)
        if len(face_list) == 0:
            print("no image found for extraction")
            return None
        else:
            # cv2.imshow('image',rgb_image)
            # cv2.waitKey(0)

            self.facial_landmarks_estimator.estimate(rgb_image,face_list)
            self.facial_features_estimator.estimate(rgb_image,face_list,self.frontalize)
            name = self.facial_features_estimator.name

            # rgb_image2 = self.face_alignement_estimator.align(rgb_image,face_list[0])
            # cv2.imshow('image',rgb_image2)

            # cv2.waitKey(0)
            return face_list[0].features[name]

    def predict(self, rgb_image_1, rgb_image_2):
        feature1 = self.extract(rgb_image_1)
        feature2 = self.extract(rgb_image_2)
        return(1 - self.metric_distance(feature1.to_array(),
                                        feature2.to_array()))



class FacialRecognitionDataLoader(object):
    def __init__(self, train_directory_path, val_directory_path):
        print("Start loading the dataset:\r\n'{}'\r\n'{}'".format(train_directory_path, val_directory_path))
        self.X_train, self.Y_train, self.train_classes = self.load_dataset(train_directory_path)
        self.X_val, self.Y_val, self.val_classes = self.load_dataset(val_directory_path)
        self.knn = None
        print("Training categories ({} different):".format(len(self.train_classes.keys())))
        print("{}\r\n".format(self.train_classes.keys()))
        print("Validation categories ({} different from training):".format(len(self.val_classes.keys())))
        print("{}\r\n".format(self.val_classes.keys()))
        self.nb_of_undetection = 0

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
                rgb_image_flip = cv2.flip(rgb_image,1)
                individual_images.append(rgb_image_flip)
                Y_data.append(n)
            try:
                X_data.append(np.stack(individual_images))
            except ValueError as e:
                print("Exception occured: {}".format(e))
            n += 1

        X_data = np.stack(X_data)
        Y_data = np.vstack(Y_data)
        return X_data, Y_data, individual_dict

    def test_recognition(self, model, N_way, trials, mode="val", verbose=True):
        """
        Tests average N way recognition accuracy of the embedding net over k trials
        """
        n_correct = 0
        if verbose:
            print("Evaluating model {} on {} random  way recognition tasks...".format(trials, N_way))
        for i in range(trials):
            true_person, support_set, targets = self.make_recognition_task(N_way, mode=mode)
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

        plt.plot(ways, train_accs, "b", label="Distance between feature")
        plt.plot(ways, 100.0/ways, "g", label="Random guessing")

        plt.xlabel("Number of people")
        plt.ylabel("% Accuracy")
        plt.title("Facial Recognition Performance")
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

    def knn_init(self, feature_name, max_distance, data_directory="", n_neighbors=1, algorithm="ball_tree", weights="distance"):
        self.knn = KNearestNeighborsAssignement(feature_name, max_distance,data_directory, n_neighbors, algorithm, weights )

    def knn_update(self, model):
        X = self.X_train.copy()
        x, y, w, h, n = X.shape
        X = X.reshape(x*y, w, h, n)
        Y = self.Y_train
        persons_list = self.train_classes
        for (image, person_id) in zip(X, Y):
            image_feature = model.extract(image)
            if image_feature is not None:
                self.knn.update(image_feature.to_array(), persons_list[person_id[0]])
            else:
                self.nb_of_undetection+=1
        if self.nb_of_undetection > 0:
            print(str(self.nb_of_undetection)+" images where not taken into account : no faces found")

    def knn_train(self):
        self.knn.train()

    def knn_validation(self, model):
        X = self.X_val.copy()
        x, y, w, h, n = X.shape
        X = X.reshape(x*y, w, h, n)
        Y = self.Y_val
        persons_list = self.val_classes
        count = 0
        for (image, person_id) in zip(X, Y):
            image_feature = model.extract(image).to_array()
            bool, value, distance = self.knn.multi_predict(image_feature)
            if value == persons_list[person_id[0]]:
                count +=1
        accuracy = count / (1.0 * len(X))
        print("The accuracy is "+str(accuracy))
        return accuracy


class Create_network(object):
    def __init__(self, faces_location):
        self.frdl = FacialRecognitionDataLoader(faces_location,faces_location)
        self.frdl.load_dataset(faces_location)
        self.ofd = OpenFaceRecognition(detector_model, detector_model_txt,
        detector_config_filename,face_3d_model_filename,embedding_model_file,shape_predictor_config_filename,
        frontalize = False)

    def create(self, name):
        self.frdl.knn_init("visage",0.9)
        self.frdl.knn_update(self.ofd)
        self.frdl.knn_train()
        self.frdl.knn.save(name)

if __name__ == '__main__':
    detector_model = "../../../models/detection/opencv_face_detector_uint8.pb"
    detector_model_txt = "../../../models/detection/opencv_face_detector.pbtxt"
    embedding_model_file = "../../../models/features/nn4.small2.v1.t7"
    #detector_model_test = "../../../models/detection/ssd_mobilenet_v2_coco_2018_03_29.pb"
    detector_config_filename = "../../../config/detection/face_config.yaml"
    face_3d_model_filename = "../../../config/estimation/face_3d_model.npy"
    shape_predictor_config_filename= "../../../models/estimation/shape_predictor_68_face_landmarks.dat"
    faces_location = "../../../data/face_recognition/snapshots_2faces"
    cn = Create_network(faces_location)
    cn.create("Test2faces")
