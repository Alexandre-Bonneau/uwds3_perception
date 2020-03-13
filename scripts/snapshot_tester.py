#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import cv2
import argparse
from uwds3_perception.detection.opencv_dnn_detector import OpenCVDNNDetector
from uwds3_perception.recognition.knn_assignement import KNearestNeighborsAssignement
from uwds3_perception.recognition.facial_recognition import OpenFaceRecognition
from uwds3_perception.estimation.face_alignement_estimator import FaceAlignementEstimator
from uwds3_perception.estimation.facial_landmarks_estimator import FacialLandmarksEstimator
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Record RGB snapshots for machine learning")
    parser.add_argument('label', type=str, help='The label used to name the data directory')
    parser.add_argument("-d", "--data_dir", type=str, default="/tmp/snapshots/", help="The root data directory (default '/tmp/snapshots/')")
    args = parser.parse_args()
    snapshot_directory = args.data_dir + args.label + "/"
    detector_model = "../models/detection/opencv_face_detector_uint8.pb"
    detector_model_txt = "../models/detection/opencv_face_detector.pbtxt"
    detector_config_filename = "../config/detection/face_config.yaml"
    shape_predictor_config_filename= "../models/estimation/shape_predictor_68_face_landmarks.dat"
    face_3d_model_filename = "../config/estimation/face_3d_model.npy"
    embedding_model_file = "../models/features/nn4.small2.v1.t7"
    facial_landmarks_estimator= FacialLandmarksEstimator(shape_predictor_config_filename)
    model = OpenFaceRecognition(detector_model,detector_model_txt,detector_config_filename,face_3d_model_filename,embedding_model_file,shape_predictor_config_filename)
    face_alignement_estimator = FaceAlignementEstimator()
    face_detector = OpenCVDNNDetector(detector_model,
                                        detector_model_txt,
                                        detector_config_filename,
                                        300)
    file = open("../data/face_recognition/Test2faces",'r')
    knn = pickle.load(file)
    file.close()


    try:
        os.makedirs(snapshot_directory)
    except OSError as e:
        if not os.path.isdir(snapshot_directory):
            raise RuntimeError("{}".format(e))
    snapshot_index = 0

    capture = cv2.VideoCapture(0)
    while True:
        ok, frame = capture.read()
        viz_frame = frame.copy()
        if ok:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_list = face_detector.detect(rgb_image)
            if len(face_list)>0:
                facial_landmarks_estimator.estimate(rgb_image,face_list)


                _,a,score  = knn.multi_predict(model.extract(rgb_image).to_array())

                face_list[0].confidence = score
                face_list[0].label += " " + a
                color = (0,0,251)
                if a == "Alexandre":
                    color = (251,0,0)
                if a == "Yoan":
                    color = (0,251,0)
                face_list[0].draw(frame,color)
            k = cv2.waitKey(1) & 0xFF
            if k == 32 and len(face_list)>0:
                print("Save image "+str(snapshot_index)+".jpg !")
                cv2.imwrite(snapshot_directory+str(snapshot_index)+".jpg", viz_frame)
                snapshot_index += 1
                cv2.imshow("Snapshot recorder", (255-frame))
            else:
                cv2.imshow("Snapshot recorder", frame)
    capture.release()
