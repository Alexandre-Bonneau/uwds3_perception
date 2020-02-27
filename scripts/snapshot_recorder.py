#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import cv2
import argparse
from uwds3_perception.detection.opencv_dnn_detector import OpenCVDNNDetector
from uwds3_perception.recognition.knn_assignement import KNearestNeighborsAssignement
from uwds3_perception.recognition.facial_recognition import OpenFaceRecognition
import pickle
file = open("../src/uwds3_perception/recognition/test_knn",'r')
knn = pickle.load(file)
file.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Record RGB snapshots for machine learning")
    parser.add_argument('label', type=str, help='The label used to name the data directory')
    parser.add_argument("-d", "--data_dir", type=str, default="/tmp/snapshots/", help="The root data directory (default '/tmp/snapshots/')")
    args = parser.parse_args()
    snapshot_directory = args.data_dir + args.label + "/"
    detector_model = "../models/detection/opencv_face_detector_uint8.pb"
    detector_model_txt = "../models/detection/opencv_face_detector.pbtxt"
    detector_config_filename = "../config/detection/face_config.yaml"
    model = OpenFaceRecognition(detector_model,detector_model_txt,detector_config_filename)

    face_detector = OpenCVDNNDetector(detector_model,
                                        detector_model_txt,
                                        detector_config_filename,
                                        300)


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
                _,a,score  = knn.predict(model.extract(rgb_image).to_array())
                face_list[0].confidence = score[0]
                face_list[0].label += " " + a
                color = (0,230,0)
                if a != "alexandre":
                    color = (251,0,0)
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
