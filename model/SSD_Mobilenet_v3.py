import cv2

import numpy as np
import time


class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        ###############################################################

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
            #comprobar que se este trayendo la lista correctamente
            self.classesList.insert(0, '__Background__')
            print(self.classesList)


    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if (cap.isOpened()==False):
            print("Error opening file...")

        (success, image) = cap.read()

        while success:
            classLabelIDs, confidences, bbox = self.net.detect(image, confThreshold = 0.4)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences =