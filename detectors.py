import cv2
from collections import namedtuple


OPEN_CV_DETECTOR_CONFIG = namedtuple("OPEN_CV_DETECTOR_CONFIG", ["FACE_DETECTOR_FILE_PATH", "EYE_DETECTOR_FILE_PATH"])


class OpenCvDetector(object):
    def __init__(self, config):
        self.face_detector = cv2.CascadeClassifier(config.FACE_DETECTOR_FILE_PATH)
        self.eye_detector = cv2.CascadeClassifier(config.EYE_DETECTOR_FILE_PATH)

    def detect_faces(self, image, scale_factor, min_neighbors):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scale_factor, min_neighbors)
        return faces

    def detect_eyes(self, image, scale_factor, min_neighbors):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        eyes = self.eye_detector.detectMultiScale(gray, scale_factor, min_neighbors)
        return eyes
