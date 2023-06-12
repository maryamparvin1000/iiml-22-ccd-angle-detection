from collections import Counter
from collections import deque
import cv2 as cv
import csv
import copy
from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
import mediapipe as mp
from gesture_module import GestureUtils


class GestureRecognition:
    def __init__(self, use_static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7,
                 history_length=16):
        self.use_static_image_mode = use_static_image_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.history_length = history_length

        # import utils class
        self.gesture_utils = GestureUtils()

        # Load models
        self.hands, self.keypoint_classifier, self.keypoint_classifier_labels, \
        self.point_history_classifier, self.point_history_classifier_labels = self.load_model()

        # Finger gesture history
        self.point_history = deque(maxlen=history_length)
        self.finger_gesture_history = deque(maxlen=history_length)

    def load_model(self):
        # Model load #############################################################
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        keypoint_classifier = KeyPointClassifier()
        point_history_classifier = PointHistoryClassifier()

        # Read labels ###########################################################
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
        with open(
                'model/point_history_classifier/point_history_classifier_label.csv',
                encoding='utf-8-sig') as f:
            point_history_classifier_labels = csv.reader(f)
            point_history_classifier_labels = [
                row[0] for row in point_history_classifier_labels
            ]

        return hands, keypoint_classifier, keypoint_classifier_labels, \
               point_history_classifier, point_history_classifier_labels

    def recognize(self, image, number=-1, mode=0):
        """

        :param image:
        :param number:
        :param mode:
        :return: debug_image, gesture_id
        """

        # TODO: Move constants to other place
        USE_BRECT = True

        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Saving gesture id for drone controlling
        gesture_id = -1

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = self.gesture_utils.calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = self.gesture_utils.calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = self.gesture_utils.pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = self.gesture_utils.pre_process_point_history(
                    debug_image, self.point_history)

                # Write to the dataset file
                self.gesture_utils.logging_csv(number, mode, pre_processed_landmark_list,
                                               pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    self.point_history.append(landmark_list[8])
                else:
                    self.point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (self.history_length * 2):
                    finger_gesture_id = self.point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                self.finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    self.finger_gesture_history).most_common()

                # Drawing part
                debug_image = self.gesture_utils.draw_bounding_rect(USE_BRECT, debug_image, brect)
                debug_image = self.gesture_utils.draw_landmarks(debug_image, landmark_list)
                debug_image = self.gesture_utils.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    self.keypoint_classifier_labels[hand_sign_id],
                    self.point_history_classifier_labels[most_common_fg_id[0][0]]
                )

                # Saving gesture
                gesture_id = hand_sign_id
        else:
            self.point_history.append([0, 0])

        debug_image = self.gesture_utils.draw_point_history(debug_image, self.point_history)

        return debug_image, gesture_id


class GestureBuffer:
    def __init__(self, buffer_len=10):
        self.buffer_len = buffer_len
        self._buffer = deque(maxlen=buffer_len)

    def add_gesture(self, gesture_id):
        self._buffer.append(gesture_id)

    def get_gesture(self):
        counter = Counter(self._buffer).most_common()
        if counter[0][1] >= (self.buffer_len - 1):
            self._buffer.clear()
            return counter[0][0]
        else:
            return
