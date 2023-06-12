import enum
import cv2
from gesture_module import GestureRecognition, GestureBuffer
from PyQt5 import QtGui, QtCore, QtWidgets
import threading
import numpy as np

@enum.unique
class Phase(enum.IntEnum):
    START = enum.auto()
    TRIGGER = enum.auto()
    ACTION = enum.auto()
    END = enum.auto()


class Gesture(QtWidgets.QFrame):
    # define proper signal for activating different buttons
    edit_points_signal = QtCore.pyqtSignal(int)  # enable points 1, 2, 3, 4 for editing
    move_points_signal = QtCore.pyqtSignal(int)  # move points to left:7, right:8, up:9, down:10
    edit_signal = QtCore.pyqtSignal()  # action: enable editing
    store_signal = QtCore.pyqtSignal()  # action: enable storing
    guidance_signal = QtCore.pyqtSignal()  # action: enable guidance

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)

        self.cap = cv2.VideoCapture(0)
        self.video_frame = QtWidgets.QLabel()
        # self.video_frame.setGeometry(0, 0, 100, 80)

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.video_frame, 0)
        self.mode = 0
        self.number = -1
        self.side_switch = 0

        # set variables
        self.gesture_recognizer = GestureRecognition()
        # consider a buffer with length 10 -> return the most common gesture_id within these 10 values
        self.gr_buffer = GestureBuffer(buffer_len=10)
        self.state: Phase = Phase.START  # gesture sequence first is initialized with start phase
        self.c_timer = CountdownTimer(7, on_complete=self.on_timer_complete)  #
        self.action_gesture = None

    def nextFrameSlot(self):
        """ read the webcam data, recognize the gesture, and update the gesture states based on that """
        ret, image = self.cap.read()

        guidance_sign = cv2.imread("assets/images/y-sign.png")
        guidance_sign = cv2.resize(guidance_sign, (150, 200))
        height, width, channels = guidance_sign.shape

        offset = np.array((20, 20))

        debug_image, gesture_id = self.gesture_recognizer.recognize(image, self.number, self.mode)
        # print("gesture id: ", gesture_id)
        self.gr_buffer.add_gesture(gesture_id)  # add the gesture id to the buffer
        self.update_gesture_states()
        debug_image[offset[0]:offset[0] + height, offset[1]:offset[1] + width] = guidance_sign
        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
        debug_image = cv2.resize(debug_image, (400, 300))
        img = QtGui.QImage(debug_image, debug_image.shape[1], debug_image.shape[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)

    def confirm_action(self, action_gesture):
        """ emit the proper signal based on the recognized command """
        if action_gesture in [0, 1, 2, 3]:
            # select the point with number 1, 2, 3, 4
            self.edit_points_signal.emit(action_gesture + 1)

        if action_gesture in [7, 8, 9, 10]:
            # move points to up, down, left, right
            self.move_points_signal.emit(action_gesture)

        elif action_gesture == 5:
            # Guidance
            self.guidance_signal.emit()
        elif action_gesture == 6:
            # Edit
            self.edit_signal.emit()
        # elif action_gesture == 7:
        #     # Store
        #     self.store_signal.emit()

    def start(self):
        """ activating webcam """
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(0)

    def stop(self):
        """" deactivating the webcam """
        self.timer.stop()
        self.cap.release()

    def update_gesture_states(self):
        """ updating machine state based on the command id, the most common gesture in buffer"""
        command_id = self.gr_buffer.get_gesture()
        if command_id is not None and command_id != -1:
            if self.state == Phase.START:
                if command_id == 4:  # Trigger gesture
                    self.state = Phase.TRIGGER
                    self.c_timer.start()

            elif self.state == Phase.TRIGGER:
                if command_id is not None and command_id != 4:  # other gestures than trigger
                    self.action_gesture = command_id
                    self.state = Phase.ACTION
                    self.c_timer.reset()

                elif self.c_timer.expired():
                    self.on_timer_complete()

            elif self.state == Phase.ACTION:
                if command_id == 4:  # Trigger is detected
                    self.confirm_action(self.action_gesture)
                    self.state = Phase.END  # end of the command process
                    self.c_timer.cancel()

                elif self.c_timer.expired():
                    self.on_timer_complete()

            elif self.state == Phase.END:
                if self.c_timer.expired():
                    self.on_timer_complete()

    def on_timer_complete(self):
        """ restart the command sequence state """
        self.state = Phase.START
        self.c_timer.reset()


class CountdownTimer:
    """ implement a timer for limiting the time for one command recognition """
    def __init__(self, duration, on_complete=None):
        self.duration = duration
        self.on_complete = on_complete
        self.timer = None

    def start(self):
        self.timer = threading.Timer(self.duration, self._complete)
        self.timer.start()

    def reset(self):
        if self.timer is not None:
            self.timer.cancel()
            self.start()

    def cancel(self):
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    def expired(self):
        return self.timer is None

    def _complete(self):
        self.timer = None
        if self.on_complete is not None:
            self.on_complete()



