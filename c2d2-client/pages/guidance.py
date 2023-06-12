from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QAbstractItemView, QLabel, QGridLayout, QFrame

from utils import Page
import io


class GuidancePage(Page):
    cancel_btn = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        root_widget = QtWidgets.QWidget(self)
        self.table_widget = QtWidgets.QWidget(self)

        self.cancel = QtWidgets.QPushButton("Back")
        self.cancel.clicked.connect(self.on_cancel_btn_click)

        # Create grid layout widget and set spacing
        self.grid_left = QtWidgets.QGridLayout()
        self.grid_left.setSpacing(5)

        self.grid_right = QtWidgets.QGridLayout()
        self.grid_right.setSpacing(5)

        # Define headers for columns
        command_header_left = QLabel("Commands")
        command_header_left.setAlignment(QtCore.Qt.AlignCenter)
        command_header_left.setFont(QFont('Arial', 12, QFont.Bold))
        gesture_header_left = QLabel("Gesture")
        gesture_header_left.setAlignment(QtCore.Qt.AlignCenter)
        gesture_header_left.setFont(QFont('Arial', 12, QFont.Bold))

        command_header_right = QLabel("Commands")
        command_header_right.setAlignment(QtCore.Qt.AlignCenter)
        command_header_right.setFont(QFont('Arial', 12, QFont.Bold))
        gesture_header_right = QLabel("Gesture")
        gesture_header_right.setAlignment(QtCore.Qt.AlignCenter)
        gesture_header_right.setFont(QFont('Arial', 12, QFont.Bold))


        # Add headers to grid layout
        self.grid_left.addWidget(command_header_left, 0, 0)
        self.grid_left.addWidget(gesture_header_left, 0, 1)
        self.grid_right.addWidget(command_header_right, 0, 0)
        self.grid_right.addWidget(gesture_header_right, 0, 1)



        # define gestures and commands
        gestures_left = ["fist","Scan","y-sign", "i-sign","O_sign"]
        commands_left = ["Trigger (before and after each command)", "Scan Image","Guide Catalodge","Edit Points"
                    ,"Store"]

        gestures_right = ["Up","Down", "Left", "Right", "numbers"]
        commands_right = ["Up","Down","Left", "Right","Switch Between Lines"]

        for i in range(5):
            gesture_label_left = QLabel()
            gesture_pixmap_left = QPixmap(f"assets/images/{gestures_left[i]}.png")
            gesture_label_left.setPixmap(gesture_pixmap_left)
            gesture_label_left.setAlignment(QtCore.Qt.AlignCenter)
            gesture_label_left.setScaledContents(True)

            gesture_label_right = QLabel()
            gesture_pixmap_right = QPixmap(f"assets/images/{gestures_right[i]}.png")
            gesture_label_right.setPixmap(gesture_pixmap_right)
            gesture_label_right.setAlignment(QtCore.Qt.AlignCenter)
            gesture_label_right.setScaledContents(True)

            self.grid_left.setRowMinimumHeight(i+1,gesture_pixmap_left.height() )
            self.grid_right.setRowMinimumHeight(i + 1, gesture_pixmap_right.height())

            command_label_left = QLabel(commands_left[i])
            command_label_left.setAlignment(QtCore.Qt.AlignCenter)

            command_label_right = QLabel(commands_right[i])
            command_label_right.setAlignment(QtCore.Qt.AlignCenter)

            # Add gesture and command labels to grid layout
            self.grid_left.addWidget(command_label_left, i + 1, 0)
            self.grid_left.addWidget(gesture_label_left, i + 1, 1)

            self.grid_right.addWidget(command_label_right, i + 1, 0)
            self.grid_right.addWidget(gesture_label_right, i + 1, 1)

        # stretch the columns and rows when resizing the window
        self.grid_left.setColumnStretch(0, 1)
        self.grid_left.setColumnStretch(1, 2)
        self.grid_left.setRowStretch(0, 1)

        self.grid_right.setColumnStretch(0, 1)
        self.grid_right.setColumnStretch(1, 2)
        self.grid_right.setRowStretch(0, 1)
        for i in range(1, 9):
            self.grid_left.setRowStretch(i, 2)
            self.grid_right.setRowStretch(i, 2)

        # Set borders around cells
        for i in range(6):
            for j in range(2):
                self.grid_left.itemAtPosition(i, j).widget().setStyleSheet("border: 1px solid gray;")
                self.grid_right.itemAtPosition(i, j).widget().setStyleSheet("border: 1px solid gray;")

        root_widget.layout = QtWidgets.QVBoxLayout(root_widget)
        self.table_widget_layout = QtWidgets.QHBoxLayout(self.table_widget)
        self.table_widget_layout.setContentsMargins(0, 0, 0, 0)

        self.table_widget_layout.addLayout(self.grid_left)
        self.table_widget_layout.addLayout(self.grid_right)

        root_widget.layout.setContentsMargins(0, 0, 0, 0)
        root_widget.layout.addWidget(self.table_widget)
        root_widget.layout.addWidget(self.cancel, 0, QtCore.Qt.AlignCenter)
        self.content.layout.addWidget(root_widget)


    def on_cancel_btn_click(self):
        self.cancel_btn.emit()
