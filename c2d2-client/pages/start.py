from PyQt5 import QtGui, QtCore, QtWidgets
from utils import Page
from qtwidgets import AnimatedToggle
import skimage
from src import UNet, resize_img, extract_line, getSlope, getAngle, transparent_cmap, \
    extract_line_huber, reverseResizing
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2


class StartPage(Page):
    scan_btn = QtCore.pyqtSignal()

    angleL = 0
    angleR = 0
    scan_path = ""
    scan_data = None
    shaft_centerL = None
    shaft_centerR = None
    neck_centerL = None
    neck_centerR = None
    fixed_image_size = (512, 512)

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        root_widget = QtWidgets.QWidget(self)

        self.c2_img = QtGui.QPixmap("assets/images/c2-d2.png")
        self.c2_img_label = QtWidgets.QLabel()
        self.c2_img_label.setPixmap(self.c2_img)

        self.ccd_label = QtWidgets.QLabel()
        self.ccd_label.setText("CCD Angle Detection\n         System")
        self.ccd_label.setContentsMargins(10, 10, 10, 10)
        self.ccd_label.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        self.ccd_label.setLineWidth(2)
        self.ccd_label.setObjectName("ccd_label")

        self.start_btn = QtWidgets.QPushButton("SCAN IMAGE")
        self.start_btn.clicked.connect(self.on_start_btn_click)

        root_widget.layout = QtWidgets.QVBoxLayout(root_widget)
        root_widget.layout.setContentsMargins(0, 0, 0, 0)
        root_widget.layout.addWidget(self.c2_img_label, 0, QtCore.Qt.AlignCenter)
        root_widget.layout.addWidget(self.ccd_label, 0, QtCore.Qt.AlignCenter)
        root_widget.layout.addWidget(self.start_btn, 0, QtCore.Qt.AlignCenter)
        self.content.layout.addWidget(root_widget)

    def on_start_btn_click(self):
        scale_factor = 1
        image_size = scale_factor * 512
        img_filter = "Images (*.png *.xpm .jpg)"
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(filter=img_filter)
        try:
            if not file_path:
                return
            image = torchvision.io.read_image(file_path, mode=torchvision.io.ImageReadMode.GRAY)

            print("image shape: ", image.shape)
            print(file_path)

            # set the scan data
            self.scan_path = file_path
            image_data = skimage.io.imread(self.scan_path)
            self.scan_data = image_data

            # get the model prediction for the uploaded image
            output_array = self.start_prediction(image)  # give the image to the model to predict labels

            shaft_center_left_start, shaft_center_left_end = extract_line_huber(output_array[8])
            shaft_center_right_start, shaft_center_right_end = extract_line_huber(output_array[9])
            neck_center_left_start, neck_center_left_end = extract_line_huber(output_array[10])
            neck_center_right_start, neck_center_right_end = extract_line_huber(output_array[11])


            #                   endpoint1   endpoint2
            #                       x   y    x    y
            # self.shaft_centerL = [[10, 10], [100, 900]] #points[0]
            # self.shaft_centerR = [[30, 10], [400, 900]] #points[1]
            # self.neck_centerL = [[50, 10], [600, 900]] #points[2]
            # self.neck_centerR = [[70, 10], [800, 900]] #points[3]

            self.shaft_centerL = [list(shaft_center_left_end), list(shaft_center_left_start)]  # points[0]
            self.shaft_centerL = torch.tensor(self.shaft_centerL)
            self.shaft_centerL = reverseResizing(image, self.shaft_centerL, self.fixed_image_size)
            print("shaft line left: ", self.shaft_centerL)

            self.shaft_centerR = [list(shaft_center_right_start), list(shaft_center_right_end)]  # points[1]
            self.shaft_centerR = torch.tensor(self.shaft_centerR)
            self.shaft_centerR = reverseResizing(image, self.shaft_centerR, self.fixed_image_size)
            print("shaft line right: ", self.shaft_centerR)

            self.neck_centerL = [list(neck_center_left_start), list(neck_center_left_end)]  # points[2]
            self.neck_centerL = torch.tensor(self.neck_centerL)
            self.neck_centerL = reverseResizing(image, self.neck_centerL, self.fixed_image_size)

            self.neck_centerR = [list(neck_center_right_start), list(neck_center_right_end)]  # points[3]
            self.neck_centerR = torch.tensor(self.neck_centerR)
            self.neck_centerR = reverseResizing(image, self.neck_centerR, self.fixed_image_size)

            # caclulate angle from the lines
            m_shaft_L = getSlope([self.shaft_centerL[0], self.shaft_centerL[1]])
            m_shaft_R = getSlope([self.shaft_centerR[0], self.shaft_centerR[1]])
            m_neck_L = getSlope([self.neck_centerL[0], self.neck_centerL[1]])
            m_neck_R = getSlope([self.neck_centerR[0], self.neck_centerR[1]])

            new_angleL = getAngle(m_shaft_L, m_neck_L)
            new_angleR = getAngle(m_shaft_R, m_neck_R)

            # (180 -) takes the correct angle between the two lines
            self.angleL = new_angleL
            self.angleR = 180 - new_angleR

            self.scan_btn.emit()

        except:
            raise Exception("Couldn't open the file. please try again!")

    def start_prediction(self, image):
        """
        we get an image and then use the pretrained model to predict 12 heatmap lines
        """
        # angleL, angelR, shaft_centerL, shaft_centerR, neck_centerL, neck_centerR, = startPrediction(fn)
        # --> interface to network

        #TODO adjust the path
        PATH = "C://Fateme//FAU//5//Intraoperative_imaging//ccd-project//iiml-22-ccd-angle-detection//" \
               "c2d2-client//model//ccd_angle_model//checkpoints//epoch=799-step=24800_maryam.ckpt"
        checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
        model = UNet(in_channels=1, num_classes=12)
        # model.load_state_dict(checkpoint['state_dict'])  # model.load_state_dict(checkpoint)
        state_dict = checkpoint['state_dict']

        # Create a new state_dict that maps the keys from the saved model to the corresponding keys
        # in the current model
        new_state_dict = {}
        for key, value in state_dict.items():
            if 'model.' in key:
                key = key.replace('model.', '')
            new_state_dict[key] = value

        # Load the new state_dict into the current model
        model.load_state_dict(new_state_dict)

        # Resize the input image
        resizedImg = resize_img(image, self.fixed_image_size)

        # Add a batch dimension to the input tensor
        resizedImg = resizedImg.unsqueeze(0)

        output = model(resizedImg)
        output_array = output.squeeze().detach().cpu().numpy()  # Remove the batch dimension and move the tensor to CPU

        # output_heatmap = np.zeros((output_array.shape[1], output_array.shape[2]))

        # for i in range(8, 12):
        #     output_heatmap += output_array[i]  # Add up the heatmaps for indices 8-11
        #
        # plt.imshow(resizedImg.squeeze(), cmap="gray")
        # plt.imshow(output_heatmap, cmap='gray')  # Display the heatmap image
        # plt.show()

        return output_array
