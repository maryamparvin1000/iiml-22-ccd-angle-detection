import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from utils import Page
from widgets import Gesture
import cv2
import numpy as np
import enum
from src import getSlope, getAngle


@enum.unique
class Phase(enum.IntEnum):
    RESULT = enum.auto()
    EDIT = enum.auto()


class ResultPage(Page):
    cancel_btn = QtCore.pyqtSignal()
    store_btn = QtCore.pyqtSignal()
    guidance_btn = QtCore.pyqtSignal()
    edit_btn = QtCore.pyqtSignal()

    angleL = 0
    angleR = 0
    scan_path = ""
    scan_data = None

    shaft_centerL = None
    shaft_centerR = None
    neck_centerL = None
    neck_centerR = None

    edit_switch = False
    state = None

    edit_process_labels = None

    def __init__(self, angleL, angleR, scan_path, scan_data, shaft_centerL, shaft_centerR, neck_centerL, neck_centerR,
                 state, index, parent=None

                 ):
        super().__init__(parent)

        self.state: Phase = state

        self.angleL = angleL
        self.angleR = angleR
        self.scan_path = scan_path
        self.scan_data = scan_data

        self.shaft_centerL = shaft_centerL
        self.shaft_centerR = shaft_centerR
        self.neck_centerL = neck_centerL
        self.neck_centerR = neck_centerR

        self.edit_process_labels = ["color: #343c45", "color: #343c45", "color: #343c45", "color: #343c45"]

        self._side_switch = False
        self._selected_side = None
        self._selected_point = None
        self._selected_index = None
        self._last_pen_color = None

        self.index = index
        self.edit_process_labels[index] = "color: white"

        print("in result angleL: ", self.angleL)
        print("in result angleR: ", self.angleR)
        print("in result path: ", self.scan_path)
        root_widget = QtWidgets.QWidget(self)

        self.gesture_widget = Gesture(parent=root_widget)
        self.gesture_widget.edit_signal.connect(self.enable_edit)
        self.gesture_widget.store_signal.connect(lambda: self.on_btn_click(self.store_btn))
        self.gesture_widget.guidance_signal.connect(lambda: self.on_btn_click(self.guidance_btn))
        self.gesture_widget.edit_points_signal.connect(self.edit_req_point)
        self.gesture_widget.move_points_signal.connect(self.move_selected_point)
        # self.gesture_widget.edit_signal.connect(self.submit_point_translate)

        self.w = pg.GraphicsLayoutWidget(show=True, size=self.scan_data.shape, border=False)
        self.v = self.w.addViewBox(enableMouse=False, enableMenu=False)
        self.v.setAspectLocked()
        self.v.setBackgroundColor((4, 10, 17))
        self.v.setBorder(color=(4, 10, 17), width=40)

        self.scan = pg.ImageItem(cv2.rotate(self.scan_data, cv2.ROTATE_90_CLOCKWISE))
        self.v.addItem(self.scan)

        # Create the Bounded Graph and add it to the Image
        self.boundRect = QtCore.QRectF(0, 0, self.scan_data.shape[1], self.scan_data.shape[0])
        self.boundRect.setTopLeft(QtCore.QPointF(0, self.scan_data.shape[0]))
        self.boundRect.setBottomLeft(QtCore.QPointF(0, 0))
        self.boundRect.setTopRight(QtCore.QPointF(self.scan_data.shape[1], self.scan_data.shape[0]))
        self.boundRect.setBottomRight(QtCore.QPointF(self.scan_data.shape[1], 0))

        self.g = Graph(callback=self.get_state)
        self.v.addItem(self.g, ignoreBounds=True)

        # Define positions of nodes
        self.pos = np.array([
            self.shaft_centerL[0],
            self.shaft_centerL[1],
            self.shaft_centerR[0],
            self.shaft_centerR[1],
            self.neck_centerL[0],
            self.neck_centerL[1],
            self.neck_centerR[0],
            self.neck_centerR[1]
        ], dtype=float)

        # Define the set of connections in the graph
        self.adj = np.array([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ])

        # Define the symbol and colour to use for each node
        self.symbols = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
        self.SL_color = (255, 0, 0)
        self.SR_color = (10, 135, 52)
        self.NL_color = (0, 0, 255)
        self.NR_color = (145, 0, 255)
        self.symbolcolour = np.array([
            self.SL_color,
            self.SL_color,
            self.SR_color,
            self.SR_color,
            self.NL_color,
            self.NL_color,
            self.NR_color,
            self.NR_color,
        ], dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte)])

        # Define the line style for each connection
        self.lines = np.array([
            (255, 0, 0, 255, 3),
            (231, 177, 10, 255, 3),
            (0, 0, 255, 255, 3),
            (145, 49, 117, 255, 3),
        ], dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte), ('width', float)])

        # Define text to show next to each symbol
        self.texts = ["SL1", "SL2", "SR1", "SR2", "NL3", "NL4", "NR3", "NR4"]
        self.text_style = {'color': 'white', 'font-size': '20px', 'font-weight': 'bold'}
        self.g.setData(pos=self.pos, adj=self.adj, pen=self.lines, size=40, symbolPen=self.symbolcolour,
                       symbolBrush=self.symbolcolour, symbol=self.symbols, pxMode=False, text=self.texts,
                       textOpts=self.text_style)

        self.process_groupbox = QtWidgets.QGroupBox("Edit Process")
        self.process_groupbox.setCheckable(False)
        self.process_groupbox.setObjectName("process_widget")

        process_widget = QtWidgets.QVBoxLayout()
        self.process_groupbox.setLayout(process_widget)
        print(type(self.edit_process_labels))
        self.side_select_label = QtWidgets.QLabel("1. Select Side (Left/ Right)")
        self.side_select_label.setObjectName("process_label")
        self.side_select_label.setStyleSheet(self.edit_process_labels[0])

        self.point_select_label = QtWidgets.QLabel("2. Select Point (Number 1, 2, 3, 4)")
        self.point_select_label.setObjectName("process_label")
        self.point_select_label.setStyleSheet(self.edit_process_labels[1])

        self.move_action_label = QtWidgets.QLabel("3. Move The Point (L/ R/ U/ D)")
        self.move_action_label.setObjectName("process_label")
        self.move_action_label.setStyleSheet(self.edit_process_labels[2])

        self.submit_edit_label = QtWidgets.QLabel("4. Submit Editing")
        self.submit_edit_label.setObjectName("process_label")
        self.submit_edit_label.setStyleSheet(self.edit_process_labels[3])

        process_widget.addWidget(self.side_select_label)
        process_widget.addWidget(self.point_select_label)
        process_widget.addWidget(self.move_action_label)
        process_widget.addWidget(self.submit_edit_label)

        self.exit_edit = QtWidgets.QPushButton("EXIT EDIT")
        self.exit_edit.hide()
        self.exit_edit.clicked.connect(lambda: self.on_exit_edit_btn())

        self.cancel = QtWidgets.QPushButton("Scan New Image")
        self.cancel.clicked.connect(lambda: self.on_btn_click(self.cancel_btn))

        self.edit_points = QtWidgets.QPushButton("Edit Points")
        self.edit_points.clicked.connect(lambda: self.on_btn_click(self.edit_btn))

        self.guidance = QtWidgets.QPushButton("Display Guidance")
        self.guidance.clicked.connect(lambda: self.on_btn_click(self.guidance_btn))

        self.store = QtWidgets.QPushButton("Store")
        self.store.clicked.connect(lambda: self.on_btn_click(self.store_btn))

        self.angleL_button = QtWidgets.QPushButton("CCD Angle Left: {:.2f}".format(self.angleL))
        self.angleL_button.setObjectName("angle_btn")
        self.angleR_button = QtWidgets.QPushButton("CCD Angle Right: {:.2f}".format(self.angleR))
        self.angleR_button.setObjectName("angle_btn")

        # Top-level layout
        root_widget.layout = QtWidgets.QHBoxLayout(root_widget)

        # mp
        button_vbox = QtWidgets.QVBoxLayout()
        button_vbox.addWidget(self.gesture_widget)
        button_vbox.addWidget(self.process_groupbox)
        button_vbox.addWidget(self.exit_edit)
        button_vbox.addWidget(self.cancel)
        button_vbox.addWidget(self.edit_points)
        button_vbox.addWidget(self.guidance)
        button_vbox.addWidget(self.store)

        # Angle Buttons
        button_hbox = QtWidgets.QHBoxLayout()
        button_hbox.addWidget(self.angleR_button)
        button_hbox.addWidget(self.angleL_button)

        # Left Side
        left_side_layout = QtWidgets.QVBoxLayout()
        left_side_layout.addLayout(button_vbox)

        # Right Side
        right_side_layout = QtWidgets.QVBoxLayout()
        right_side_layout.addWidget(self.w)
        right_side_layout.addLayout(button_hbox)

        root_widget.layout.addLayout(left_side_layout)
        root_widget.layout.addLayout(right_side_layout)

        self.content.layout.addWidget(root_widget)
        self.update_state()

    def on_exit_edit_btn(self):
        print("are you here?")
        self.state = Phase.RESULT
        self.update_state()

    def enable_edit(self):
        self.state = Phase.EDIT
        if self.edit_switch is False:  # in edit state
            self.state = Phase.EDIT
            self.edit_switch = True

        elif self.edit_switch is True:
            self.state = Phase.RESULT
            self.edit_switch = False
        self.update_state()

    def on_btn_click(self, btn):
        if btn != self.edit_btn:
            self.gesture_widget.stop()
            btn.emit()
        else:
            self.state = Phase.EDIT
        self.update_state()

    def update_state(self):
        print("edit state: ", self.edit_switch)
        self.edit_points.setVisible(self.state == Phase.RESULT)
        self.exit_edit.setVisible(self.state == Phase.EDIT)
        self.process_groupbox.setVisible(self.state == Phase.EDIT)
        if self.state == Phase.EDIT:
            self.g.setData(pos=self.pos, adj=self.adj, pen=self.lines, size=40, symbolPen=self.symbolcolour,
                           symbolBrush=self.symbolcolour, symbol=self.symbols, pxMode=False, text=self.texts,
                           textOpts=self.text_style, hoverable=True, hoverPen=pg.mkPen(color='g', width=5))

            self.g.scatter.sigPlotChanged.connect(self.update)
        if self.state == Phase.RESULT:
            self.g.setData(pos=self.pos, adj=self.adj, pen=self.lines, size=40, symbolPen=self.symbolcolour,
                           symbolBrush=self.symbolcolour, symbol=self.symbols, pxMode=False, text=self.texts,
                           textOpts=self.text_style)
            if self._last_pen_color is not None:
                self.g.data['symbolPen'][self._selected_index] = self._last_pen_color
            self.reset_edit_points()

    def start(self):
        self.gesture_widget.start()

    def stop(self):
        self.gesture_widget.stop()

    def checkBoundaries(self, point):
        x_pos = point[0]
        y_pos = point[1]
        new_x_pos = x_pos
        new_y_pos = y_pos
        width, height = self.boundRect.width(), self.boundRect.height()
        if width < 0:
            width *= -1
        if height < 0:
            height *= -1
        if x_pos > width:
            new_x_pos = width
        if x_pos < 0.0:
            new_x_pos = 0
        if y_pos > height:
            new_y_pos = height
        if y_pos < 0.0:
            new_y_pos = 0
        return new_x_pos, new_y_pos

    def checkPositions(self):
        scatter_x = self.g.scatter.data['x']
        scatter_y = self.g.scatter.data['y']
        for ind in range(0, len(self.g.data['pos']), 2):
            tmp1 = self.g.data['pos'][ind]
            tmp2 = self.g.data['pos'][ind + 1]
            tmp3 = [scatter_x[ind], scatter_y[ind]]
            tmp4 = [scatter_x[ind + 1], scatter_y[ind + 1]]
            if not self.boundRect.contains(float(tmp1[0]), float(tmp1[1])):
                new_x_pos, new_y_pos = self.checkBoundaries(tmp1)
                self.g.data['pos'][ind] = [new_x_pos, new_y_pos]
            if not self.boundRect.contains(float(tmp2[0]), float(tmp2[1])):
                new_x_pos, new_y_pos = self.checkBoundaries(tmp2)
                self.g.data['pos'][ind + 1] = [new_x_pos, new_y_pos]
            if not self.boundRect.contains(float(tmp3[0]), float(tmp3[1])):
                new_x_pos, new_y_pos = self.checkBoundaries(tmp3)
                self.g.scatter.data['x'][ind] = new_x_pos
                self.g.scatter.data['y'][ind] = new_y_pos
            if not self.boundRect.contains(float(tmp4[0]), float(tmp4[1])):
                new_x_pos, new_y_pos = self.checkBoundaries(tmp4)
                self.g.scatter.data['x'][ind + 1] = new_x_pos
                self.g.scatter.data['y'][ind + 1] = new_y_pos

    def update(self):
        self.checkPositions()
        pos_array = self.g.data['pos']
        m_shaft_L = getSlope([pos_array[0], pos_array[1]])
        m_shaft_R = getSlope([pos_array[2], pos_array[3]])
        m_neck_L = getSlope([pos_array[4], pos_array[5]])
        m_neck_R = getSlope([pos_array[6], pos_array[7]])

        new_angleL = getAngle(m_shaft_L, m_neck_L)
        new_angleR = getAngle(m_shaft_R, m_neck_R)

        # arctan returns the angle in [-pi/2, pi/2]
        if new_angleL < 0:
            new_angleL += 180

        if new_angleR < 0:
            new_angleR += 180

        self.shaft_centerL = [list(pos_array[0]), list(pos_array[1])]
        self.shaft_centerR = [list(pos_array[2]), list(pos_array[3])]
        self.neck_centerL = [list(pos_array[4]), list(pos_array[5])]
        self.neck_centerR = [list(pos_array[6]), list(pos_array[7])]

        # (180 -) takes the correct angle between the two lines
        self.angleL = new_angleL
        self.angleR = 180 - new_angleR

        self.angleL_button.setText("CCD Angle Left: {:.2f}".format(self.angleL))
        self.angleR_button.setText("CCD Angle Right: {:.2f}".format(self.angleR))

    def edit_req_point(self, num):
        if self._last_pen_color is not None:
            self.g.data['symbolPen'][self._selected_index] = self._last_pen_color

        # self.status.setText(f"EDIT POINT NUMBER {num}")
        self.point_select_label.setStyleSheet("color:white")
        self.index = 1
        self._selected_point = num
        side = self._selected_side
        self.point_select_label.setStyleSheet("color:#343c45")
        self.move_action_label.setStyleSheet("color:white")
        self.index = 2
        self._selected_index = self._select_point(num, side)

    def move_selected_point(self, move):
        if self._side_switch is False:
            self.set_selected_side(move)
        elif self._side_switch is True:
            self._GR_change_points_coordinate(index=self._selected_index, action=move)

    def set_selected_side(self, move):
        if self.state == Phase.EDIT:
            self.side_select_label.setStyleSheet("color:#343c45")
            self.point_select_label.setStyleSheet("color:white")
            self.index = 1
            if move == 9:
                self._selected_side = "left"
            elif move == 10:
                self._selected_side = "right"
            self._side_switch = True

    def _select_point(self, num, side):
        symbol_size = [40, 40, 40, 40, 40, 40, 40, 40]
        point_mapping = {
            (1, "left"): (0, self.SL_color),  # Shaft Left point 1
            (1, "right"): (2, self.SR_color),  # Shaft Right point 1
            (2, "left"): (1, self.SL_color),  # Shaft Left point 2
            (2, "right"): (3, self.SR_color),  # Shaft Right point 2
            (3, "left"): (4, self.NL_color),  # Neck Left point 3
            (3, "right"): (6, self.NR_color),  # Neck Right point 3
            (4, "left"): (5, self.NL_color),  # Neck Left point 4
            (4, "right"): (7, self.NR_color)  # Neck Right point 4
        }
        index, color = point_mapping[(num, side)]
        self._last_pen_color = color
        symbol_size[index] = 70  # Change size of the selected point
        print("symbol size: ", symbol_size)
        self.g.data['symbolPen'][index] = (255, 255, 0)
        self.g.data['size'] = symbol_size  # Update symbol size list
        self.g.updateGraph()
        return index

    def _GR_change_points_coordinate(self, index, action):
        self.submit_edit_label.setStyleSheet("color:white")
        self.index = 3
        self.move_action_label.setStyleSheet("color:#343c45")
        if self._selected_side is not None and self._selected_point is not None:
            # Update the position of the first point
            if action == 7:
                # action_move = "up"
                self.g.data['pos'][index][1] += 100  # y coordinate

            elif action == 8:
                # action_move = "down"
                self.g.data['pos'][index][1] -= 100  # y coordinate

            elif action == 9:
                # action_move = "right"
                self.g.data['pos'][index][0] -= 100  # x coordinate

            elif action == 10:
                # action_move = "left"
                self.g.data['pos'][index][0] += 100  # x coordinate

            # self.g.data['size'] = 80
            self.g.updateGraph()

    def submit_point_translate(self):
        # self.status.setText("point's coordinate updated")
        self._side_switch = False

    def get_state(self):
        if self.state == Phase.EDIT:
            return "EDIT"
        else:
            return "RESULT"

    def reset_edit_points(self):
        # self.status.setText("START")
        self.side_select_label.setStyleSheet("color:white")
        self.index = 0
        self.point_select_label.setStyleSheet("color:#343c45")
        self.move_action_label.setStyleSheet("color: #343c45")
        self.submit_edit_label.setStyleSheet("color:#343c45")
        self._selected_side = None
        self._selected_point = None
        self._selected_index = None
        self._side_switch = False
        self._last_pen_color = None


class Graph(pg.GraphItem):
    def __init__(self, callback):
        self.callback = callback
        self.dragPoint = None
        self.dragOffset = None
        self.textItems = []
        pg.GraphItem.__init__(self)

    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.setTexts(self.text)
        self.updateGraph()

    def setTexts(self, text):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        for t in text:
            item = pg.TextItem(t, (0, 0, 0))
            self.textItems.append(item)
            item.setParentItem(self)

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        for i, item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])

    def mouseDragEvent(self, ev):
        state = self.callback()

        if state == "EDIT":
            if ev.button() != QtCore.Qt.MouseButton.LeftButton:
                ev.ignore()
                return

            if ev.isStart():
                # We are already one step into the drag.
                # Find the point(s) at the mouse cursor when the button was first
                # pressed:
                pos = ev.buttonDownPos()
                pts = self.scatter.pointsAt(pos)
                if len(pts) == 0:
                    ev.ignore()
                    return
                self.dragPoint = pts[0]
                ind = pts[0].data()[0]
                self.dragOffset = self.data['pos'][ind] - pos
            elif ev.isFinish():
                self.dragPoint = None
                return
            else:
                if self.dragPoint is None:
                    ev.ignore()
                    return

            ind = self.dragPoint.data()[0]
            self.data['pos'][ind] = ev.pos() + self.dragOffset
            self.updateGraph()
            ev.accept()
