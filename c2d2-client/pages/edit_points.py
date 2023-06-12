import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
from utils import Page
import numpy as np
import cv2
from widgets import Gesture


class EditPointsPage(Page):
    cancel_btn = QtCore.pyqtSignal()
    store_btn = QtCore.pyqtSignal()
    guidance_btn = QtCore.pyqtSignal()

    scan_data = None
    scan_path = None

    shaft_centerL = None
    shaft_centerR = None
    neck_centerL = None
    neck_centerR = None

    angleL = 0
    angleR = 0

    def __init__(self, angleL, angleR, scan_data, scan_path, shaft_centerL, shaft_centerR, neck_centerL, neck_centerR,
                 parent: QtWidgets.QWidget = None):
        super().__init__(parent)

        # init data
        self.scan_path = scan_path
        self.scan_data = scan_data
        self.shaft_centerL = shaft_centerL
        self.shaft_centerR = shaft_centerR
        self.neck_centerL = neck_centerL
        self.neck_centerR = neck_centerR
        self.angleL = angleL
        self.angleR = angleR

        self._side_switch = False
        self._selected_side = None
        self._selected_point = None
        self._selected_index = None
        self._last_pen_color = None

        root_widget = QtWidgets.QWidget(self)
        root_widget.layout = QtWidgets.QHBoxLayout(root_widget)

        # Gesture recognition
        self.gesture_widget = Gesture(parent=root_widget)
        self.gesture_widget.store_signal.connect(lambda: self.on_btn_click(self.store_btn))
        self.gesture_widget.guidance_signal.connect(lambda: self.on_btn_click(self.guidance_btn))
        self.gesture_widget.edit_points_signal.connect(self.edit_req_point)
        self.gesture_widget.move_points_signal.connect(self.move_selected_point)
        self.gesture_widget.edit_signal.connect(self.submit_point_translate)

        # Create Widget for the image
        # TODO check shape of scan_data could be more than 2-D
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

        self.g = Graph()
        self.v.addItem(self.g, ignoreBounds=True)

        # Define positions of nodes
        pos = np.array([
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
        adj = np.array([
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ])

        # Define the symbol and colour to use for each node
        symbols = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
        self.SL_color = (255, 0, 0)
        self.SR_color = (10, 135, 52)
        self.NL_color = (0, 0, 255)
        self.NR_color = (255, 0, 255)
        symbolcolour = np.array([
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
        lines = np.array([
            (255, 0, 0, 255, 3),
            (10, 135, 52, 255, 3),
            (0, 0, 255, 255, 3),
            (255, 0, 255, 255, 3),
        ], dtype=[('red', np.ubyte), ('green', np.ubyte), ('blue', np.ubyte), ('alpha', np.ubyte), ('width', float)])

        # Define text to show next to each symbol
        texts = ["SL1", "SL2", "SR1", "SR2", "NL3", "NL4", "NR3", "NR4"]

        # essential button Box
        button_vbox = QtWidgets.QVBoxLayout()

        self.status = QtWidgets.QPushButton("START")

        self.cancel = QtWidgets.QPushButton("Cancel")
        self.cancel.clicked.connect(lambda: self.on_btn_click(self.cancel_btn))

        self.store = QtWidgets.QPushButton("Store")
        self.store.clicked.connect(lambda: self.on_btn_click(self.store_btn))

        self.guidance = QtWidgets.QPushButton("Guidance")
        self.guidance.clicked.connect(lambda: self.on_btn_click(self.guidance_btn))

        button_vbox.addWidget(self.status)
        button_vbox.addWidget(self.store)
        button_vbox.addWidget(self.guidance)
        button_vbox.addWidget(self.cancel)

        # Angle Buttons
        angle_button_hbox = QtWidgets.QHBoxLayout()
        self.angleL_button = QtWidgets.QPushButton("CCD Angle Left: {:.2f}".format(self.angleL))
        self.angleR_button = QtWidgets.QPushButton("CCD Angle Right: {:.2f}".format(self.angleR))
        angle_button_hbox.addWidget(self.angleR_button)
        angle_button_hbox.addWidget(self.angleL_button)

        # Update the graph
        self.g.scatter.sigPlotChanged.connect(self.update)
        #TODO adjust width of hoverPen
        self.g.setData(pos=pos, adj=adj, pen=lines, size=40, symbolPen=symbolcolour, symbolBrush=symbolcolour,
                       symbol=symbols, pxMode=False, text=texts, hoverable=True, hoverPen=pg.mkPen(color='g', width=5))

        # Left Side
        left_side_layout = QtWidgets.QVBoxLayout()
        left_side_layout.addWidget(self.gesture_widget)
        left_side_layout.addLayout(button_vbox)

        # Right Side
        right_side_layout = QtWidgets.QVBoxLayout()
        right_side_layout.addWidget(self.w)  # contains image and lines
        right_side_layout.addLayout(angle_button_hbox)

        # put everything together
        root_widget.layout.addLayout(left_side_layout)
        root_widget.layout.addLayout(right_side_layout)

        self.content.layout.addWidget(root_widget)

    def on_btn_click(self, btn):
        self.stop()
        btn.emit()

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

    def getSlope(self, points):
        x1, x2, y1, y2 = points[0][0], points[1][0], points[0][1], points[1][1]
        if x2 - x1 == 0:
            # TODO return something different?
            return np.Inf
        return (y2 - y1) / (x2 - x1)

    def getAngle(self, m1, m2):
        theta = np.arctan((m1 - m2) / (1 + m1 * m2))
        return theta * (180 / np.pi)

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
        m_shaft_L = self.getSlope([pos_array[0], pos_array[1]])
        m_shaft_R = self.getSlope([pos_array[2], pos_array[3]])
        m_neck_L = self.getSlope([pos_array[4], pos_array[5]])
        m_neck_R = self.getSlope([pos_array[6], pos_array[7]])

        new_angleL = self.getAngle(m_shaft_L, m_neck_L)
        new_angleR = self.getAngle(m_shaft_R, m_neck_R)

        # arctan returns the angle in [-pi/2, pi/2]
        if new_angleL < 0:
            new_angleL += 180

        if new_angleR < 0:
            new_angleR += 180

        self.shaft_centerL = [pos_array[0], pos_array[1]]
        self.shaft_centerR = [pos_array[2], pos_array[3]]
        self.neck_centerL = [pos_array[4], pos_array[5]]
        self.neck_centerR = [pos_array[6], pos_array[7]]

        # (180 -) takes the correct angle between the two lines
        self.angleL = 180 - new_angleL
        self.angleR = new_angleR

        self.angleL_button.setText("CCD Angle Left: {:.2f}".format(self.angleL))
        self.angleR_button.setText("CCD Angle Right: {:.2f}".format(self.angleR))

    def edit_req_point(self, num):
        if self._last_pen_color is not None:
            self.g.data['symbolPen'][self._selected_index] = self._last_pen_color
        print("color: ", self._last_pen_color)
        self.status.setText(f"EDIT POINT NUMBER {num}")
        side = self._selected_side
        self._selected_index = self._select_point(num, side)

    def move_selected_point(self, move):
        if self._side_switch is False:
            self.set_selected_side(move)
        elif self._side_switch is True:
            self._GR_change_points_coordinate(index=self._selected_index, action=move)

    def set_selected_side(self, move):
        if move == 9:
            self._selected_side = "left"
            self.status.setText("SELECT LEFT SIDE")
        elif move == 10:
            self._selected_side = "right"
            self.status.setText("SELECT RIGHT SIDE")

        self._side_switch = True

    def _select_point(self, num, side):
        index = None

        if num == 1:
            if side == "left":
                index = 0  # Shaft Left point 1
                self._last_pen_color = self.SL_color
            elif side == "right":
                index = 2  # Shaft Right point 1
                self._last_pen_color = self.SR_color

        elif num == 2:
            if side == "left":
                index = 1  # Shaft Left point 2
                self._last_pen_color = self.SL_color
            else:
                index = 3  # Shaft Right point 2
                self._last_pen_color = self.SR_color

        elif num == 3:
            if side == "left":
                index = 4  # Neck Left point 3
                self._last_pen_color = self.NL_color
            else:
                index = 6  # Neck Right point 3
                self._last_pen_color = self.NR_color

        elif num == 4:
            if side == "left":
                index = 5  # Neck Left point 4
                self._last_pen_color = self.NL_color
            else:
                index = 7  # Neck Left point 4
                self._last_pen_color = self.NR_color

        # self._last_pen_color = self.g.data['symbolPen'][index]
        self.g.data['symbolPen'][index] = (255, 255, 0)
        self.g.data['size'] = 80
        self.g.updateGraph()
        return index

    def _GR_change_points_coordinate(self, index, action):

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

        self.g.data['size'] = 80
        self.g.updateGraph()

    def submit_point_translate(self):
        self.status.setText("point's coordinate updated")
        self._side_switch = False


class Graph(pg.GraphItem):
    def __init__(self):
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
        #print("hahashdhahsd")
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

