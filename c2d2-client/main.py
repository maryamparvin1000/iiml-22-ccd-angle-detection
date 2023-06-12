
import sys
from PyQt5 import QtCore, QtWidgets
import pages
from pathlib import Path
import os
import json


class MainWindow(QtWidgets.QWidget):

    angleL = 0
    angleR = 0
    scan_path = ""
    scan_data = None

    shaft_centerL = None
    shaft_centerR = None
    neck_centerL = None
    neck_centerR = None

    result_state = pages.Phase.RESULT

    index = 0

    def __init__(self):

        QtWidgets.QWidget.__init__(self)

        self.start_page = pages.StartPage()
        self.start_page.scan_btn.connect(self._on_scan_img_btn)

        self.stack = QtWidgets.QStackedWidget(self)
        self.stack.addWidget(self.start_page)

        self.content = QtWidgets.QVBoxLayout(self)
        self.content.addWidget(self.stack, 0, QtCore.Qt.AlignCenter)

        self.setLayout(self.content)
        self.setWindowTitle('C2D2')
        self.setMinimumSize(1200, 800)

    def _pop_stack(self) -> None:
        if self.stack.count() > 0:
            widget = self.stack.widget(self.stack.count() - 1)

            if hasattr(widget, "on_cleanup") is True:
                widget.on_cleanup()

            self.stack.removeWidget(widget)
            widget.deleteLater()

    def _push_stack(self, widget: QtWidgets.QWidget) -> None:
        self.stack.addWidget(widget)
        self.stack.setCurrentIndex(self.stack.count() - 1)

    def create_result_page(self):
        self.result_page = pages.ResultPage(self.angleL, self.angleR, self.scan_path, self.scan_data, self.shaft_centerL,
                                            self.shaft_centerR, self.neck_centerL, self.neck_centerR, self.result_state,
                                            self.index)

        self.result_page.cancel_btn.connect(self._on_result_cancel_btn)
        self.result_page.store_btn.connect(self._on_result_store_btn)
        self.result_page.guidance_btn.connect(self._on_result_guidance_btn)
        self.result_page.edit_btn.connect(self._on_result_edit_btn)

        self.result_page.setWindowFlags(QtCore.Qt.Tool)
        self.result_page.start()

    def create_edit_page(self):
        self.edit_page = pages.EditPointsPage(self.angleL, self.angleR, self.scan_data, self.scan_path, self.shaft_centerL,
                                              self.shaft_centerR, self.neck_centerL, self.neck_centerR)
        self.edit_page.cancel_btn.connect(self._on_edit_cancel_btn)
        self.edit_page.guidance_btn.connect(self._on_edit_guidance_btn)
        self.edit_page.store_btn.connect(self._on_edit_store_btn)
        self.edit_page.setWindowFlags(QtCore.Qt.Tool)
        self.edit_page.start()

    def safe_page_state(self, page):
        self.angleL = page.angleL
        self.angleR = page.angleR
        self.scan_path = page.scan_path
        self.scan_data = page.scan_data

        self.shaft_centerL = page.shaft_centerL
        self.shaft_centerR = page.shaft_centerR
        self.neck_centerL = page.neck_centerL
        self.neck_centerR = page.neck_centerR

        if hasattr(page, 'state'):
            self.result_state = page.state

        if hasattr(page, 'index'):
            self.index = page.index

    def _on_scan_img_btn(self):
        self.safe_page_state(self.start_page)
        self.create_result_page()
        self._push_stack(self.result_page)

    def _on_result_edit_btn(self):
        self.create_edit_page()
        self._pop_stack()
        self._push_stack(self.edit_page)
        self.edit_page.start()

    def _on_result_cancel_btn(self):
        self.result_state = pages.Phase.RESULT
        self._pop_stack()

    def _on_result_guidance_btn(self):
        self.page_before_guidance = "result"
        self.safe_page_state(self.result_page)

        self.guidance_page = pages.GuidancePage()
        self.guidance_page.cancel_btn.connect(self._on_guidance_cancel_btn)
        self._pop_stack()
        self._push_stack(self.guidance_page)

    def _on_result_store_btn(self):
        # write the angle, patient information, file_path,... to a file
        self.safe_page_state(self.result_page)
        self.result_state = pages.Phase.RESULT
        dictionary = {
            "angleL": self.angleL,
            "angleR": self.angleR,
            "Femoral Shaft Centerline Left": self.shaft_centerL,
            "Femoral Shaft Centerline Right": self.shaft_centerR,
            "Femoral Neck Centerline Left": self.neck_centerL,
            "Femoral Neck Centerline Right": self.neck_centerR,
            "path_to_image": self.scan_path
        }

        json_object = json.dumps(dictionary, indent=4)
        with open(self.scan_path.replace(".png", ".json"), "w") as outfile:
            outfile.write(json_object)

        self._pop_stack()

    def _on_guidance_cancel_btn(self):
        self._pop_stack()
        self.create_result_page()
        self._push_stack(self.result_page)

    def _on_edit_cancel_btn(self):
        self.safe_edit_page_state()

        self._pop_stack()
        self.create_result_page()
        self._push_stack(self.result_page)

    def _on_edit_guidance_btn(self):
        self.page_before_guidance = "edit"

        self.safe_edit_page_state()

        self.guidance_page = pages.GuidancePage()
        self.guidance_page.cancel_btn.connect(self._on_guidance_cancel_btn)
        self._pop_stack()
        self._push_stack(self.guidance_page)

    def _on_edit_store_btn(self):
        # write the angle, patient information, file_path,... to a file
        self._pop_stack()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    theme_path = (
            Path(os.path.dirname(__file__)) / "assets/theme/theme.css"
    )

    with open(theme_path, "r") as theme_stream:
        app.setStyleSheet(theme_stream.read())

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
