from typing import Optional

from PyQt5 import QtCore, QtWidgets

MARGINS = QtCore.QMargins(15, 5, 15, 15)


class Page(QtWidgets.QWidget):
    _header: QtWidgets.QWidget
    _content: QtWidgets.QWidget

    def __init__(
        self,
        parent: QtWidgets.QWidget = None,
    ):
        super().__init__(parent)

        self._content_wrapper = QtWidgets.QWidget(self)

        self._content = QtWidgets.QWidget(self._content_wrapper)
        self._content.layout = QtWidgets.QVBoxLayout(self._content)
        self._content.layout.setContentsMargins(0, 0, 0, 0)

        self._content_wrapper.layout = QtWidgets.QVBoxLayout(
            self._content_wrapper
        )
        self._content_wrapper.layout.setContentsMargins(MARGINS)
        self._content_wrapper.layout.addWidget(self._content)

        self._header = QtWidgets.QWidget(self)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self._header, 0, QtCore.Qt.AlignTop)
        self.layout.addWidget(self._content_wrapper, 1)

    @property
    def header(self) -> QtWidgets.QWidget:
        return self._header

    @header.setter
    def header(self, widget: QtWidgets.QWidget) -> None:
        self.layout.replaceWidget(self._header, widget)
        self._header.deleteLater()
        self._header = widget

    @property
    def content(self) -> QtWidgets.QWidget:
        return self._content

    @content.setter
    def content(self, widget: QtWidgets.QWidget) -> None:
        self._content_wrapper.layout.replaceWidget(self._content, widget)
        self._content.deleteLater()
        self._content = widget

    @property
    def padded(self) -> bool:
        return self._content_wrapper.layout.contentsMargins().left != 0

    @padded.setter
    def padded(self, value: bool) -> None:
        print(value)
        self._content_wrapper.layout.setContentsMargins(
            MARGINS if value is True else QtCore.QMargins(0, 0, 0, 0)
        )