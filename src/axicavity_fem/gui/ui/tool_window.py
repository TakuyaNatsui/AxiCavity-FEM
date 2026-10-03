"""結果の図（2D 断面図・S パラメータ図）を出す別ウィンドウ.

メインウィンドウが持ち主の `Qt.Window`（いつもメインウィンドウの上に出て、最小化も一緒。タスクバーには出ない）。
最大化や別モニタへの移動ができる。閉じるボタンでは隠すだけで closed を出す（リボンのトグルを戻す）。
初めて開くときはメインウィンドウの中央に FRACTION の大きさで置き（`offset` だけずらす）、その後は利用者が
動かした位置と大きさのまま開く。``settings_key`` があれば位置と大きさを設定に保存し、次に起動したときも再現する。

下のドックに置いていた図は小さくて見づらかった（2026-09-23 のユーザー要望で 2D 断面図、続いて S パラメータ図を
別ウィンドウに）。浮動ドックにしないのは、Windows でタイトルバーをダブルクリックすると最大化ではなくドックに
戻ってしまい、「小さく出る」元の問題に戻るため。
"""

from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets

from ..i18n.tr import tr
from .app_settings import clear_window_geometry, load_window_geometry, save_window_geometry

# 初期の大きさ（メインウィンドウに対する割合）と最小
FRACTION = (0.6, 0.72)
MIN_SIZE = (720, 560)


class ToolWindow(QtWidgets.QWidget):
    """図のウィジェット 1 つを入れる別ウィンドウ（`content`）."""

    closed = QtCore.Signal()

    def __init__(self, title_key: str, content: QtWidgets.QWidget,
                 parent: Optional[QtWidgets.QWidget] = None, offset: tuple[int, int] = (0, 0),
                 settings_key: Optional[str] = None):
        """``settings_key`` を渡すと、位置と大きさをアプリの設定に保存して次回も再現する（メインウィンドウ）."""
        super().__init__(parent, QtCore.Qt.Window)
        self.title_key = title_key
        self.content = content
        self.offset = offset
        self.settings_key = settings_key
        self.setWindowTitle(tr(title_key))
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(content)
        self._placed = False

    def open_window(self) -> None:
        """開いて前面に出す（初回は前回の位置、無ければメインウィンドウの中央に置く）."""
        if not self._placed:
            self._placed = True
            saved = load_window_geometry(self.settings_key) if self.settings_key else None
            if saved is None or not self.restoreGeometry(saved):
                self._place_over(self.parentWidget())
        if self.isMinimized():
            self.showNormal()
        else:
            self.show()
        self.raise_()
        self.activateWindow()

    def _place_over(self, owner: Optional[QtWidgets.QWidget]) -> None:
        screen = owner.screen() if owner is not None else QtWidgets.QApplication.primaryScreen()
        area = screen.availableGeometry()
        frame = owner.frameGeometry() if owner is not None else area
        w = min(area.width(), max(MIN_SIZE[0], int(frame.width() * FRACTION[0])))
        h = min(area.height(), max(MIN_SIZE[1], int(frame.height() * FRACTION[1])))
        x = min(max(frame.center().x() - w // 2 + self.offset[0], area.left()), area.right() + 1 - w)
        y = min(max(frame.center().y() - h // 2 + self.offset[1], area.top()), area.bottom() + 1 - h)
        self.resize(w, h)
        self.move(x, y)

    def reset_placement(self) -> None:
        """次に開くときはメインウィンドウの中央に置く（「レイアウトを初期状態に戻す」）."""
        self._placed = False
        if self.settings_key:
            clear_window_geometry(self.settings_key)
        if self.isVisible():
            self._placed = True
            self._place_over(self.parentWidget())

    def hideEvent(self, event) -> None:
        if self.settings_key and self._placed:
            save_window_geometry(self.settings_key, self.saveGeometry())
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        super().closeEvent(event)
        self.closed.emit()

    def retranslate(self) -> None:
        self.setWindowTitle(tr(self.title_key))
        retranslate = getattr(self.content, "retranslate", None)
        if retranslate is not None:
            retranslate()
