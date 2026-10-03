"""結果ビュー（中央）: matplotlib の Qt キャンバスに場を描く（ver2.3 Result Viewer の描画ペイン相当）.

``Figure`` + ``FigureCanvasQTAgg`` + ``NavigationToolbar2QT``（パン・ズーム・保存）。描画そのものは
:class:`~axicavity_fem.gui.ui.result_renderer.ResultRenderer`（Qt 非依存）。:class:`ResultModel` の
loaded / selection_changed / options_changed に追従し、隠れているあいだは描かない（表示されたときに描く）。
ダブルクリックで ``point_picked(z, r)``（場の値のポップアップは MainWindow）。
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from ..i18n.tr import tr
from .result_model import ResultModel


class ResultView(QtWidgets.QWidget):
    """Signals:
        point_picked(float, float): 図をダブルクリックした点 (z, r) [m]。
        drawn(): 描き直した（テスト用）。
    """

    point_picked = QtCore.Signal(float, float)
    drawn = QtCore.Signal()

    def __init__(self, model: ResultModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setIconSize(QtCore.QSize(18, 18))
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.toolbar)
        self.last_error: str = ""
        self.last_title: str = ""
        self._dirty = True
        self.canvas.mpl_connect("button_press_event", self._on_click)
        model.loaded.connect(self.invalidate)
        model.selection_changed.connect(self.invalidate)
        model.options_changed.connect(self.invalidate)

    # ---- 描画 -----------------------------------------------------------

    def invalidate(self) -> None:
        self._dirty = True
        if self.isVisible():
            self.redraw()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._dirty:
            QtCore.QTimer.singleShot(0, self.redraw)

    def redraw(self) -> None:
        """今の選択・オプションで描き直す（結果が無ければ案内文）."""
        self._dirty = False
        model = self.model
        fig = self.figure
        if model.data is None or model.renderer is None:
            fig.clf()
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            text = model.error or tr("results.noResult")
            ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, wrap=True,
                    color="#b91c1c" if model.error else "#6b7280")
            self.last_title = ""
            self.last_error = model.error
            self.canvas.draw_idle()
            self.drawn.emit()
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            _freq, sel = model.renderer.draw(fig, model.selection, model.options)
            self.last_title = model.data.title(sel)
            self.last_error = ""
        except Exception as exc:  # noqa: BLE001 — 描けない結果は図の代わりに理由を出す
            fig.clf()
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            self.last_error = f"{type(exc).__name__}: {exc}"
            ax.text(0.5, 0.5, tr("results.drawError", message=self.last_error), ha="center", va="center",
                    transform=ax.transAxes, wrap=True, color="#b91c1c")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.canvas.draw_idle()
        self.drawn.emit()

    def _on_click(self, event) -> None:
        if not getattr(event, "dblclick", False) or event.xdata is None or event.ydata is None:
            return
        if self.model.data is None:
            return
        self.point_picked.emit(float(event.xdata), float(event.ydata))

    # ---- 保存 -----------------------------------------------------------

    def save_png(self, path: str | Path, dpi: int = 150) -> None:
        if self._dirty:
            self.redraw()
        self.figure.savefig(str(path), dpi=dpi)

    def canvas_size(self) -> tuple[int, int]:
        size = self.canvas.size()
        return max(int(size.width()), 400), max(int(size.height()), 300)

    def grab_image(self) -> QtGui.QImage:
        return self.canvas.grab().toImage()
