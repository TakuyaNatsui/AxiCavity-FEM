"""GIF アニメーション保存のパラメータ（フレーム数 / fps / 出力先。ver2.3 ``GifAnimationDialog`` の移植）と、
進捗ダイアログ付きの生成（キャンセル可）.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtWidgets

from ...i18n.tr import tr
from ..panel_helpers import MUTED, label
from ..result_renderer import ResultRenderer, Selection, ViewOptions, render_gif


class GifDialog(QtWidgets.QDialog):
    def __init__(self, default_path: str, *, n_frames: int = 36, fps: int = 12, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("gif.title"))
        self.setMinimumWidth(440)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(label(tr("gif.hint"), MUTED))
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        self.frames_spin = QtWidgets.QSpinBox()
        self.frames_spin.setRange(4, 360)
        self.frames_spin.setValue(n_frames)
        form.addRow(tr("gif.frames"), self.frames_spin)
        self.fps_spin = QtWidgets.QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(fps)
        form.addRow(tr("gif.fps"), self.fps_spin)
        row = QtWidgets.QHBoxLayout()
        self.output_edit = QtWidgets.QLineEdit(default_path)
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(32)
        browse.clicked.connect(self._browse)
        row.addWidget(self.output_edit, 1)
        row.addWidget(browse)
        form.addRow(tr("gif.output"), row)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText(tr("gif.save"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, tr("gif.output"), self.output_edit.text(),
                                                        tr("gif.filter"))
        if path:
            if not path.lower().endswith(".gif"):
                path += ".gif"
            self.output_edit.setText(path)

    def params(self) -> dict:
        return {"n_frames": int(self.frames_spin.value()), "fps": int(self.fps_spin.value()),
                "output_path": self.output_edit.text().strip()}


def save_gif_with_progress(renderer: ResultRenderer, sel: Selection, opts: ViewOptions, params: dict,
                           size_px: tuple[int, int], parent: Optional[QtWidgets.QWidget] = None) -> Optional[Path]:
    """進捗ダイアログを出しながら GIF を作る. 保存できたらパス、キャンセルなら None."""
    n_frames = params["n_frames"]
    out = Path(params["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    progress = QtWidgets.QProgressDialog(tr("gif.rendering"), tr("gif.cancel"), 0, n_frames, parent)
    progress.setWindowTitle(tr("gif.title"))
    progress.setWindowModality(QtCore.Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)

    def step(i: int, n: int) -> bool:
        progress.setValue(i)
        progress.setLabelText(tr("gif.frame", i=i + 1, n=n, theta=f"{360.0 * i / n:.1f}"))
        QtWidgets.QApplication.processEvents()
        return not progress.wasCanceled()

    try:
        ok = render_gif(renderer, sel, opts, out, n_frames=n_frames, fps=params["fps"], size_px=size_px,
                        progress=step)
    finally:
        progress.close()
    return out if ok else None
