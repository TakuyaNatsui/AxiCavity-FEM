"""表示中の結果（プロジェクトの結果、または外部の h5）と、選択（n / 位相 / モード / 時間位相）・表示オプション.

結果ビュー（:class:`~axicavity_fem.gui.ui.result_view.ResultView`）と結果パネル、リボンの結果タブはこのモデルに追従する。
ファイルの読み込みは同期（h5 の読み込みは速い。場の再構成は描画のときに行い、レンダラがキャッシュする）。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from PySide6 import QtCore

from ..io.axiproj import ResultEntry, raw_file_name
from .result_renderer import ResultData, ResultRenderer, Selection, ViewOptions, load_result


def result_source(entry: ResultEntry) -> Optional[Path]:
    """結果フォルダで表示に使う h5（post 済みがあればそれ、無ければ生の結果）."""
    processed = entry.file("processed")
    if processed is not None and processed.exists():
        return processed
    raw = entry.file("raw") or entry.dir / raw_file_name(entry.kind)
    return raw if raw.exists() else None


class ResultModel(QtCore.QObject):
    """Signals:
        loaded(): 表示する結果が変わった（読み込み・読み直し・クリア・エラー）。
        selection_changed(): n / 位相 / モード / 時間位相が変わった。
        options_changed(): 表示オプションが変わった。
    """

    loaded = QtCore.Signal()
    selection_changed = QtCore.Signal()
    options_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entry: Optional[ResultEntry] = None
        self.path: Optional[Path] = None
        self.data: Optional[ResultData] = None
        self.renderer: Optional[ResultRenderer] = None
        self.error: str = ""
        self.selection = Selection()
        self.options = ViewOptions()
        self._entry_key: Optional[tuple] = None

    # ---- 問い合わせ -----------------------------------------------------

    @property
    def has_data(self) -> bool:
        return self.data is not None

    @property
    def is_hom(self) -> bool:
        return self.data is not None and self.data.is_hom

    @property
    def is_traveling(self) -> bool:
        return self.data is not None and self.data.analysis_type(self.selection.n) == "traveling"

    def source_name(self) -> str:
        if self.path is None:
            return ""
        return self.path.name

    def output_base(self, suffix: str) -> Path:
        """書き出し（場・GIF・PNG）の既定のパス. プロジェクトの結果は ``exports/``、外部の h5 はその隣."""
        assert self.path is not None
        sel = self.selection
        stem = self.path.stem + suffix + f"_m{sel.mode}"
        if self.is_hom:
            stem += f"_n{sel.n}"
        if self.entry is not None:
            return self.entry.dir / "exports" / stem
        return self.path.with_name(stem)

    @staticmethod
    def entry_key(entry: ResultEntry) -> tuple:
        return (entry.id, entry.status, entry.finished_at, tuple(sorted((entry.files or {}).items())))

    # ---- 読み込み -------------------------------------------------------

    def load_entry(self, entry: ResultEntry) -> bool:
        source = result_source(entry)
        if source is None:
            self._set_error(entry, None, "結果ファイル（h5）がありません")
            return False
        ok = self._load(source, entry)
        self._entry_key = self.entry_key(entry) if ok else None
        return ok

    def load_file(self, path: str | Path) -> bool:
        return self._load(Path(path), None)

    def refresh_entry(self, entry: Optional[ResultEntry]) -> None:
        """プロジェクトの結果一覧が作り直されたあと、表示中の結果を追従させる（post 後は読み直す）."""
        if self.entry is None:
            return
        if entry is None:
            self.clear()
            return
        key = self.entry_key(entry)
        if key == self._entry_key:
            self.entry = entry
            return
        self.load_entry(entry)

    def clear(self) -> None:
        self.entry = self.path = self.data = self.renderer = None
        self.error = ""
        self._entry_key = None
        self.loaded.emit()

    def _set_error(self, entry: Optional[ResultEntry], path: Optional[Path], message: str) -> None:
        self.entry, self.path, self.data, self.renderer = entry, path, None, None
        self.error = message
        self._entry_key = None
        self.loaded.emit()

    def _load(self, path: Path, entry: Optional[ResultEntry]) -> bool:
        try:
            data = ResultData(path, load_result(path))
        except Exception as exc:  # noqa: BLE001 — 壊れた h5・旧形式など。UI に出す
            self._set_error(entry, path, f"{type(exc).__name__}: {exc}")
            return False
        same_file = self.path is not None and self.path.resolve() == path.resolve()
        self.entry, self.path, self.data, self.renderer, self.error = entry, path, data, ResultRenderer(data), ""
        if same_file:
            self.selection = data.normalize(self.selection)      # post の読み直し: 選択は保つ
        else:
            self.selection = data.normalize(Selection(n=data.n_orders[0]))
            if data.is_hom:
                self.options = replace(self.options, show_lines=False, show_vectors=True)   # ver2.3 の既定
        self.loaded.emit()
        return True

    # ---- 選択・オプション ------------------------------------------------

    def set_selection(self, **changes) -> None:
        sel = replace(self.selection, **changes)
        if self.data is not None:
            sel = self.data.normalize(sel)
        if sel != self.selection:
            self.selection = sel
            self.selection_changed.emit()

    def set_options(self, **changes) -> None:
        opts = replace(self.options, **changes)
        if opts != self.options:
            self.options = opts
            self.options_changed.emit()
