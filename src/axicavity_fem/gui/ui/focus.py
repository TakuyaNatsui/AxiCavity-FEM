"""ビュー（3D / スケッチ）のキーボードフォーカスの補助（Qt のみ。VTK を import しない）（EM-CAD-py `emcad/viewport/focus.py` からの転用）."""

from __future__ import annotations

from PySide6 import QtWidgets


def text_entry_has_focus() -> bool:
    """文字入力中（編集できる入力欄にフォーカスがある）か.

    3D ビュー / スケッチ画面はマウスが入るとキーボードのフォーカスを取る（F・Esc・作図キーを
    クリックせずに使えるように）が、入力欄で文字を打っている最中は奪わない。
    """
    w = QtWidgets.QApplication.focusWidget()
    if isinstance(w, QtWidgets.QLineEdit):
        return not w.isReadOnly()
    if isinstance(w, (QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
        return not w.isReadOnly()
    if isinstance(w, QtWidgets.QAbstractSpinBox):
        return True
    if isinstance(w, QtWidgets.QComboBox):
        return w.isEditable()
    return False
