"""拘束に対する UI 操作（拘束一覧とスケッチ画面のラベルで共用）.

- `select_constraint`: 拘束を選択し、関係するエンティティも選択状態にする
- `edit_dimension_value`: 寸法値（式）の入力ダイアログ。矛盾して解けない値は
  ストアが拒否するので警告を出す
"""

from __future__ import annotations

from typing import Optional

from PySide6 import QtWidgets

from ..app.store import DocumentStore
from ..core.document import Id, SketchConstraint, is_dimension_constraint
from ..core.sketch.model import constraint_entity_ids
from ..i18n.tr import tr


def find_constraint(store: DocumentStore, constraint_id: Id) -> Optional[SketchConstraint]:
    sketch = store.active_sketch()
    if sketch is None:
        return None
    return next((c for c in sketch.constraints if c.id == constraint_id), None)


def select_constraint(store: DocumentStore, constraint_id: Id) -> None:
    c = find_constraint(store, constraint_id)
    if c is None:
        return
    store.set_selected_constraint(constraint_id)
    store.set_selection(constraint_entity_ids(c))


def edit_dimension_value(store: DocumentStore, constraint_id: Id,
                         parent: Optional[QtWidgets.QWidget] = None) -> bool:
    """寸法値の編集ダイアログを出す。値が変わって解けたら True."""
    c = find_constraint(store, constraint_id)
    if c is None or not is_dimension_constraint(c):
        return False
    text, ok = QtWidgets.QInputDialog.getText(
        parent, tr(f"constraints.{c.type}"), tr("params.expression"), text=c.value)
    if not ok:
        return False
    value = text.strip()
    if not value or value == c.value:
        return False
    if store.set_constraint_value(constraint_id, value):
        return True
    info = store.solve_info
    QtWidgets.QMessageBox.warning(
        parent, tr("solve.status.conflict"),
        (tr(info.message) if info and info.message else tr("solve.failed"))
        + (f"\n{info.error}" if info and info.error else ""))
    return False
