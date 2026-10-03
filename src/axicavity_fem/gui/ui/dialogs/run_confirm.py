"""「解析実行」の確認ダイアログ（解析の種類と条件・領域と材料・境界条件・メッシュ・出力先・警告。EM-CAD-py 転用）.

表示する内容は実際に子プロセスへ渡す内容（``build_commands`` の argv と解決済みの領域）から作る。
:func:`run_summary` は Qt に依存しない。
"""

from __future__ import annotations

import html
from typing import Optional

from PySide6 import QtCore, QtWidgets

from ...core.convert import ResolvedRegion
from ...core.document import BC_NAMES, AxiDocument
from ...i18n.tr import tr

WARNING_COLOR = "#b45309"


def run_summary(doc: AxiDocument, regions: list[ResolvedRegion], bc_counts: dict[str, int],
                mesh_reused: bool, mesh_elements: Optional[int], output: str, commands: list[str],
                warnings: list[str]) -> list[tuple[str, list[str]]]:
    """確認ダイアログの中身: [(節の見出し, [行, …]), …]（翻訳済み、HTML ではない）."""
    a = doc.analysis
    analysis = [tr("run.solver", kind=tr("run.hom" if a.type == "hom" else "run.tm0"),
                   wave=tr("run.traveling", phases=a.phases) if a.wave == "traveling" else tr("run.standing"))]
    if a.type == "hom":
        analysis.append(tr("run.azOrders", orders=a.azOrders))
    analysis.append(tr("run.modes", count=a.numModes, order=doc.mesh.order))
    analysis.append(tr("run.target", f=f"{a.targetFreqGHz:g}") if a.targetFreqGHz else tr("run.targetAuto"))
    analysis.append(tr("run.post", cond=f"{doc.post.cond:.3g}", beta=f"{doc.post.beta:g}") if a.runPost
                    else tr("run.noPost"))

    region_lines = []
    for r in regions:
        key = "run.regionLoss" if r.tan_delta > 0 else "run.region"
        region_lines.append(tr(key, name=r.name, tag=r.material_tag, eps=f"{r.eps_r:g}", tan=f"{r.tan_delta:g}"))
    boundaries = [tr("run.bc", name=bc, count=bc_counts.get(bc, 0)) for bc in BC_NAMES if bc_counts.get(bc)]
    size = doc.mesh.sizeExpr if doc.mesh.sizeExpr else f"{doc.mesh.size:g}"
    mesh = [tr("run.meshSize", size=size, units=doc.meta.units, order=doc.mesh.order)]
    mesh.append(tr("run.meshReuse", elements=f"{mesh_elements:,}" if isinstance(mesh_elements, int) else "-")
                if mesh_reused else tr("run.meshNew"))
    sections = [
        (tr("run.analysis"), analysis),
        (tr("run.regions"), region_lines or [tr("run.noRegions")]),
        (tr("run.boundaries"), boundaries),
        (tr("run.mesh"), mesh),
        (tr("run.output"), [output] + commands),
    ]
    if warnings:
        sections.append((tr("run.warnings"), list(warnings)))
    return sections


def summary_html(sections: list[tuple[str, list[str]]]) -> str:
    warnings_title = tr("run.warnings")
    parts = []
    for title, lines in sections:
        body = "<br>".join("&nbsp;&nbsp;" + html.escape(line).replace("\n", "<br>&nbsp;&nbsp;") for line in lines)
        if title == warnings_title:
            body = f"<span style='color:{WARNING_COLOR}'>{body}</span>"
        parts.append(f"<b>{html.escape(title)}</b><br>{body}")
    return "<br><br>".join(parts)


def summary_text(sections: list[tuple[str, list[str]]]) -> str:
    return "\n".join(f"[{title}]\n" + "\n".join(f"  {line}" for line in lines) for title, lines in sections)


class RunConfirmDialog(QtWidgets.QDialog):
    """解析の条件を示して「実行 / キャンセル」を聞く（実行が既定のボタン）."""

    def __init__(self, sections: list[tuple[str, list[str]]], parent=None):
        super().__init__(parent)
        self.sections = sections
        self.setWindowTitle(tr("run.title"))
        self.setMinimumWidth(520)
        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel(tr("run.header"))
        header.setWordWrap(True)
        layout.addWidget(header)
        self.body = QtWidgets.QLabel()
        self.body.setTextFormat(QtCore.Qt.RichText)
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.body.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.body.setText(summary_html(sections))
        self.body.setContentsMargins(8, 8, 8, 8)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.body)
        content = self.body.heightForWidth(500)
        scroll.setMinimumHeight(min(560, (content if content > 0 else self.body.sizeHint().height()) + 6))
        layout.addWidget(scroll, 1)
        buttons = QtWidgets.QDialogButtonBox()
        self.run_button = buttons.addButton(tr("run.run"), QtWidgets.QDialogButtonBox.AcceptRole)
        self.run_button.setDefault(True)
        font = self.run_button.font()
        font.setBold(True)
        self.run_button.setFont(font)
        self.cancel_button = buttons.addButton(tr("run.cancel"), QtWidgets.QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def summary_text(self) -> str:
        return summary_text(self.sections)
