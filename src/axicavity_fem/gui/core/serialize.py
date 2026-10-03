"""ドキュメントの JSON 化と読み込み（``to_jsonable`` / エンティティ / 拘束の変換は EM-CAD-py
``emcad/core/serialize.py`` から転用）.

書き出しでは None の項目を省略し、エンティティに判別子 ``type`` を含める。読み込みは欠けた
項目・不正な値を既定値で埋める（手で編集した JSON や将来の版差に耐える）。
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

from . import document as d


class DocumentParseError(ValueError):
    pass


READABLE_VERSIONS = (d.SCHEMA_VERSION,)
_KEY_ORDER = ("schemaVersion", "meta", "params", "sketch", "regions", "boundaries",
              "mesh", "analysis", "post", "report", "view")


# ---------------------------------------------------------------------------
# 書き出し
# ---------------------------------------------------------------------------

def to_jsonable(obj: Any) -> Any:
    """dataclass を再帰的に dict/list に変換する（None は省略、判別子を含める）."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        for disc in ("type", "kind"):
            value = getattr(type(obj), disc, None)
            if isinstance(value, str) and disc not in {
                    f.name for f in dataclasses.fields(obj)}:
                out[disc] = value
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            if value is None:
                continue
            out[f.name] = to_jsonable(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items() if v is not None}
    return obj


def document_to_dict(doc: d.AxiDocument) -> dict:
    data = to_jsonable(doc)
    return {key: data[key] for key in _KEY_ORDER}


def serialize_document(doc: d.AxiDocument) -> str:
    return json.dumps(document_to_dict(doc), indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------

def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _finite_or(value, fallback: float) -> float:
    return float(value) if _is_number(value) else fallback


def _nullable_finite(value):
    return float(value) if _is_number(value) else None


def _optional_str(value) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _vec(value, n: int) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != n or not all(_is_number(v) for v in value):
        raise DocumentParseError(f"Expected a {n}-vector, got {value!r}")
    return tuple(float(v) for v in value)


def _entity(raw: dict):
    cls = d.ENTITY_CLASSES.get(raw.get("type"))
    if cls is None:
        raise DocumentParseError(f"Unknown sketch entity type: {raw.get('type')!r}")
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in names})


def _constraint(raw: dict) -> d.SketchConstraint:
    if raw.get("type") not in d.CONSTRAINT_TYPES:
        raise DocumentParseError(f"Unknown constraint type: {raw.get('type')!r}")
    names = {f.name for f in dataclasses.fields(d.SketchConstraint)}
    return d.SketchConstraint(**{k: v for k, v in raw.items() if k in names})


def _sketch(raw) -> d.SketchFeature:
    if not isinstance(raw, dict):
        return d.SketchFeature(id=d.new_id(), name="Sketch1")
    plane_raw = raw.get("plane") or {}
    plane = d.PlaneRef.origin(str(plane_raw.get("plane", "XY")) if isinstance(plane_raw, dict) else "XY")
    try:
        entities = [_entity(e) for e in raw.get("entities") or []]
        constraints = [_constraint(c) for c in raw.get("constraints") or []]
    except (KeyError, TypeError, ValueError) as exc:
        raise DocumentParseError(f"Invalid sketch: {exc}") from exc
    return d.SketchFeature(id=str(raw.get("id") or d.new_id()), name=str(raw.get("name") or "Sketch1"),
                           plane=plane, suppressed=bool(raw.get("suppressed", False)),
                           entities=entities, constraints=constraints)


def _region(raw) -> d.RegionSetting | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("key"), str):
        return None
    try:
        anchor = _vec(raw.get("anchor"), 2)
    except DocumentParseError:
        return None
    eps_r = _finite_or(raw.get("epsR"), 1.0)
    mu_r = _finite_or(raw.get("muR"), 1.0)
    tan_delta = _finite_or(raw.get("tanDelta"), 0.0)
    return d.RegionSetting(
        key=raw["key"], anchor=anchor,
        name=str(raw.get("name") or d.DEFAULT_REGION_NAME),
        materialTag=str(raw.get("materialTag") or d.DEFAULT_MATERIAL_TAG),
        epsR=eps_r if eps_r > 0 else 1.0, epsRExpr=_optional_str(raw.get("epsRExpr")),
        muR=mu_r if mu_r > 0 else 1.0,
        tanDelta=tan_delta if tan_delta >= 0 else 0.0,
        tanDeltaExpr=_optional_str(raw.get("tanDeltaExpr")))


def _boundaries(raw) -> dict:
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(k, str) and v in d.BC_NAMES}


def _mesh(raw) -> d.MeshSettings:
    raw = raw if isinstance(raw, dict) else {}
    size = _finite_or(raw.get("size"), 5.0)
    return d.MeshSettings(size=size if size > 0 else 5.0, sizeExpr=_optional_str(raw.get("sizeExpr")),
                          order=1 if raw.get("order") == 1 else 2)


def _analysis(raw) -> d.AnalysisSettings:
    raw = raw if isinstance(raw, dict) else {}
    base = d.AnalysisSettings()
    num_modes = round(_finite_or(raw.get("numModes"), base.numModes))
    target = _nullable_finite(raw.get("targetFreqGHz"))
    return d.AnalysisSettings(
        type=raw.get("type") if raw.get("type") in d.ANALYSIS_TYPES else base.type,
        wave=raw.get("wave") if raw.get("wave") in d.WAVE_TYPES else base.wave,
        phases=str(raw.get("phases") or base.phases),
        numModes=max(1, num_modes),
        azOrders=str(raw.get("azOrders") if raw.get("azOrders") is not None else base.azOrders),
        targetFreqGHz=target if target is not None and target > 0 else None,
        runPost=raw.get("runPost") is not False)


def _post(raw) -> d.PostSettings:
    raw = raw if isinstance(raw, dict) else {}
    base = d.PostSettings()
    cond = _finite_or(raw.get("cond"), base.cond)
    beta = _finite_or(raw.get("beta"), base.beta)
    return d.PostSettings(cond=cond if cond > 0 else base.cond, beta=beta if beta > 0 else base.beta)


def _report(raw) -> d.ReportSettings:
    raw = raw if isinstance(raw, dict) else {}
    base = d.ReportSettings()
    dpi = round(_finite_or(raw.get("dpi"), base.dpi))
    return d.ReportSettings(animate=raw.get("animate") is True,
                            timePhase=_finite_or(raw.get("timePhase"), base.timePhase),
                            showMesh=raw.get("showMesh") is not False,
                            dpi=dpi if dpi > 0 else base.dpi)


def _view(raw) -> d.ViewSettings:
    raw = raw if isinstance(raw, dict) else {}
    return d.ViewSettings(**{k: _nullable_finite(raw.get(k)) for k in ("zmin", "zmax", "rmin", "rmax")})


def document_from_dict(obj) -> d.AxiDocument:
    if not isinstance(obj, dict):
        raise DocumentParseError("Document must be an object")
    version = obj.get("schemaVersion")
    if not isinstance(version, int) or isinstance(version, bool) or version not in READABLE_VERSIONS:
        raise DocumentParseError(
            f"Unsupported schema version: {version} (expected {d.SCHEMA_VERSION})")
    meta_raw = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    units = meta_raw.get("units")
    meta = d.DocumentMeta(name=str(meta_raw.get("name", "Untitled")),
                          units=units if units in d.UNITS else "mm",
                          createdAt=str(meta_raw.get("createdAt", "")),
                          modifiedAt=str(meta_raw.get("modifiedAt", "")))
    params = []
    for p in obj.get("params") or []:
        if isinstance(p, dict) and isinstance(p.get("name"), str):
            params.append(d.Param(id=str(p.get("id") or d.new_id()), name=p["name"],
                                  expression=str(p.get("expression", ""))))
    regions = [r for r in map(_region, obj.get("regions") or []) if r is not None]
    return d.AxiDocument(meta=meta, params=params, sketch=_sketch(obj.get("sketch")),
                         regions=regions, boundaries=_boundaries(obj.get("boundaries")),
                         mesh=_mesh(obj.get("mesh")), analysis=_analysis(obj.get("analysis")),
                         post=_post(obj.get("post")), report=_report(obj.get("report")),
                         view=_view(obj.get("view")))


def parse_document(text: str | bytes) -> d.AxiDocument:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DocumentParseError(f"Invalid JSON: {exc}") from exc
    return document_from_dict(raw)
