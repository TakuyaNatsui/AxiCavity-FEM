"""旧 ``.gmshproj`` の取り込みと書き出し（本体は ``core.legacy``）."""

from ..core.legacy import GMSHPROJ_SUFFIX, load_gmshproj, save_gmshproj  # noqa: F401

FILE_FILTER = "Multi-Region project (*.gmshproj)"
