"""Offline Chinese font setup for the static trajectory report.

The small bundled font is an OFL-licensed Noto Sans SC subset. It contains the
report's Chinese labels, Latin letters and digits; it is not a general CJK font.
"""

import base64
import hashlib
import os
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parent / "assets"
BUNDLED_FONT = ASSET_DIR / "score_plot_cjk.ttf.base64"
_renderer = Path(__file__).with_name("static_context_plots.py")
_labels = (
    _renderer.read_text(encoding="utf-8")
    if _renderer.exists()
    else "原始轨迹推动道路自车未来时间位移"
)
REQUIRED_CHARS = "".join(sorted({c for c in _labels if ord(c) > 127 and not c.isspace()}))


def validate_font(path, required_text=""):
    from matplotlib.ft2font import FT2Font

    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"Chinese font file does not exist: {path}")
    chars = set(REQUIRED_CHARS + required_text)
    coverage = FT2Font(str(path)).get_charmap()
    missing = sorted(c for c in chars if not c.isspace() and ord(c) not in coverage)
    if missing:
        raise ValueError(
            f"Font {path} is missing report glyphs: {''.join(missing[:24])}. "
            "Use the bundled font or a complete Noto Sans CJK/Source Han Sans font."
        )
    return path


def bundled_font_path():
    data = base64.b64decode(BUNDLED_FONT.read_bytes())
    sha = hashlib.sha256(data).hexdigest()
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    directory = cache / "score_function" / "fonts"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"ScorePlotCJK-{sha[:16]}.ttf"
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != sha:
        temp = target.with_name(target.name + f".tmp.{os.getpid()}")
        temp.write_bytes(data)
        temp.replace(target)
    return target


def configure_chinese_font(font_path=None, required_text=""):
    """Register a verified local font and return its Matplotlib family name."""
    from matplotlib import font_manager

    explicit = font_path or os.environ.get("SCORE_PLOT_FONT")
    if explicit:
        path = validate_font(explicit, required_text)
    elif BUNDLED_FONT.exists():
        path = validate_font(bundled_font_path(), required_text)
    else:
        path = None
        for family in (
            "Noto Sans CJK SC",
            "Noto Sans SC",
            "Source Han Sans SC",
            "SimHei",
            "Microsoft YaHei",
        ):
            try:
                candidate = font_manager.findfont(family, fallback_to_default=False)
                path = validate_font(candidate, required_text)
                break
            except ValueError:
                continue
        if path is None:
            raise ValueError(
                "Chinese font unavailable. Download scripts/assets/score_plot_cjk.ttf.base64 "
                "from the same commit, or pass --font /path/to/a/Chinese-font.ttf."
            )
    font_manager.fontManager.addfont(str(path))
    return font_manager.FontProperties(fname=str(path)).get_name()
