"""Chinese plot fonts must render actual glyphs without downloading at runtime."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from matplotlib import font_manager
from matplotlib.ft2font import FT2Font

SCRIPT = Path(__file__).parents[1] / "scripts/static_context_fonts.py"
SPEC = importlib.util.spec_from_file_location("static_context_fonts", SCRIPT)
FONTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FONTS)


class StaticContextFontsTest(unittest.TestCase):
    def test_default_font_covers_required_chinese(self):
        family = FONTS.configure_chinese_font()
        self.assertIsInstance(family, str)
        self.assertTrue(family)
        path = font_manager.findfont(
            font_manager.FontProperties(family=family), fallback_to_default=False
        )
        characters = {ord(c) for c in FONTS.REQUIRED_CHARS if not c.isspace()}
        self.assertTrue(any(c >= 0x4E00 for c in characters))
        self.assertFalse(characters - set(FT2Font(path).get_charmap()))

    def test_explicit_latin_font_cannot_silently_emit_missing_glyphs(self):
        path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
        with self.assertRaises((ValueError, RuntimeError)):
            FONTS.configure_chinese_font(path)

    def test_missing_explicit_font_is_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-cjk-font.ttf"
            with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)) as raised:
                FONTS.configure_chinese_font(missing)
            self.assertTrue(str(raised.exception))


if __name__ == "__main__":
    unittest.main()
