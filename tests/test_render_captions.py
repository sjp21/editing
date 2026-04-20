"""ASS caption generation — pure-Python, no ffmpeg."""

from __future__ import annotations

import pytest

from clipper.render import (
    _build_ass,
    _drawtext_escape,
    _ffmpeg_escape_path,
    _fmt_ass_time,
    _hex_to_ass,
    _words_in_trim,
)
from clipper.schemas import CaptionStyle, TranscriptWord


def _words(texts: list[str], start: float = 10.0, step: float = 0.5) -> list[TranscriptWord]:
    out = []
    t = start
    for txt in texts:
        out.append(TranscriptWord(start=t, end=t + step, word=txt))
        t += step
    return out


def test_hex_to_ass_bgr_order():
    # #FFD400 -> R=FF G=D4 B=00 -> &H0000D4FF
    assert _hex_to_ass("#FFD400") == "&H0000D4FF"
    assert _hex_to_ass("#FFFFFF") == "&H00FFFFFF"
    assert _hex_to_ass("#000000") == "&H00000000"


def test_fmt_ass_time_centisecond_precision():
    assert _fmt_ass_time(0) == "0:00:00.00"
    assert _fmt_ass_time(1.234) == "0:00:01.23"
    assert _fmt_ass_time(65.5) == "0:01:05.50"
    assert _fmt_ass_time(3661.99) == "1:01:01.99"


def test_words_in_trim_drops_outside_and_truncates_edges():
    words = _words(["a", "b", "c", "d", "e"], start=10.0, step=1.0)
    # words: (10,11),(11,12),(12,13),(13,14),(14,15)
    clipped = _words_in_trim(words, trim_start=11.2, trim_end=13.8)
    # Mids: 10.5, 11.5, 12.5, 13.5, 14.5 — keep 11.5, 12.5, 13.5
    assert [w.word for w in clipped] == ["b", "c", "d"]
    # Edge truncation: first word's start clamped to 11.2
    assert clipped[0].start == pytest.approx(11.2)
    # Last word's end clamped to 13.8
    assert clipped[-1].end == pytest.approx(13.8)


def test_build_ass_is_deterministic():
    words = _words(["hello", "world", "again"], start=5.0, step=1.0)
    style = CaptionStyle(font_size=48, words_visible=3)
    a = _build_ass(words, trim_start=5.0, trim_end=8.0, style=style, width=1080, height=1920)
    b = _build_ass(words, trim_start=5.0, trim_end=8.0, style=style, width=1080, height=1920)
    assert a == b


def test_build_ass_one_active_per_event():
    words = _words(["alpha", "beta", "gamma"], start=0.0, step=1.0)
    style = CaptionStyle(words_visible=3)
    out = _build_ass(words, 0.0, 3.0, style, 1080, 1920)
    # One Dialogue line per word.
    lines = [ln for ln in out.splitlines() if ln.startswith("Dialogue:")]
    assert len(lines) == 3
    # Each dialogue contains exactly one \rActive marker.
    for ln in lines:
        assert ln.count("\\rActive") == 1


def test_build_ass_respects_visible_count():
    words = _words(["one", "two", "three", "four", "five"], start=0.0, step=1.0)
    style = CaptionStyle(words_visible=3)
    out = _build_ass(words, 0.0, 5.0, style, 1080, 1920)
    dialogues = [ln for ln in out.splitlines() if ln.startswith("Dialogue:")]
    # Each line shows 3 words maximum.
    for ln in dialogues:
        # Count tokens after the Inactive prefix.
        payload = ln.split(",,", 1)[1]
        # Count "\rInactive" + "\rActive" markers as visible words.
        n_markers = payload.count("\\rActive") + payload.count("\\rInactive")
        # There's one leading {\\rInactive} setting base style; each word is marked once.
        assert n_markers - 1 <= 3  # subtract the leading default


def test_build_ass_outside_trim_drops_words():
    words = _words(["before", "inside", "after"], start=0.0, step=1.0)
    style = CaptionStyle(words_visible=1)
    # mids: 0.5, 1.5, 2.5 — trim [1.0, 2.0] keeps only "inside"
    out = _build_ass(words, 1.0, 2.0, style, 1080, 1920)
    dialogues = [ln for ln in out.splitlines() if ln.startswith("Dialogue:")]
    assert len(dialogues) == 1
    assert "inside" in dialogues[0]
    assert "before" not in dialogues[0]
    assert "after" not in dialogues[0]


def test_ass_escapes_braces():
    words = [TranscriptWord(start=0.0, end=1.0, word="{meta}")]
    style = CaptionStyle()
    out = _build_ass(words, 0.0, 1.0, style, 1080, 1920)
    # literal braces must be escaped.
    assert "\\{meta\\}" in out


def test_drawtext_escape_apostrophe_uses_close_reopen():
    # ffmpeg's filter-graph parser does not honor backslash escapes inside
    # single-quoted regions — apostrophes must be close-escape-reopen so the
    # surrounding text='…' wrapper survives contractions in hook text.
    assert _drawtext_escape("don't quit") == "don'\\''t quit"
    assert _drawtext_escape("it's what's next") == "it'\\''s what'\\''s next"


def test_ffmpeg_escape_path_handles_filter_delimiters():
    # The filter-graph parser splits on , and treats [ ] as stream labels.
    assert _ffmpeg_escape_path("/tmp/Music, Inc/out") == r"/tmp/Music\, Inc/out"
    assert _ffmpeg_escape_path("/p/[WIP]/out") == r"/p/\[WIP\]/out"
    assert _ffmpeg_escape_path("/p/a:b'c") == r"/p/a\:b\'c"
