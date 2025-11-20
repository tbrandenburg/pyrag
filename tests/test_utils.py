"""Tests for PyRAG utilities."""

from pyrag.utils import clip_text, looks_like_url


def test_looks_like_url():
    """Test URL detection function."""
    assert looks_like_url("https://example.com")
    assert looks_like_url("http://example.com")
    assert not looks_like_url("/path/to/file")
    assert not looks_like_url("file.txt")


def test_clip_text():
    """Test text clipping function."""
    short_text = "Short text"
    assert clip_text(short_text) == short_text

    long_text = "a" * 200
    clipped = clip_text(long_text, 150)
    assert len(clipped) == 153  # 150 + "..."
    assert clipped.endswith("...")


def test_clip_text_threshold():
    """Test text clipping with different thresholds."""
    text = "Hello world"
    assert clip_text(text, 5) == "Hello..."
    assert clip_text(text, 50) == "Hello world"
