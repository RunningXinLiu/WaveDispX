"""Memory diagnostics must not prevent importing the Windows application."""
import sys
from types import SimpleNamespace

from wavedispx import artifacts


def test_peak_rss_native():
    assert artifacts._peak_rss_bytes() > 0


def test_peak_rss_units(monkeypatch):
    monkeypatch.setattr(artifacts, "resource", SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=123)))
    monkeypatch.setattr(sys, "platform", "darwin")
    assert artifacts._peak_rss_bytes() == 123
    monkeypatch.setattr(sys, "platform", "linux")
    assert artifacts._peak_rss_bytes() == 123 * 1024


def test_unavailable_is_not_zero(monkeypatch):
    monkeypatch.setattr(artifacts, "resource", None)
    monkeypatch.setattr(sys, "platform", "unsupported")
    assert artifacts._peak_rss_bytes() == -1
