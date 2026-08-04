import os
from pathlib import Path
import re
import time

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Importing the package first applies the macOS pip-PySide/Conda-Qt runtime
# guard before any Qt library is loaded.
import wavedispx.desktop  # noqa: E402,F401

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QComboBox  # noqa: E402

from wavedispx.app_service import load_template_mapping  # noqa: E402
from wavedispx.desktop.app import MainWindow, TEMPLATES  # noqa: E402
from wavedispx.desktop.config_editor import ConfigTreeEditor  # noqa: E402
from wavedispx.desktop.help_text import (  # noqa: E402
    canonical_path,
    has_specific_help,
    parameter_detail_html,
    parameter_spec,
)


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance() or QApplication([])
    yield application


def test_config_tree_round_trip_preserves_complete_mapping(qapp):
    mapping = load_template_mapping(TEMPLATES["被动源｜二维面阵"])
    editor = ConfigTreeEditor()
    editor.set_mapping(mapping)
    assert editor.mapping() == mapping
    assert editor.tree.topLevelItemCount() == len(mapping)


def test_every_example_scalar_has_structured_chinese_guidance():
    root = Path(__file__).resolve().parents[1]
    scalar_paths: set[str] = set()

    def collect(value, path=()):
        if isinstance(value, dict):
            for key, child in value.items():
                collect(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect(child, (*path, f"[{index}]"))
        else:
            scalar_paths.add(canonical_path(".".join(path)))

    for template in (root / "examples").rglob("*.toml"):
        collect(load_template_mapping(template))

    assert len(scalar_paths) >= 69
    for path in scalar_paths:
        assert has_specific_help(path), f"missing structured help: {path}"
        spec = parameter_spec(path)
        for field in (
            spec.label,
            spec.unit,
            spec.summary,
            spec.constraint,
            spec.recommended,
            spec.effect,
        ):
            assert field
        combined = "".join(
            (spec.label, spec.summary, spec.constraint, spec.recommended, spec.effect)
        )
        assert any("\u4e00" <= character <= "\u9fff" for character in combined)


def test_hover_tip_contains_meaning_constraints_recommendation_effect_and_live_value():
    mapping = load_template_mapping(TEMPLATES["被动源｜快速检查"])
    html = parameter_detail_html("noise.duration_s", 60.0, mapping, compact=True)
    for heading in ("它控制什么", "允许/约束", "推荐起点", "调大/调小的影响", "当前设置"):
        assert heading in html
    assert "6 个独立窗" in html


@pytest.mark.parametrize(
    ("template", "mode", "receiver_count", "source_count"),
    (
        ("主动源｜快速检查", "active", 16, 1),
        ("被动源｜快速检查", "passive", 41, 24),
        ("被动源｜二维面阵", "passive", 121, 24),
    ),
)
def test_desktop_templates_render_ready_preflight(
    qapp, template, mode, receiver_count, source_count
):
    window = MainWindow(TEMPLATES[template])
    try:
        assert window._review is not None
        assert window._review.ready
        assert window.workflow.config.mode == mode
        assert window._review.diagnostics["receiver_count"] == receiver_count
        assert window._review.diagnostics["source_count"] == source_count
        assert window.review_page.figure._canvas is not None
        duration = window.parameter_page.editor.item_for_path("noise.duration_s")
        if mode == "passive":
            assert duration is not None
            assert duration.text(0) == "记录总时长"
            assert duration.text(2) == "s"
            assert "允许/约束" in duration.toolTip(0)
    finally:
        window.close()


def test_enum_parameter_uses_explained_combo_box(qapp):
    window = MainWindow(TEMPLATES["被动源｜快速检查"])
    try:
        item = window.parameter_page.editor.item_for_path("processing.dispersion_method")
        assert item is not None
        index = window.parameter_page.editor.tree.indexFromItem(item, 1)
        delegate = window.parameter_page.editor.tree.itemDelegate()
        combo = delegate.createEditor(window.parameter_page.editor.tree, None, index)
        assert isinstance(combo, QComboBox)
        assert [combo.itemData(i) for i in range(combo.count())] == [
            "phase_shift",
            "bessel",
            "hankel",
        ]
        assert "相移叠加" in combo.itemText(0)
        combo.setCurrentIndex(combo.findData("bessel"))
        delegate.setModelData(
            combo,
            window.parameter_page.editor.tree.model(),
            index,
        )
        assert window.parameter_page.mapping()["processing"]["dispersion_method"] == "bessel"
    finally:
        window.close()


def test_desktop_active_quick_runs_all_in_background_and_populates_gallery(qapp, tmp_path):
    window = MainWindow(TEMPLATES["主动源｜快速检查"])
    mapping = window.parameter_page.mapping()
    mapping["run"]["output_directory"] = str(tmp_path / "desktop_active")
    window.parameter_page.set_mapping(mapping)
    window._mark_dirty()
    assert window.validate_case(navigate=False)

    window.start_stage("all")
    deadline = time.monotonic() + 45.0
    while window._running and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()

    try:
        assert not window._running
        assert window.workflow.artifacts is not None
        assert (tmp_path / "desktop_active" / "summary.png").is_file()
        assert window.results_page.images.count() >= 1
        assert window.workflow.status()[-1]["status"] == "complete"
    finally:
        window.close()


def test_desktop_launcher_is_declared():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert 'wavedispx-desktop = "wavedispx.desktop.launcher:main"' in pyproject.read_text()


def test_desktop_html_manual_references_real_complete_screenshots():
    root = Path(__file__).resolve().parents[1]
    manual = root / "docs/WaveDispX_DESKTOP_MANUAL_CN.html"
    content = manual.read_text()
    image_sources = re.findall(r'<img class="shot" src="([^"]+)"', content)
    assert len(image_sources) == 8
    assert len(set(image_sources)) == 8
    assert all((manual.parent / source).is_file() for source in image_sources)
    for label in ("主动线阵", "主动二维面阵", "被动线阵", "被动二维面阵"):
        assert label in content
