from pathlib import Path
import re

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "python_src/wavedispx/ui/app.py"
HTML_MANUAL = ROOT / "docs/WaveDispX_UI_MANUAL_CN.html"


def test_streamlit_app_renders_all_workflow_pages_without_forward_modelling():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "1 建立算例",
        "2 输入审核",
        "3 运行",
        "4 结果与导出",
        "方法说明",
    ]
    assert app.title[0].value == "WaveDispX 波场正演与频散分析"
    labels = {button.label for button in app.button}
    assert {"载入模板", "读取 TOML", "校验并绘制输入"}.issubset(labels)

    submit = next(button for button in app.button if button.label == "校验并绘制输入")
    app = submit.click().run()
    assert not app.exception
    labels = {button.label for button in app.button}
    assert {
        "1 正演/传递函数",
        "2 观测记录/NCF",
        "3 频散与拾取",
        "4 保存结果",
        "一键运行全部",
    }.issubset(labels)


def test_streamlit_app_runs_the_default_quick_case_end_to_end():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    submit = next(button for button in app.button if button.label == "校验并绘制输入")
    app = submit.click().run()
    run_all = next(button for button in app.button if button.label == "一键运行全部")
    app = run_all.click().run(timeout=60)
    assert not app.exception
    workflow = app.session_state.case_session.workflow
    assert workflow.artifacts is not None
    assert workflow.artifacts.results_h5.is_file()
    assert workflow.artifacts.summary_png.is_file()
    assert (workflow.artifacts.output_directory / "input.toml").is_file()


def test_streamlit_dependent_inputs_switch_without_submitting():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    receiver_layout = next(item for item in app.radio if item.label == "Geometry 输入方式")
    app = receiver_layout.set_value("grid").run()
    number_labels = {item.label for item in app.number_input}
    assert {"x 起点 (m)", "y 起点 (m)", "径向距离箱 (m)"}.issubset(number_labels)

    mode = next(item for item in app.selectbox if item.label == "模式")
    app = mode.set_value("active").run()
    number_labels = {item.label for item in app.number_input}
    assert {"炮点 x (m)", "炮点 y (m)", "Ricker 主频 (Hz)"}.issubset(number_labels)
    radio_labels = {item.label for item in app.radio}
    assert "炮点输入方式" in radio_labels


def test_streamlit_invalid_parameter_shows_location_problem_and_fix():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    case_name = next(item for item in app.text_input if item.label == "算例名称")
    app = case_name.set_value("").run()
    submit = next(button for button in app.button if button.label == "校验并绘制输入")
    app = submit.click().run()
    assert not app.exception
    assert any("run.name" in item.value for item in app.error)
    markdown = "\n".join(item.value for item in app.markdown)
    assert "位置：" in markdown
    assert "基础参数" in markdown
    assert "问题：" in markdown
    assert "怎么改：" in markdown


def test_every_visible_parameter_widget_has_chinese_hover_help():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    for collection in (
        app.text_input,
        app.number_input,
        app.selectbox,
        app.radio,
        app.slider,
        app.checkbox,
    ):
        for widget in collection:
            assert widget.help, f"missing help: {widget.label}"
            assert any("\u4e00" <= character <= "\u9fff" for character in widget.help)

    mode = next(item for item in app.selectbox if item.label == "模式")
    app = mode.set_value("active").run()
    for collection in (app.text_input, app.number_input, app.selectbox, app.radio, app.slider):
        for widget in collection:
            assert widget.help, f"missing active help: {widget.label}"


def test_html_manual_contains_documented_cases_and_all_local_images():
    content = HTML_MANUAL.read_text()
    for case_name in (
        "manual_active_line",
        "manual_active_array",
        "manual_passive_line",
        "manual_passive_array",
    ):
        assert case_name in content
        assert (ROOT / "examples" / f"{case_name}.toml").is_file()
    for case_name in ("active_high_modes_3c", "passive_high_modes_3c"):
        assert case_name in content
        assert (ROOT / "examples" / f"{case_name}.toml").is_file()
    image_sources = re.findall(r'<img src="([^"]+)"', content)
    assert len(image_sources) == 22
    assert len(set(image_sources)) == 22
    assert all((HTML_MANUAL.parent / source).is_file() for source in image_sources)
