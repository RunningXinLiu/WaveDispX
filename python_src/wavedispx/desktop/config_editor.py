"""Type-preserving hierarchical editor for WaveDispX TOML mappings."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIntValidator
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QStyledItemDelegate,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .help_text import (
    SECTION_HELP,
    canonical_path,
    group_label,
    parameter_choices,
    parameter_detail_html,
    parameter_spec,
)


PATH_ROLE = int(Qt.ItemDataRole.UserRole) + 1
TYPE_ROLE = int(Qt.ItemDataRole.UserRole) + 2


class _ValueOnlyDelegate(QStyledItemDelegate):
    @staticmethod
    def _path(index) -> str:
        return str(index.sibling(index.row(), 0).data(PATH_ROLE) or "")

    def createEditor(self, parent, option, index):  # noqa: N802 - Qt API
        path = self._path(index)
        if index.column() != 1 or not path:
            return None
        choices = parameter_choices(path)
        if choices:
            editor = QComboBox(parent)
            for raw, label in choices:
                editor.addItem(f"{label}  [{raw}]", raw)
            editor.setToolTip("从中文选项中选择；写入 TOML 的仍是方括号中的标准值。")
            return editor
        editor = super().createEditor(parent, option, index)
        if isinstance(editor, QLineEdit) and str(index.data(TYPE_ROLE)) == "int":
            editor.setValidator(QIntValidator(-2_147_483_648, 2_147_483_647, editor))
        return editor

    def setEditorData(self, editor, index):  # noqa: N802 - Qt API
        if isinstance(editor, QComboBox):
            raw = str(index.data(Qt.ItemDataRole.EditRole))
            choice_index = editor.findData(raw)
            editor.setCurrentIndex(max(0, choice_index))
            return
        super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):  # noqa: N802 - Qt API
        if isinstance(editor, QComboBox):
            model.setData(index, str(editor.currentData()), Qt.ItemDataRole.EditRole)
            return
        super().setModelData(editor, model, index)


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return str(value)


def _parse_scalar(text: str, kind: str) -> Any:
    stripped = text.strip()
    if kind == "str":
        return text
    if kind == "bool":
        lowered = stripped.lower()
        if lowered in {"true", "1", "yes", "是"}:
            return True
        if lowered in {"false", "0", "no", "否"}:
            return False
        raise ValueError("布尔值只能填 true 或 false")
    if kind == "int":
        value = int(stripped)
        if str(value) != stripped and stripped not in {f"+{value}", f"-{abs(value)}"}:
            raise ValueError("请输入整数")
        return value
    if kind == "float":
        return float(stripped)
    raise ValueError(f"未知参数类型：{kind}")


class ConfigTreeEditor(QWidget):
    """Edit every scalar in a nested config while preserving Python types."""

    mapping_changed = Signal()
    validation_changed = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mapping: dict[str, Any] = {}
        self._updating = False
        self._invalid_items: set[QTreeWidgetItem] = set()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("筛选参数"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("例如：记录时长、道间距、频散、noise、velocity...")
        self.search.setClearButtonEnabled(True)
        search_row.addWidget(self.search, 1)
        layout.addLayout(search_row)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(("参数（中文）", "当前值", "单位", "它控制什么"))
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setMouseTracking(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.tree.setItemDelegate(_ValueOnlyDelegate(self.tree))
        self.tree.header().resizeSection(0, 245)
        self.tree.header().resizeSection(1, 185)
        self.tree.header().resizeSection(2, 82)
        self.tree.header().setStretchLastSection(True)
        layout.addWidget(self.tree, 1)

        self.details = QTextBrowser()
        self.details.setObjectName("parameterHelpCard")
        self.details.setOpenExternalLinks(False)
        self.details.setMinimumHeight(154)
        self.details.setMaximumHeight(190)
        self.details.setHtml(
            "<b>参数帮助</b><br>把鼠标移到任意参数，或单击一行；这里会持续显示单位、硬约束、推荐起点和调整影响。"
        )
        layout.addWidget(self.details)

        self.status = QLabel("双击“当前值”修改；枚举和开关会显示中文下拉选项。悬停任意单元格可看完整说明。")
        self.status.setObjectName("editorHint")
        layout.addWidget(self.status)

        self.search.textChanged.connect(self._filter)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemEntered.connect(self._show_item_help)
        self.tree.currentItemChanged.connect(
            lambda current, previous=None: self._show_item_help(current, 0)
        )

    def set_mapping(self, mapping: dict[str, Any]) -> None:
        self._mapping = deepcopy(mapping)
        self._updating = True
        self.tree.clear()
        self._invalid_items.clear()
        for key, value in self._mapping.items():
            self._add_value(self.tree.invisibleRootItem(), key, value, (key,))
        self._updating = False
        self.tree.expandToDepth(1)
        first = self._first_scalar_item()
        if first is not None:
            self.tree.setCurrentItem(first)
            self._show_item_help(first, 0)
        self.status.setText("双击“当前值”修改；枚举和开关会显示中文下拉选项。悬停任意单元格可看完整说明。")
        self.validation_changed.emit(True, "")

    def mapping(self) -> dict[str, Any]:
        if self._invalid_items:
            paths = [str(item.data(0, PATH_ROLE)) for item in self._invalid_items]
            raise ValueError("这些参数值无效：" + "，".join(paths))
        return deepcopy(self._mapping)

    def set_path(self, path: tuple[str | int, ...], value: Any) -> None:
        target: Any = self._mapping
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        self.set_mapping(self._mapping)
        self.mapping_changed.emit()

    def replace_table(self, path: tuple[str | int, ...], value: Any) -> None:
        self.set_path(path, deepcopy(value))

    def _display_path(self, path: tuple[str | int, ...]) -> str:
        parts = []
        for part in path:
            if isinstance(part, int):
                parts.append(f"[{part}]")
            else:
                parts.append(str(part))
        return ".".join(parts)

    def _add_value(
        self,
        parent: QTreeWidgetItem,
        label: str,
        value: Any,
        path: tuple[str | int, ...],
    ) -> None:
        display_path = self._display_path(path)
        spec = parameter_spec(display_path)
        if isinstance(value, dict):
            display_label = group_label(display_path, label)
            if path and isinstance(path[-1], int) and "thickness_m" in value:
                thickness = float(value["thickness_m"])
                layer_type = "半空间" if math.isinf(thickness) else f"厚 {thickness:g} m"
                display_label = f"{label}｜{layer_type}"
            item = QTreeWidgetItem(parent, (display_label, "", "", spec.summary))
            item.setFirstColumnSpanned(False)
            item.setData(0, PATH_ROLE, display_path)
            group_tip = (
                f'<div style="width:420px"><b>{display_label}</b><br>'
                f'<code>{display_path}</code><hr>{spec.summary}</div>'
            )
            for column in range(4):
                item.setToolTip(column, group_tip)
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            for key, child in value.items():
                self._add_value(item, key, child, (*path, key))
            return
        if isinstance(value, list):
            display_label = group_label(display_path, label)
            item = QTreeWidgetItem(parent, (display_label, f"{len(value)} 层", "", spec.summary))
            item.setData(0, PATH_ROLE, display_path)
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            group_tip = (
                f'<div style="width:420px"><b>{display_label}</b><br>'
                f'<code>{display_path}</code><hr>{spec.summary}</div>'
            )
            for column in range(4):
                item.setToolTip(column, group_tip)
            for index, child in enumerate(value):
                self._add_value(item, f"第 {index + 1} 层", child, (*path, index))
            return

        item = QTreeWidgetItem(
            parent,
            (spec.label, _format_scalar(value), spec.unit, spec.summary),
        )
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        item.setData(0, PATH_ROLE, display_path)
        item.setData(1, TYPE_ROLE, _type_name(value))
        tooltip = parameter_detail_html(display_path, value, self._mapping, compact=True)
        for column in range(4):
            item.setToolTip(column, tooltip)
            item.setStatusTip(column, f"{spec.label}｜{display_path}")

    def _path_tuple(self, display_path: str) -> tuple[str | int, ...]:
        parts: list[str | int] = []
        for token in display_path.split("."):
            if token.startswith("[") and token.endswith("]"):
                parts.append(int(token[1:-1]))
            else:
                parts.append(token)
        return tuple(parts)

    def _item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating or column != 1:
            return
        path_text = item.data(0, PATH_ROLE)
        if not path_text:
            return
        try:
            value = _parse_scalar(item.text(1), str(item.data(1, TYPE_ROLE)))
            target: Any = self._mapping
            path = self._path_tuple(path_text)
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
            item.setBackground(1, QColor("transparent"))
            self._invalid_items.discard(item)
            self._refresh_help_items()
            spec = parameter_spec(path_text)
            self.status.setText(f"已修改：{spec.label}（{path_text}）；请重新执行输入审核。")
            self.mapping_changed.emit()
        except (TypeError, ValueError) as exc:
            item.setBackground(1, QColor("#ffd6d6"))
            item.setToolTip(1, f"输入无效：{exc}")
            self._invalid_items.add(item)
            self.status.setText(f"输入无效：{path_text}｜{exc}")
        valid = not self._invalid_items
        self.validation_changed.emit(valid, self.status.text())

    def _iter_items(self):
        stack = [
            self.tree.topLevelItem(index)
            for index in reversed(range(self.tree.topLevelItemCount()))
        ]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(index) for index in reversed(range(item.childCount())))

    def item_for_path(self, path: str) -> QTreeWidgetItem | None:
        """Return the visible item for a technical TOML path (useful for help/tests)."""
        return next(
            (item for item in self._iter_items() if str(item.data(0, PATH_ROLE) or "") == path),
            None,
        )

    def _first_scalar_item(self) -> QTreeWidgetItem | None:
        return next((item for item in self._iter_items() if item.data(1, TYPE_ROLE)), None)

    def _value_for_path(self, display_path: str) -> Any:
        target: Any = self._mapping
        for part in self._path_tuple(display_path):
            target = target[part]
        return target

    def _show_item_help(self, item: QTreeWidgetItem | None, column: int = 0) -> None:
        if item is None:
            return
        path = str(item.data(0, PATH_ROLE) or "")
        if not path:
            return
        if item.data(1, TYPE_ROLE):
            try:
                value = self._value_for_path(path)
            except (KeyError, IndexError, TypeError):
                value = item.text(1)
            self.details.setHtml(parameter_detail_html(path, value, self._mapping))
            return
        top = canonical_path(path).split(".", 1)[0]
        section_name, text = SECTION_HELP.get(top, (item.text(0), parameter_spec(path).summary))
        self.details.setHtml(
            f"<div style='width:760px'><b style='font-size:15px'>{section_name}</b>"
            f"<br><code>{path}</code><hr>{text}</div>"
        )

    def _refresh_help_items(self) -> None:
        for item in self._iter_items():
            path = str(item.data(0, PATH_ROLE) or "")
            if not path or not item.data(1, TYPE_ROLE):
                continue
            try:
                value = self._value_for_path(path)
            except (KeyError, IndexError, TypeError):
                value = item.text(1)
            tooltip = parameter_detail_html(path, value, self._mapping, compact=True)
            for column in range(4):
                item.setToolTip(column, tooltip)
        self._show_item_help(self.tree.currentItem(), 0)

    def _filter(self, text: str) -> None:
        needle = text.strip().lower()

        def visit(item: QTreeWidgetItem) -> bool:
            child_match = any(visit(item.child(i)) for i in range(item.childCount()))
            path = str(item.data(0, PATH_ROLE) or "").lower()
            self_match = needle in path or any(
                needle in item.text(column).lower() for column in range(4)
            )
            visible = not needle or self_match or child_match
            item.setHidden(not visible)
            if needle and visible:
                item.setExpanded(True)
            return visible

        root = self.tree.invisibleRootItem()
        for index in range(root.childCount()):
            visit(root.child(index))
