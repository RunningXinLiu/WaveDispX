"""Public contact information; opening the dialog makes no network requests."""
from html import escape
import sys

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextBrowser, QVBoxLayout

from wavedispx import __version__

AUTHOR_WEBSITE = "https://runningxinliu.github.io/"
AUTHOR_EMAIL = "xinliu_geo@outlook.com"
PROJECT_WEBSITE = "https://github.com/RunningXinLiu/WaveDispX"


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("studioAboutDialog")
        self.setWindowTitle("关于 WaveDispX")
        self.resize(660, 500)
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser(self)
        self.browser.setObjectName("studioAboutContent")
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(f"""
            <h2>WaveDispX Studio</h2>
            <p>波场正演与频散实验工作台</p>
            <p>Python 包版本：{escape(__version__)} · Studio {'桌面试用版' if getattr(sys, 'frozen', False) else '源码版'}</p>
            <h3>开发与维护 · 联系方式</h3>
            <p>个人主页：<a href="{AUTHOR_WEBSITE}">{AUTHOR_WEBSITE}</a></p>
            <p>联系邮箱：<a href="mailto:{AUTHOR_EMAIL}">{AUTHOR_EMAIL}</a></p>
            <p>项目源码与说明：<a href="{PROJECT_WEBSITE}">GitHub · WaveDispX</a></p>
            <p>仅点击链接时打开系统浏览器或默认邮件程序；不会自动发送邮件。</p>
            <h3>来源与致谢</h3>
            <p>本项目沿用原 PassiveFW / MATLAB 程序的工作；原作者署名和参考文献保留在仓库 Readme.txt。
            感谢 Pan Lei 的 <a href="https://github.com/pan3rock/QEDispInv">QEDispInv 仓库</a>，为本软件提供理论频散计算后端。
            界面和运行环境使用 PySide6 / Qt、NumPy、SciPy、Matplotlib 等组件。</p>
            <p>各组件保留原有版权及许可，开发维护信息不替代第三方署名。
            <a href="{PROJECT_WEBSITE}/blob/main/packaging/classroom/THIRD_PARTY_zh.md">第三方组件与现有许可记录</a></p>
            <p>适用范围与验证限制请查看操作说明；软件版本号不代表所有模型已通过科学验收。</p>
        """)
        layout.addWidget(self.browser)
        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
