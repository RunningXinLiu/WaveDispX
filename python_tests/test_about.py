"""About/contact UI does not require a simulation or launch external apps."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QDialogButtonBox
from wavedispx.desktop.about import AboutDialog, AUTHOR_EMAIL, AUTHOR_WEBSITE
from wavedispx import __version__


def test_about_contact_version_and_close():
    app = QApplication.instance() or QApplication([])
    dialog = AboutDialog()
    dialog.show()
    app.processEvents()
    text = dialog.browser.toPlainText()
    html = dialog.browser.toHtml()
    assert AUTHOR_WEBSITE == "https://runningxinliu.github.io/"
    assert AUTHOR_EMAIL == "xinliu_geo@outlook.com"
    assert f'href="{AUTHOR_WEBSITE}"' in html
    assert f'href="mailto:{AUTHOR_EMAIL}"' in html
    assert __version__ in text and "源码版" in text
    assert "原作者署名" in text and "Pan Lei" in text and "QEDispInv" in text
    assert 'href="https://github.com/pan3rock/QEDispInv"' in html
    assert dialog.browser.openExternalLinks()
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Close).click()
    assert not dialog.isVisible()
