"""wxPython GUI エントリポイント (axicavity-fem-gui コマンド).

ver1 ``app.py`` を ver2 パッケージ構成に移植したもの。GUI は内部で統一 CLI
``axicavity-fem`` をサブプロセス起動して solve / post を実行する。
"""

import sys


def main() -> int:
    import wx
    from .main_frame import MyFrame

    class MyApp(wx.App):
        def OnInit(self):
            self.frame = MyFrame(None, wx.ID_ANY, "")
            self.SetTopWindow(self.frame)
            self.frame.Show()
            return True

    app = MyApp(0)
    app.MainLoop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
