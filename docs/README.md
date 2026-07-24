# AxiCavity-FEM ドキュメント索引

| ドキュメント | 対象 | 内容 |
|---|---|---|
| [../README.md](../README.md) / [../README.ja.md](../README.ja.md) | 全員 | 概要・インストール・コマンド早見 |
| [../USER_MANUAL.md](../USER_MANUAL.md) | 利用者 | GUI の操作手順と CLI の全オプション |
| [../PHYSICS_AND_CONVENTIONS.md](../PHYSICS_AND_CONVENTIONS.md) | 利用者（上級）／開発者 | 物理規約（時間規約・規格化・P_flow・周期境界の符号） |
| [BC_NAMING.md](BC_NAMING.md) | 利用者／開発者 | 境界条件 PEC / E-short / M-short / None の物理-数学対応 |
| [HDF5_SCHEMA.md](HDF5_SCHEMA.md) | 開発者 | 出力 HDF5 のスキーマ |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | **開発者（まずこれ）** | コードマップ・規約と落とし穴・検証戦略・拡張レシピ |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 開発者 | レイヤ設計と各層の責務 |
| [../CHANGELOG.md](../CHANGELOG.md) | 全員 | 変更履歴 |

## 開発を始めるとき

```bash
pip install -e ".[dev]"     # GUI も触るなら ".[dev,gui]"
pytest -q                   # 373 passed（wxPython 無しの環境では GUI テストが skip）
```

1. [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) を読む（特に §1 レイヤ構成、§4 規約と落とし穴）。
2. editable インストールの実体パスを確認する
   （`python -c "import axicavity_fem, inspect; print(inspect.getfile(axicavity_fem))"` が
   作業中のフォルダを指すこと）。
3. 変更後は必ず全テストを回す。数値に関わる変更なら、解析解や既存の期待値と照合する
   テストを追加する（§5）。
4. GUI のレイアウトは wxGlade が `../main_frame_ui.wxg` / `../result_viewer_ui.wxg` から
   生成しています。`*_ui.py` を直接編集せず `.wxg` を編集して再生成し、動作は手書きの
   サブクラス（`main_frame.py`, `result_viewer.py`）側に書いてください。
5. 見た目に関わる変更（GUI・PNG・HTML レポート）は自動テストでは検証できません。
   人の目での確認が必要です。
