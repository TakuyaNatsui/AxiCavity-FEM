# AxiCavity-FEM documentation / ドキュメント索引

| Document | For | Contents |
|---|---|---|
| [../USER_MANUAL.md](../USER_MANUAL.md) | users | GUI tutorials and reference, every CLI option, the Python batch API, Windows executable（日本語） |
| [../README.md](../README.md) / [../README.ja.md](../README.ja.md) | everyone | overview, installation, quick start |
| [../PHYSICS_AND_CONVENTIONS.md](../PHYSICS_AND_CONVENTIONS.md) | users / developers | time convention, normalization, power flow, periodic-BC sign |
| [BC_NAMING.md](BC_NAMING.md) | users / developers | what `PEC` / `E-short` / `M-short` / `None` mean physically and numerically |
| [HDF5_SCHEMA.md](HDF5_SCHEMA.md) | developers | layout of the output HDF5 files |
| [ARCHITECTURE.md](ARCHITECTURE.md) | developers | layers and their responsibilities |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | developers | file-by-file code map, conventions and pitfalls, testing, extension recipes, the Windows build |
| [../packaging/README.md](../packaging/README.md) | maintainers | building the Windows executable |
| [../CHANGELOG.md](../CHANGELOG.md) | everyone | release notes |

開発を始めるときは [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) の §1（レイヤ構成）と §4（規約と落とし穴）から読み、
`pip install -e ".[gui,viz3d,dev]"` のあと `pytest` が通ることを確認してください（参照結果が要るテストは、
ファイルが無ければ skip されます）。
