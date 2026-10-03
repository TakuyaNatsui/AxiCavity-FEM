"""パッケージインポートのスモークテスト."""

import axicavity_fem


def test_version():
    assert axicavity_fem.__version__ == "3.0.0"


def test_subpackage_imports():
    import axicavity_fem.shared
    import axicavity_fem.physics_core
    import axicavity_fem.fem_tm0
    import axicavity_fem.fem_hom
    import axicavity_fem.reports
    import axicavity_fem.cli


def test_cli_entrypoint():
    from axicavity_fem.cli.main import cli_main, build_parser
    parser = build_parser()
    assert parser.prog == "axicavity-fem"


def test_constants():
    from axicavity_fem.shared.constants import C0, MU0, EPS0
    assert abs(C0 - 299_792_458.0) < 1e-6
    # MU0 * EPS0 * C0^2 = 1
    assert abs(MU0 * EPS0 * C0 ** 2 - 1.0) < 1e-12
