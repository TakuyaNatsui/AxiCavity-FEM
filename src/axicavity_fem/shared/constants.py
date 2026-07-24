"""物理定数: c0, eps0, mu0 等."""

import math

C0 = 299_792_458.0          # 真空中の光速 [m/s]
MU0 = 4.0e-7 * math.pi      # 真空透磁率 [H/m]
EPS0 = 1.0 / (MU0 * C0 ** 2)  # 真空誘電率 [F/m]


def k2_from_freq_ghz(freq_ghz: float) -> float:
    """周波数 [GHz] を固有値 k^2 = (2πf/c)^2 [1/m^2] に変換する.

    固有値解析のシフト値 sigma を周波数から与えるときに使う (ver2.3)。
    """
    return (2 * math.pi * float(freq_ghz) * 1e9 / C0) ** 2
