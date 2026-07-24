"""HOM 専用 FEM ソルバ（DOF: Nédélec エッジ + 節点スカラー）."""

from .field_recon import (
    HOMFieldReconstructor,
    calculate_E_theta_from_rE_theta,
    split_hom_dofs,
)
from .post_process import (
    HOMModeParameters,
    HOMParameters,
    calc_p_flow_hom,
    compute_hom_parameters,
)
from .solver import (
    HOMModeSet,
    HOMResult,
    solve_hom,
    solve_hom_standing,
    solve_hom_traveling,
)

__all__ = [
    "HOMModeSet",
    "HOMResult",
    "solve_hom",
    "solve_hom_standing",
    "solve_hom_traveling",
    "HOMFieldReconstructor",
    "calculate_E_theta_from_rE_theta",
    "split_hom_dofs",
    "HOMModeParameters",
    "HOMParameters",
    "calc_p_flow_hom",
    "compute_hom_parameters",
]
