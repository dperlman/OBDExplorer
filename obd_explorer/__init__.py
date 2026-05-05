"""OBD graph explorer library: shard-aware data, viewport math, Qt graphics helpers."""

from obd_explorer.constants import DEFAULT_GRAPH_P_STEPS, SCALED, SORTED
from obd_explorer.grid import BinomialGrid, build_binomial_grid_from_shards
from obd_explorer.model import TIE_COLOR_AXIS_CHOICES, TieColorSpec
from obd_explorer.numeric import (
    expected_rank,
    interpolate_y_at_p,
    subtract_endpoint_chord,
)

__all__ = [
    "DEFAULT_GRAPH_P_STEPS",
    "SCALED",
    "SORTED",
    "BinomialGrid",
    "build_binomial_grid_from_shards",
    "TIE_COLOR_AXIS_CHOICES",
    "TieColorSpec",
    "expected_rank",
    "interpolate_y_at_p",
    "subtract_endpoint_chord",
]
