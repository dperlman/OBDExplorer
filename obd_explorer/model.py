"""Pure-Python data models (no Qt)."""

from __future__ import annotations

from dataclasses import dataclass

# Each viewport half uses one of these axes for tie/fill colormap scalars.
TIE_COLOR_AXIS_CHOICES: tuple[str, ...] = ("black", "i", "j", "l", "r", "d", "e")
_TIE_COLOR_AXIS_SET: frozenset[str] = frozenset(TIE_COLOR_AXIS_CHOICES)


@dataclass(frozen=True)
class TieColorSpec:
    """Tie / fill colormap: independent ``left`` and ``right`` axis choices.

    ``i`` / ``j``: rank indices along the band. ``l`` / ``r``: ``slope_left`` / ``slope_right``
    from tie metadata. ``d`` / ``e``: ``slope_right - slope_left`` / ``slope_left - slope_right``
    (ranges computed per band from the same shard slopes).

    In a **full** p window, :meth:`key_at_p` uses ``left`` when ``p < 0.5`` and ``right`` when
    ``p ≥ 0.5``. In **left** ``[0, 0.5]`` only ``left`` applies; in **right** ``[0.5, 1]`` only
    ``right`` applies. ``black`` yields solid black for that half (no colormap).
    """

    left: str = "i"
    right: str = "j"

    @staticmethod
    def _norm_axis(s: str, *, field: str) -> str:
        v = str(s).strip().lower()
        if v not in _TIE_COLOR_AXIS_SET:
            raise ValueError(
                f'{field} must be one of {", ".join(TIE_COLOR_AXIS_CHOICES)}; got {s!r}'
            )
        return v

    @staticmethod
    def parse_lr(left: str, right: str) -> TieColorSpec:
        return TieColorSpec(
            left=TieColorSpec._norm_axis(left, field="tie_color_left"),
            right=TieColorSpec._norm_axis(right, field="tie_color_right"),
        )

    def key_at_p(self, p: float, vp_range_norm: str) -> str:
        w = vp_range_norm.strip().lower()
        if w == "full":
            return self.left if float(p) < 0.5 else self.right
        if w == "left":
            return self.left
        if w == "right":
            return self.right
        raise ValueError(
            'vp_range_norm must be "full", "left", or "right" '
            f"(case insensitive); got {vp_range_norm!r}"
        )

    def axis_at_p(self, p: float, vp_range_norm: str) -> str:
        """Alias for :meth:`key_at_p` (backward compatible name)."""
        return self.key_at_p(p, vp_range_norm)
