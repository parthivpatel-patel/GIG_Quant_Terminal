"""Validated European option research and volatility-surface diagnostics."""
from .analytics import (
    OptionPosition,
    OptionQuote,
    SurfaceDiagnostics,
    SurfacePoint,
    black_scholes,
    diagnose_call_slice,
    implied_volatility,
    interpolate_total_variance,
    portfolio_exposure,
    scenario_grid,
    validate_surface,
)

__all__ = [
    "black_scholes", "implied_volatility", "OptionQuote", "SurfacePoint", "SurfaceDiagnostics",
    "diagnose_call_slice", "validate_surface", "interpolate_total_variance", "OptionPosition", "portfolio_exposure", "scenario_grid",
]
