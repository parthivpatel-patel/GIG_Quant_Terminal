"""Validated European option research and volatility-surface diagnostics."""
from .analytics import (
    black_scholes, implied_volatility, OptionQuote, SurfacePoint, SurfaceDiagnostics,
    diagnose_call_slice, validate_surface, interpolate_total_variance, OptionPosition, portfolio_exposure, scenario_grid,
)

__all__ = [
    "black_scholes", "implied_volatility", "OptionQuote", "SurfacePoint", "SurfaceDiagnostics",
    "diagnose_call_slice", "validate_surface", "interpolate_total_variance", "OptionPosition", "portfolio_exposure", "scenario_grid",
]
