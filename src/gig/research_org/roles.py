"""Role contracts for the research organization.

A role is a permitted workflow with a data boundary. Nothing here calls a
language model, and nothing here is allowed to trade, spend, or change code.
"""

from __future__ import annotations

ROLES: tuple[dict[str, object], ...] = (
    {"id": "orchestrator", "name": "Chief Research Orchestrator",
     "does": "Chooses the next approved hypothesis and records why it ran.",
     "tools": ["research ledger", "hypothesis catalog"], "budget": "one hypothesis per cycle"},
    {"id": "quant", "name": "Quantitative Researcher",
     "does": "Runs the chosen experiment on its declared dataset.",
     "tools": ["causal signal", "performance metrics"], "budget": "one authored series per cycle"},
    {"id": "ml", "name": "Machine Learning Researcher",
     "does": "No model is trained or promoted in this version.",
     "tools": [], "budget": "none"},
    {"id": "microstructure", "name": "Market Microstructure Researcher",
     "does": "Not invoked. Existing microstructure research stays in its own workspace.",
     "tools": [], "budget": "none"},
    {"id": "options", "name": "Options and Volatility Researcher",
     "does": "Not invoked. The 3D volatility lab is unchanged.",
     "tools": [], "budget": "none"},
    {"id": "macro", "name": "Macro and Global Intelligence Researcher",
     "does": "Not invoked. Global Intelligence keeps its own adapters.",
     "tools": [], "budget": "none"},
    {"id": "risk", "name": "Portfolio and Risk Researcher",
     "does": "Reads the experiment's drawdown and turnover. Does not change risk limits.",
     "tools": ["performance metrics"], "budget": "read only"},
    {"id": "skeptic", "name": "Independent Research Skeptic",
     "does": "Rejects lookahead, compares a shuffled placebo, and refuses a positive claim on synthetic data.",
     "tools": ["lookahead probe", "placebo shuffle", "deflated Sharpe"], "budget": "16 placebo draws"},
    {"id": "developer", "name": "Quantitative Developer",
     "does": "Owns this code. The organization cannot rewrite it.",
     "tools": [], "budget": "no code changes from a cycle"},
)

MAY = (
    "Read the authored series declared by the hypothesis.",
    "Run one bounded experiment.",
    "Record a positive, negative, or inconclusive finding.",
    "Recommend a follow-up without starting it.",
)
MUST_NOT = (
    "Rewrite code.",
    "Change risk controls.",
    "Promote a model.",
    "Deploy a service.",
    "Submit a brokerage order.",
    "Spend money.",
    "Read credentials.",
    "Disable the audit log.",
)

# Registered in code, but this organization does not execute them.
REGISTERED_NOT_RUN = (
    {"id": "equity_ls", "name": "Equity long/short", "module": "gig.strategies.equity_ls",
     "status": "REGISTERED_NOT_EXECUTED",
     "note": "Existing strategy module. A cycle does not run it and reports no performance for it."},
    {"id": "futures_tsmom", "name": "Futures time-series momentum", "module": "gig.strategies.futures_tsmom",
     "status": "REGISTERED_NOT_EXECUTED",
     "note": "Existing strategy module. Not executed by the organization."},
    {"id": "options_vrp", "name": "Options variance-risk premium", "module": "gig.strategies.options_vrp",
     "status": "REGISTERED_NOT_EXECUTED",
     "note": "Existing strategy module. Not executed by the organization."},
)
