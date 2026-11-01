# Versioned market-session estimates

`gig.data.calendars.ExchangeCalendar` adapts the optional open-source
`exchange_calendars` package. No provider, brokerage account, network request,
or instrument lookup occurs during calendar computations. The caller supplies
an explicit calendar identity; tickers never determine an exchange implicitly.

The validated dependency is `exchange_calendars` 4.13.2. Its required support
packages were installed without replacing the existing NumPy/Pandas/SciPy
environment. Pip installation reports and package versions are retained under
the execution workspace evidence directory. An unavailable dependency raises
`ImportError`; an unknown identity raises `ValueError`. No weekday-only fallback
is substituted.

## APIs and bounds

```python
from gig.data.calendars import ExchangeCalendar, calendar_analytics

calendar = ExchangeCalendar("XNYS")
schedule = calendar.schedule("2026-11-23", "2026-11-30")
status = calendar.status("2026-11-27T17:00:00Z")
freshness = calendar.freshness(
    "2026-11-27T18:00:00Z", "2026-11-29T20:00:00Z",
    max_missed_sessions=0, grace_seconds=0,
)
validated = calendar.validate_daily_closes(
    ["2026-11-25T21:00:00Z", "2026-11-27T18:00:00Z"],
    require_contiguous=True,
)
```

The default coverage is 2020-01-01 through 2030-01-01, inclusive. Custom
coverage must span 32–3,660 calendar days. Schedule responses cover at most
366 calendar days; daily-close validation accepts 1–2,000 observations.
Date inputs use exact `YYYY-MM-DD` labels. Aware instant inputs retain up to
nanosecond precision and normalize to UTC; naive, out-of-coverage, malformed,
and subnanosecond inputs are rejected. Date coverage is interpreted using the
calendar's own timezone, including calendars whose sessions cross UTC dates.

Each session gives its date label, UTC opening and closing instants, optional
break endpoints, early-close flag, elapsed session span, and trading seconds
excluding breaks. Calendar schedule responses with no sessions explicitly
report zero scheduled sessions, such as a weekend query. This is an available
calendar result and says nothing about available observations.

`status()` evaluates the exact supplied instant: opening is inclusive, closing
is exclusive, and break start/end use the same half-open convention. It returns
scheduled `OPEN`, `BREAK`, or `CLOSED` and the most recent completed close.
An early close, holiday, or DST transition comes from the identified calendar
rules. It is never a broker-observed market-open flag or an order approval.

`freshness()` requires the observed daily bar to match an actual scheduled close
and not lie after the cutoff. Expected closes become due when
`close + grace_seconds <= as_of`; grace is 0–3,600 seconds. Missed-session
tolerance is an explicit integer from zero to 30. It reports `AVAILABLE` or
`STALE` according to those supplied assumptions, and `UNKNOWN` if no expected
close exists inside coverage. Weekends do not invent missing bars. The first
64 missing session labels are displayed together with total count, completeness,
and a hash of all labels, so bounded display does not erase evidence identity.

`validate_daily_closes()` applies only to bars whose declared timestamp semantics
are session closing instants. It does not reinterpret midnight or session-open
vendor bars. Duplicate UTC-equivalent instants, reversed order, holidays,
intraday timestamps, or wrong early-close times are rejected. Missing scheduled
sessions are reported explicitly and may be made a rejection by setting
`require_contiguous=True`.

## Analytical HTTP adapter

`calendar_analytics(payload)` returns a bounded schedule result and optional
status, freshness, and daily-close validation. Required fields are `calendar`,
`start`, `end`. Optional fields are `as_of`, `last_bar_close`,
`max_missed_sessions`, `grace_seconds`, `daily_closes`, and `require_contiguous`.
Unused policy fields and unknown fields are rejected. Freshness requires an
explicit cutoff. The service owner can wrap this dictionary in the existing
typed `MODEL_ESTIMATE` envelope; embedded classification is `CALENDAR_ESTIMATE`.

Provenance includes requested and canonical calendar identity, package version,
timezone, explicit coverage, and a SHA-256 over all detached scheduled sessions.
Returned dictionaries cannot mutate the calendar snapshot or the package cache.
All outputs retain trading approval false.

Calendar rules use the installed package's current knowledge. The adapter does
not reconstruct historical announcement-time calendar revisions, authenticate
provider feeds, observe unscheduled halts, verify licenses for market data, or
establish instrument exchange membership. Those are separate admission and
operational requirements. Institutional, economic, ARM64, and broker validation
are not claimed.

## Verification

```sh
PYTHONPATH=/workspace/work/dependencies:/workspace/work/tooling:src \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_market_calendars.py
```

Observed tests exercise actual 2026 NYSE Thanksgiving and Juneteenth holidays,
Thanksgiving early close, both DST transitions, exact nanosecond boundaries,
weekend freshness, grace deadlines, equivalent timezones, duplicate/gap
admission, dependency absence, identity/mutation isolation, and Hong Kong's
midday break. Ruff and scoped mypy checks pass. These numerical/calendar-rule
checks do not certify agreement with every exchange's current operating notice.
