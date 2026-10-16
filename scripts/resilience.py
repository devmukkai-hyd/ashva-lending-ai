"""
resilience.py - Milestone 7: hardening the lending data calls (the "external" credit bureau first).

Every call the underwriting committee makes for bureau / bank / sector / policy data goes through
resilient_call() (from underwriting_committee.LendingData). For each call:

    circuit breaker ── open? ──────────────────────────────────────────────┐ (no live call at all)
          │ closed / half-open probe                                       │
          ▼                                                                ▼
    attempt 1..N  (timeout per attempt, then exponential backoff + jitter,   fallback sources, in order
                   capped by a total time budget)                            (bureau: cached last good
          │ each attempt: [fault injection] -> live call -> validate          pull, then the nightly
          │                                                                  bureau snapshot)
          ├─ ok ──────────────► result + provenance {"source": "live", ...}       │
          └─ all failed ───────────────────────────────────────────────────────────┘
                                                      result + provenance {"source": "bureau_cache", "stale": True, ...}
                                                      or AllSourcesFailed -> the memo says "unavailable"

Failure classes (classify()):
    transient   ConnectionError / TimeoutError / OSError   -> retried, counts against the breaker
    bad_data    the result failed validation (malformed)    -> retried once at most, counts against the breaker
    permanent   bad input, "no report on file", tool error  -> not retried, not counted, raised straight away
A permanent error is the caller's problem, not the dependency's: it must never trip the breaker.

Circuit breakers are per tool and live for the whole process, so they remember failures across
applications (one LendingData is created per application; a breaker inside it would never trip).

Fault injection (seeded, repeatable) simulates a flaky bureau for tests and the capstone demo:
    LENDING_FAULTS="get_credit_bureau:fail=0.2,timeout=0.1,malformed=0.1,seed=42"
    LENDING_FAULTS="get_credit_bureau:outage=20-60"          # requests 20..59 all fail (sustained outage)
or set_faults("...") / --faults on the CLIs. Nothing is injected unless you ask for it.

Every attempt, breaker refusal and fallback is a span (tracing.py), nested under the "mcp:<tool>" span.

CLI
    python resilience.py bench [--n 200] [--faults SPEC] [--outage A-B]   # B1-B4 style comparison
    python resilience.py report                                            # from traces.jsonl + runs
    python resilience.py status                                            # breaker settings per tool
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from pathlib import Path

from tracing import span

TIME_SCALE = float(os.getenv("RESILIENCE_TIME_SCALE", "1.0"))   # < 1 shortens every backoff (tests, bench)
CACHE_MAX_AGE_DAYS = int(os.getenv("BUREAU_CACHE_MAX_AGE_DAYS", "30"))


# ---------------------------------------------------------------------------
# Errors and classification
# ---------------------------------------------------------------------------
class BadData(Exception):
    """The dependency answered, but the answer failed validation (malformed / incomplete)."""


class CircuitOpenError(Exception):
    """The breaker is open: the call was refused without touching the dependency."""


class AllSourcesFailed(Exception):
    """Live calls and every fallback failed."""


TRANSIENT = (ConnectionError, TimeoutError, asyncio.TimeoutError, OSError)


def classify(exc: BaseException) -> str:
    if isinstance(exc, BadData):
        return "bad_data"
    if isinstance(exc, TRANSIENT):
        return "transient"
    text = str(exc).lower()
    # MCP reports a server-side exception as an error RESULT; look at the message for transport trouble.
    if any(w in text for w in ("timed out", "timeout", "connection", "unavailable", "temporarily", "503", "502")):
        return "transient"
    return "permanent"


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolPolicy:
    max_attempts: int = 3
    base_delay: float = 0.5          # seconds; backoff = random(0, min(max_delay, base * 2^(n-1)))  ("full jitter")
    max_delay: float = 4.0
    attempt_timeout: float = 10.0    # per attempt
    total_budget: float = 20.0       # never spend longer than this on live attempts
    retry_bad_data: int = 1          # extra attempts allowed after a validation failure
    validate: bool = True
    fallbacks: tuple = ()
    breaker_threshold: int = 3       # consecutive dependency failures that open the circuit
    breaker_reset: float = 30.0      # seconds open before a half-open probe
    use_breaker: bool = True


POLICIES = {
    "get_credit_bureau": ToolPolicy(max_attempts=4, attempt_timeout=10, total_budget=25,
                                    fallbacks=("bureau_cache", "bureau_snapshot")),
    "get_bank_statement": ToolPolicy(max_attempts=3),
    "get_sector_policy": ToolPolicy(max_attempts=3),
    "search_credit_policy": ToolPolicy(max_attempts=2, attempt_timeout=90, total_budget=120, retry_bad_data=0),
}
DEFAULT_POLICY = ToolPolicy()


def policy_for(tool: str) -> ToolPolicy:
    return POLICIES.get(tool, DEFAULT_POLICY)


# ---------------------------------------------------------------------------
# Validation: a result that doesn't look right is a failure, not data
# ---------------------------------------------------------------------------
def _int_in(v, lo, hi) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


def validate_bureau(r) -> list[str]:
    if not isinstance(r, dict):
        return ["not a JSON object"]
    p = []
    if not _int_in(r.get("bureau_score"), 300, 900):
        p.append(f"bureau_score {r.get('bureau_score')!r} not an integer in 300-900")
    for k in ("enquiries_last_6m", "active_trade_lines", "dpd_30_last_12m"):
        if not _int_in(r.get(k), 0, 1000):
            p.append(f"{k} {r.get(k)!r} missing or invalid")
    if not isinstance(r.get("dpd_90_ever"), bool):
        p.append("dpd_90_ever missing")
    if not isinstance(r.get("written_off_amount_inr"), (int, float)) or r.get("written_off_amount_inr") < 0:
        p.append("written_off_amount_inr missing or negative")
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(r.get("report_pulled_on") or "")):
        p.append("report_pulled_on missing")
    return p


def validate_bank(r) -> list[str]:
    if not isinstance(r, dict):
        return ["not a JSON object"]
    p = []
    if not _int_in(r.get("months_covered"), 0, 600):
        p.append("months_covered missing")
    if not isinstance(r.get("red_flags"), list):
        p.append("red_flags missing")
    return p


def validate_sector(r) -> list[str]:
    if not isinstance(r, dict):
        return ["not a JSON object"]
    return [f"{k} missing" for k in ("sector", "dscr_floor", "max_tenor_months") if r.get(k) is None]


def validate_policy(r) -> list[str]:
    if not isinstance(r, dict) or not isinstance(r.get("clauses"), list):
        return ["clauses missing"]
    return [f"clause {i} has no text" for i, c in enumerate(r["clauses"]) if not (c or {}).get("text")]


VALIDATORS = {"get_credit_bureau": validate_bureau, "get_bank_statement": validate_bank,
              "get_sector_policy": validate_sector, "search_credit_policy": validate_policy}


def validate(tool: str, result) -> list[str]:
    fn = VALIDATORS.get(tool)
    return fn(result) if fn else []


# ---------------------------------------------------------------------------
# Circuit breaker (one per tool, shared by every application in the process)
# ---------------------------------------------------------------------------
class CircuitBreaker:
    """closed -> (threshold consecutive FAILED REQUESTS) -> open -> (reset timeout) -> half_open -> one probe:
    success closes it, failure re-opens it. Only dependency failures are recorded here.

    A request counts as failed only once its retries are exhausted. Counting every attempt would trip the
    breaker on ordinary flakiness (three unlucky attempts inside one request) - the breaker is for outages;
    retries are for flakiness."""

    def __init__(self, name: str, threshold: int = 3, reset_timeout: float = 30.0, clock=time.monotonic):
        self.name, self.threshold, self.reset_timeout, self.clock = name, threshold, reset_timeout, clock
        self.state, self.failures, self.opened_at, self.opened_count = "closed", 0, None, 0
        self._probe_in_flight = False
        self._lock = threading.Lock()

    def before_call(self) -> str:
        """Raises CircuitOpenError, or returns the state the call goes out in ("closed" / "half_open")."""
        with self._lock:
            if self.state == "open":
                if self.clock() - self.opened_at < self.reset_timeout:
                    raise CircuitOpenError(f"{self.name}: circuit open "
                                           f"({self.reset_timeout - (self.clock() - self.opened_at):.0f}s left)")
                self.state = "half_open"
            if self.state == "half_open":
                if self._probe_in_flight:            # only ONE trial call while half-open
                    raise CircuitOpenError(f"{self.name}: circuit half-open, probe already in flight")
                self._probe_in_flight = True
            return self.state

    def record_success(self) -> None:
        with self._lock:
            self.state, self.failures, self._probe_in_flight = "closed", 0, False

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.state == "half_open" or self.failures >= self.threshold:
                if self.state != "open":
                    self.opened_count += 1
                self.state, self.opened_at = "open", self.clock()
            self._probe_in_flight = False

    def release(self) -> None:
        """A call that ended with a permanent (caller-side) error: neither success nor failure."""
        with self._lock:
            self._probe_in_flight = False

    def snapshot(self) -> dict:
        return {"state": self.state, "consecutive_failures": self.failures, "times_opened": self.opened_count}


BREAKERS: dict[str, CircuitBreaker] = {}
_BREAKERS_LOCK = threading.Lock()


def get_breaker(tool: str, policy: ToolPolicy | None = None) -> CircuitBreaker:
    policy = policy or policy_for(tool)
    with _BREAKERS_LOCK:
        if tool not in BREAKERS:
            BREAKERS[tool] = CircuitBreaker(tool, policy.breaker_threshold, policy.breaker_reset)
        return BREAKERS[tool]


def reset_breakers() -> None:
    with _BREAKERS_LOCK:
        BREAKERS.clear()


# ---------------------------------------------------------------------------
# Fault injection (opt-in, seeded, repeatable)
# ---------------------------------------------------------------------------
@dataclass
class FaultConfig:
    seed: int = 42
    fail: float = 0.0          # ConnectionError
    timeout: float = 0.0       # TimeoutError (raised after `delay` seconds)
    malformed: float = 0.0     # the call "succeeds" but returns a corrupted result
    delay: float = 0.05
    outage: tuple | None = None   # (start, end): requests start..end-1 fail completely (a sustained outage)

    def __post_init__(self):
        if self.fail + self.timeout + self.malformed > 1.0:
            raise ValueError("fault rates must not exceed 100%")


class FaultInjector:
    def __init__(self, tool: str, cfg: FaultConfig):
        self.tool, self.cfg = tool, cfg
        self.rng = random.Random(cfg.seed)
        self.calls = 0            # calls that reached the (simulated) dependency
        self.requests = 0         # logical requests (one per resilient_call, however many attempts)
        self._lock = threading.Lock()

    def begin_request(self) -> None:
        with self._lock:
            self.requests += 1

    def roll(self) -> str:
        with self._lock:
            self.calls += 1
            req = self.requests - 1
            r = self.rng.random()
        c = self.cfg
        if c.outage and c.outage[0] <= req < c.outage[1]:
            return "fail"
        if r < c.fail:
            return "fail"
        if r < c.fail + c.timeout:
            return "timeout"
        if r < c.fail + c.timeout + c.malformed:
            return "malformed"
        return "ok"


FAULTS: dict[str, FaultInjector] = {}


def parse_faults(spec: str) -> dict[str, FaultConfig]:
    """'get_credit_bureau:fail=0.2,timeout=0.1,malformed=0.1,seed=42;get_bank_statement:fail=0.1'"""
    out = {}
    for part in filter(None, (p.strip() for p in (spec or "").split(";"))):
        tool, _, opts = part.partition(":")
        kw = {}
        for kv in filter(None, (o.strip() for o in opts.split(","))):
            k, _, v = kv.partition("=")
            k = k.strip()
            if k == "outage":
                a, _, b = v.partition("-")
                kw["outage"] = (int(a), int(b))
            elif k == "seed":
                kw["seed"] = int(v)
            elif k in ("fail", "timeout", "malformed", "delay"):
                kw[k] = float(v)
            else:
                raise ValueError(f"unknown fault option {k!r} in {part!r}")
        out[tool.strip()] = FaultConfig(**kw)
    return out


def set_faults(spec: str | None) -> dict[str, FaultConfig]:
    """Replace the active fault injection (None/"" turns it off). Returns the parsed configs."""
    FAULTS.clear()
    cfgs = parse_faults(spec or "")
    for tool, cfg in cfgs.items():
        FAULTS[tool] = FaultInjector(tool, cfg)
    return cfgs


set_faults(os.getenv("LENDING_FAULTS", ""))


def _corrupt(tool: str, result):
    """A plausible malformed response: a dropped / null field, the kind validation must catch."""
    if not isinstance(result, dict):
        return {"error": "unexpected payload"}
    bad = dict(result)
    key = {"get_credit_bureau": "bureau_score", "get_bank_statement": "months_covered",
           "get_sector_policy": "dscr_floor", "search_credit_policy": "clauses"}.get(tool)
    if key:
        bad[key] = None
    return bad


# ---------------------------------------------------------------------------
# Fallback data sources (bureau)
# ---------------------------------------------------------------------------
_CACHE_PATH: Path | None = None
_CACHE_LOCK = threading.Lock()


def configure(data_dir: Path) -> None:
    """Where the bureau cache lives (next to the other outputs)."""
    global _CACHE_PATH
    _CACHE_PATH = Path(data_dir) / "bureau_cache.json"


def _cache_load() -> dict:
    if _CACHE_PATH and _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def cache_bureau(applicant_id: str, report: dict) -> None:
    """Remember the last GOOD live bureau pull per applicant (written after every validated live result)."""
    if not _CACHE_PATH or not applicant_id:
        return
    with _CACHE_LOCK:
        data = _cache_load()
        data[applicant_id] = {"report": report, "cached_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        tmp = _CACHE_PATH.with_suffix(".tmp")
        try:
            _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
            tmp.replace(_CACHE_PATH)
        except OSError:
            pass


def _fallback_bureau_cache(args: dict):
    entry = _cache_load().get((args.get("applicant_id") or "").strip())
    if not entry:
        return None, {"miss": "no cached pull for this applicant"}
    cached_at = entry.get("cached_at", "")
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(cached_at)).days
    except ValueError:
        age = None
    if age is not None and age > CACHE_MAX_AGE_DAYS:
        return None, {"miss": f"cached pull is {age} days old (max {CACHE_MAX_AGE_DAYS})"}
    return entry["report"], {"as_of": entry["report"].get("report_pulled_on"), "cached_at": cached_at,
                             "label": "cached last good bureau pull"}


def _fallback_bureau_snapshot(args: dict):
    import risk_agent as ra          # the nightly bureau extract the rule engine also reads
    try:
        report = ra.tool_get_credit_bureau((args.get("applicant_id") or "").strip())
    except ValueError as e:
        return None, {"miss": str(e)}
    return report, {"as_of": report.get("report_pulled_on"), "label": "bureau snapshot (bureau_reports.json)"}


FALLBACKS = {"bureau_cache": _fallback_bureau_cache, "bureau_snapshot": _fallback_bureau_snapshot}


# ---------------------------------------------------------------------------
# Counters (for the bench and the end-of-run summary)
# ---------------------------------------------------------------------------
STATS: dict[str, dict] = {}


def _count(tool: str, key: str, n: int = 1) -> None:
    STATS.setdefault(tool, {}).setdefault(key, 0)
    STATS[tool][key] += n


def reset_stats() -> None:
    STATS.clear()


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------
_JITTER = random.Random(int(os.getenv("RESILIENCE_JITTER_SEED", "7")))


def backoff_delay(policy: ToolPolicy, attempt: int) -> float:
    """Full jitter: random between 0 and the exponential cap, so many callers don't retry in lockstep."""
    cap = min(policy.max_delay, policy.base_delay * (2 ** (attempt - 1)))
    return _JITTER.uniform(0, cap) * TIME_SCALE


async def resilient_call(tool: str, arguments: dict, raw_call, *, policy: ToolPolicy | None = None,
                         breaker: CircuitBreaker | None | bool = None, injector: FaultInjector | None = None):
    """Call `raw_call(arguments)` (an async function: the MCP or in-process call) with retries, validation,
    a circuit breaker and fallbacks. Returns (result, provenance). Raises the permanent error, or
    AllSourcesFailed when nothing worked."""
    policy = policy or policy_for(tool)
    if breaker is False:                       # explicitly disabled (bench)
        breaker = None
    elif breaker is None:
        breaker = get_breaker(tool, policy) if policy.use_breaker else None
    injector = injector if injector is not None else FAULTS.get(tool)
    _count(tool, "requests")
    if injector:
        injector.begin_request()
    errors: list[str] = []
    attempts, bad_data_retries = 0, 0
    t_start = time.monotonic()
    max_attempts = policy.max_attempts

    refused = None
    if breaker is not None:
        try:
            if breaker.before_call() == "half_open":
                max_attempts = 1                    # the probe is a single attempt, no retries
                _count(tool, "probes")
        except CircuitOpenError as e:
            refused = str(e)
            _count(tool, "short_circuited")
            with span("breaker:refused", as_type="event", input={"tool": tool},
                      metadata=breaker.snapshot()) as s:
                s.update(level="WARNING", status_message=refused)

    while refused is None and attempts < max_attempts:
        attempts += 1
        _count(tool, "attempts")
        with span(f"attempt:{attempts}", as_type="span", input={"tool": tool},
                  metadata={"breaker": breaker.state if breaker else "off"}) as s:
            try:
                fault = injector.roll() if injector else "ok"
                if fault == "fail":
                    await asyncio.sleep(injector.cfg.delay * TIME_SCALE)
                    raise ConnectionError(f"[injected] {tool}: connection refused")
                if fault == "timeout":
                    await asyncio.sleep(injector.cfg.delay * TIME_SCALE)
                    raise TimeoutError(f"[injected] {tool}: timed out")
                result = await asyncio.wait_for(raw_call(arguments), timeout=policy.attempt_timeout)
                if fault == "malformed":
                    result = _corrupt(tool, result)
                problems = validate(tool, result) if policy.validate else []
                if problems:
                    raise BadData("; ".join(problems))
            except Exception as e:
                kind = classify(e)
                msg = f"{type(e).__name__}: {e}"[:300]
                if kind == "permanent":
                    if breaker:
                        breaker.release()
                    _count(tool, "permanent_errors")
                    raise                                   # caller's problem: no retry, no fallback
                errors.append(msg)
                _count(tool, kind)
                s.update(level="WARNING", status_message=msg, output={"failure": kind})
                if kind == "bad_data":
                    bad_data_retries += 1
                    if bad_data_retries > policy.retry_bad_data:
                        break
                delay = backoff_delay(policy, attempts)
                if attempts >= max_attempts or time.monotonic() - t_start + delay > policy.total_budget:
                    break
                await asyncio.sleep(delay)
                continue
            if breaker:
                breaker.record_success()
            if tool == "get_credit_bureau":
                cache_bureau(arguments.get("applicant_id"), result)
            _count(tool, "live_ok")
            s.update(output={"ok": True})
            return result, {"source": "live", "stale": False, "attempts": attempts,
                            "retried_errors": errors or None,
                            "breaker": breaker.state if breaker else "off"}

    # ---- live path exhausted (or refused by the breaker): fallbacks, in order --------------------
    if refused is None and breaker is not None:
        was = breaker.state
        breaker.record_failure()                    # one failed REQUEST (retries exhausted)
        if breaker.state == "open" and was != "open":
            _count(tool, "breaker_opened")
            with span("breaker:opened", as_type="event", input={"tool": tool}, metadata=breaker.snapshot()) as ev:
                ev.update(level="WARNING", status_message=f"{tool}: circuit opened after "
                                                          f"{breaker.failures} failed request(s)")
    live_error = refused or (errors[-1] if errors else "no live attempt")
    for name in policy.fallbacks:
        with span(f"fallback:{name}", as_type="span", input={"tool": tool}) as s:
            try:
                result, meta = await asyncio.to_thread(FALLBACKS[name], arguments)
            except Exception as e:
                s.update(level="WARNING", status_message=f"{type(e).__name__}: {e}"[:300])
                continue
            if result is None:
                s.update(output={"miss": meta.get("miss")})
                continue
            problems = validate(tool, result) if policy.validate else []
            if problems:
                s.update(level="WARNING", status_message="fallback data invalid: " + "; ".join(problems))
                continue
            _count(tool, f"fallback_{name}")
            s.update(output={"used": True, "as_of": meta.get("as_of")})
            return result, {"source": name, "stale": True, "label": meta.get("label"), "as_of": meta.get("as_of"),
                            "cached_at": meta.get("cached_at"), "attempts": attempts, "live_error": live_error,
                            "breaker": breaker.state if breaker else "off"}
    _count(tool, "failed")
    raise AllSourcesFailed(f"{tool}: live call failed after {attempts} attempt(s) ({live_error})"
                           + ("; no fallback source had data" if policy.fallbacks else ""))


# ---------------------------------------------------------------------------
# CLI: bench (the B1-B4 comparison on the bureau call) and report
# ---------------------------------------------------------------------------
BENCH_BREAKER_RESET_REQUESTS = 20

STRATEGIES = {
    "B1 no protection": ToolPolicy(max_attempts=1, validate=False, use_breaker=False, fallbacks=()),
    "B2 retries": ToolPolicy(max_attempts=4, validate=False, use_breaker=False, fallbacks=(), base_delay=0.5),
    "B3 retries+validate+fallback": ToolPolicy(max_attempts=4, use_breaker=False,
                                               fallbacks=("bureau_cache", "bureau_snapshot")),
    "B4 + circuit breaker": ToolPolicy(max_attempts=4, use_breaker=True, breaker_reset=5.0,
                                       fallbacks=("bureau_cache", "bureau_snapshot")),
}


async def _bench_one(name: str, policy: ToolPolicy, cfg: FaultConfig, applicants: list[str], n: int) -> dict:
    import risk_agent as ra
    injector = FaultInjector("get_credit_bureau", cfg)
    # The bench breaker counts time in REQUESTS (probe again after 20 requests), not seconds: the shrunken
    # waits would otherwise make the result depend on how fast this machine is. Production uses seconds.
    breaker = CircuitBreaker("bench", policy.breaker_threshold, BENCH_BREAKER_RESET_REQUESTS,
                             clock=lambda: injector.requests) if policy.use_breaker else None
    if _CACHE_PATH and _CACHE_PATH.exists():
        _CACHE_PATH.unlink()
    dep_calls = {"n": 0}

    async def raw(args):
        dep_calls["n"] += 1
        return ra.tool_get_credit_bureau(args["applicant_id"])

    live = stale = failed = bad_passed = 0
    t0 = time.perf_counter()
    for i in range(n):
        aid = applicants[i % len(applicants)]
        try:
            res, prov = await resilient_call("get_credit_bureau", {"applicant_id": aid}, raw,
                                             policy=policy, breaker=breaker or False, injector=injector)
            if validate_bureau(res):
                bad_passed += 1                          # accepted, but it was malformed
            elif prov["source"] == "live":
                live += 1
            else:
                stale += 1
        except (AllSourcesFailed, *TRANSIENT):
            failed += 1
    return {"strategy": name, "good_live": live, "stale_fallback": stale, "bad_data_passed": bad_passed,
            "failed": failed, "dependency_calls": injector.calls,
            "breaker_opened": breaker.opened_count if breaker else "-",
            "seconds": round(time.perf_counter() - t0, 2)}


def bench(n: int, spec: str, outage: str | None) -> None:
    import risk_agent as ra
    cfg = parse_faults(spec).get("get_credit_bureau") or FaultConfig()
    if outage:
        a, _, b = outage.partition("-")
        cfg = replace(cfg, outage=(int(a), int(b)))
    applicants = [r["applicant_id"] for r in ra._load_lookup("bureau_reports.json")][:40]
    print(f"bureau faults: fail={cfg.fail} timeout={cfg.timeout} malformed={cfg.malformed} seed={cfg.seed}"
          + (f" outage=requests {cfg.outage[0]}-{cfg.outage[1] - 1}" if cfg.outage else "")
          + f"   requests={n}  time scale={TIME_SCALE}\n")
    rows = [asyncio.run(_bench_one(name, pol, cfg, applicants, n)) for name, pol in STRATEGIES.items()]
    cols = ["strategy", "good_live", "stale_fallback", "bad_data_passed", "failed", "dependency_calls",
            "breaker_opened", "seconds"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    print(f"\nbreaker (B4): opens after {STRATEGIES['B4 + circuit breaker'].breaker_threshold} failed requests in a row, "
          f"probes again after {BENCH_BREAKER_RESET_REQUESTS} requests")
    print("good_live: validated live data   stale_fallback: cached/snapshot data, disclosed as stale\n"
          "bad_data_passed: malformed data accepted as if good (the silent failure)   "
          "dependency_calls: load put on the bureau")


def report(data_dir: Path) -> None:
    traces = data_dir / "traces.jsonl"
    runs = data_dir / "committee_runs.jsonl"
    if traces.exists():
        rows = [json.loads(l) for l in traces.read_text(encoding="utf-8").splitlines() if l.strip()]
        groups: dict[str, list] = {}
        for r in rows:
            name = r["name"]
            if name.startswith(("committee:", "loan_workflow")):
                continue
            key = re.sub(r":\d+$", ":n", name) if name.startswith("attempt") else name
            groups.setdefault(key, []).append(r)
        print(f"Spans in {traces} ({len(rows)} spans)\n")
        print(f"{'span':<34}{'calls':>7}{'failed':>8}{'warn':>6}{'avg ms':>10}{'p95 ms':>10}{'max ms':>10}")
        for key, rs in sorted(groups.items(), key=lambda kv: -sum(x["ms"] for x in kv[1])):
            ms = sorted(x["ms"] for x in rs)
            p95 = ms[min(len(ms) - 1, int(round(0.95 * (len(ms) - 1))))]
            print(f"{key:<34}{len(rs):>7}{sum(not x['ok'] for x in rs):>8}"
                  f"{sum(x.get('level') == 'WARNING' for x in rs):>6}{statistics.mean(ms):>10.1f}{p95:>10.1f}{ms[-1]:>10.1f}")
    else:
        print(f"No {traces}")
    if runs.exists():
        sources: dict[str, int] = {}
        for l in runs.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            for e in json.loads(l).get("evidence", []):
                if e.get("source") == "bureau":
                    src = (e.get("provenance") or {}).get("source") or e.get("status")
                    sources[src] = sources.get(src, 0) + 1
        print(f"\nBureau evidence by source across committee runs: {sources}")


def main() -> None:
    global TIME_SCALE
    p = argparse.ArgumentParser(description="M7 resilience: bench and report")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--faults", default="get_credit_bureau:fail=0.2,timeout=0.1,malformed=0.1,seed=42")
    b.add_argument("--outage", help="calls A-B fail completely, e.g. 50-150")
    b.add_argument("--time-scale", type=float, default=0.01, help="shrinks backoff waits (default 0.01)")
    r = sub.add_parser("report")
    r.add_argument("--data-dir")
    sub.add_parser("status")
    args = p.parse_args()
    import risk_agent as ra
    data_dir = Path(getattr(args, "data_dir", None) or ra.DATA_FILE.parent)
    configure(data_dir)
    if args.cmd == "bench":
        TIME_SCALE = args.time_scale
        configure(data_dir / "bench")
        bench(args.n, args.faults, args.outage)
    elif args.cmd == "report":
        report(data_dir)
    else:
        for tool, pol in POLICIES.items():
            print(f"{tool:<22} attempts={pol.max_attempts} timeout={pol.attempt_timeout}s budget={pol.total_budget}s "
                  f"breaker={pol.breaker_threshold} fails/{pol.breaker_reset}s fallbacks={list(pol.fallbacks) or '-'}")
        print(f"\nactive fault injection: { {t: vars(i.cfg) for t, i in FAULTS.items()} or 'none' }")


if __name__ == "__main__":
    main()
