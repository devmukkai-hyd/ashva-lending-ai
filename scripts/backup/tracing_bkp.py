"""tracing.py - one tracing layer for the underwriting pipeline (loan_workflow + underwriting committee).

Today (M6) every span is written to a local JSON-lines file, so you can already see each run:
    traces.jsonl     one line per span: trace_id, span_id, parent_id, name, type, ms, ok, level,
                     status_message, input, output, metadata
M7 turns on Langfuse WITHOUT touching the pipeline code: set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY
(and LANGFUSE_HOST) and `pip install langfuse`. Spans then also go to Langfuse, with the same names,
nesting, inputs, outputs and error levels. The local file keeps being written either way.

API
    with trace_run("committee", trace_key="ASH-L-00147", metadata={...}):    # one trace per application
        ...
    @traced("analyst")             # a span named "agent:analyst" around a node (sync or async)
    with span("mcp:get_credit_bureau", as_type="tool", input={...}) as s:
        s.update(output=..., level="ERROR", status_message="...")
    RUN_EVENTS                     # [{"role", "ms", "ok"}], in memory, for quick per-run summaries
    flush()

Interrupts are not errors: LangGraph pauses a graph by raising GraphInterrupt (the human-review step).
traced() records those as level "INFO", status "paused", and re-raises them untouched.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import itertools
import inspect
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

TRACE_FILE = Path(os.getenv("TRACE_FILE", "traces.jsonl"))
MAX_FIELD_CHARS = 4000          # keep trace lines readable; full data lives in the run outputs
PAUSE_EXCEPTIONS = {"GraphInterrupt", "NodeInterrupt", "Interrupt", "_Pause"}

RUN_EVENTS: list[dict] = []     # each entry: {"role": str, "ms": float, "ok": bool}
_SEQ = itertools.count()          # start order of spans within this process (timestamps can tie)

_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
_parent_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("parent_id", default=None)


def set_trace_file(path: Path) -> None:
    """Where local spans go (unless TRACE_FILE is set in the environment)."""
    global TRACE_FILE
    if not os.getenv("TRACE_FILE"):
        TRACE_FILE = Path(path)


# ---------------------------------------------------------------------------
# Langfuse (M7): used only when configured; never required
# ---------------------------------------------------------------------------
_LANGFUSE = None
_LANGFUSE_CHECKED = False


def langfuse_client():
    global _LANGFUSE, _LANGFUSE_CHECKED
    if not _LANGFUSE_CHECKED:
        _LANGFUSE_CHECKED = True
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse import get_client
                _LANGFUSE = get_client()
            except Exception as e:   # not installed / bad keys: local tracing still works
                print(f"[tracing] Langfuse not available ({e}); writing local traces only.")
    return _LANGFUSE


def tracing_backend() -> str:
    return "langfuse + local file" if langfuse_client() else f"local file ({TRACE_FILE})"


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------
def _short(value):
    """JSON-safe, size-capped copy of a span input/output."""
    if value is None:
        return None
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        text = json.dumps(str(value))
    if len(text) <= MAX_FIELD_CHARS:
        return json.loads(text)
    return text[:MAX_FIELD_CHARS] + "...(truncated)"


class _Span:
    def __init__(self, name: str, as_type: str, input, metadata):
        self.name, self.as_type = name, as_type
        self.span_id = uuid.uuid4().hex[:16]
        self.trace_id = _trace_id.get() or uuid.uuid4().hex
        self.parent_id = _parent_id.get()
        self.input, self.output, self.metadata = input, None, dict(metadata or {})
        self.level, self.status_message = "DEFAULT", None
        self.t0 = time.perf_counter()
        self.seq = next(_SEQ)
        self.started = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self._lf = None                       # the Langfuse observation, when Langfuse is on

    def update(self, *, input=None, output=None, metadata=None, level=None, status_message=None):
        if input is not None:
            self.input = input
        if output is not None:
            self.output = output
        if metadata:
            self.metadata.update(metadata)
        if level:
            self.level = level
        if status_message:
            self.status_message = status_message
        if self._lf is not None:
            kw = {k: v for k, v in dict(input=input, output=output, metadata=metadata, level=level,
                                        status_message=status_message).items() if v is not None}
            try:
                self._lf.update(**kw)
            except Exception:
                pass

    def _write(self, ok: bool) -> None:
        row = {"trace_id": self.trace_id, "span_id": self.span_id, "parent_id": self.parent_id,
               "name": self.name, "type": self.as_type, "started_at": self.started, "seq": self.seq,
               "ms": round((time.perf_counter() - self.t0) * 1000, 1), "ok": ok, "level": self.level,
               "status_message": self.status_message, "input": _short(self.input),
               "output": _short(self.output), "metadata": _short(self.metadata) or None}
        try:
            TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(TRACE_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass                                   # tracing must never break the pipeline


@contextlib.contextmanager
def span(name: str, as_type: str = "span", input=None, metadata: dict | None = None):
    """A span around any block. An exception marks it ERROR (unless it is a graph pause) and is re-raised."""
    s = _Span(name, as_type, input, metadata)
    lf = langfuse_client()
    lf_cm = None
    if lf is not None:
        try:
            lf_type = as_type if as_type in ("span", "generation", "tool", "agent", "event") else "span"
            lf_cm = lf.start_as_current_observation(as_type=lf_type, name=name, input=_short(input),
                                                    metadata=s.metadata or None)
            s._lf = lf_cm.__enter__()
        except Exception:
            lf_cm, s._lf = None, None
    tok_parent = _parent_id.set(s.span_id)
    tok_trace = _trace_id.set(s.trace_id)
    ok = True
    try:
        yield s
    except BaseException as e:
        if type(e).__name__ in PAUSE_EXCEPTIONS:
            s.update(level="INFO", status_message="paused (waiting for a person)")
        else:
            ok = False
            s.update(level="ERROR", status_message=f"{type(e).__name__}: {e}"[:500])
        raise
    finally:
        _parent_id.reset(tok_parent)
        _trace_id.reset(tok_trace)
        s._write(ok)
        if lf_cm is not None:
            try:
                lf_cm.__exit__(None, None, None)
            except Exception:
                pass


@contextlib.contextmanager
def trace_run(name: str, trace_key: str | None = None, metadata: dict | None = None, input=None):
    """The root span of one run (one application). trace_key (e.g. the application id) is recorded so
    every span of a case can be found again. Called inside another run (the committee inside the loan
    workflow), it nests as a child span of that trace instead of starting a new one."""
    md = {**(metadata or {}), **({"trace_key": trace_key} if trace_key else {})}
    token = _trace_id.set(_trace_id.get() or uuid.uuid4().hex)
    try:
        with span(f"{name}:{trace_key}" if trace_key else name, as_type="trace", input=input, metadata=md) as s:
            yield s
    finally:
        _trace_id.reset(token)


def current_trace_id() -> str | None:
    return _trace_id.get()


# ---------------------------------------------------------------------------
# @traced: a span per agent / node, sync or async
# ---------------------------------------------------------------------------
def traced(role: str, prefix: str = "agent", reads=None):
    """Span `<prefix>:<role>` around a node. Records the keys the node read (reads(state) -> list, or the
    state's keys), the keys it wrote and its log lines, timing in RUN_EVENTS, and ERROR on exceptions.
    Put it OUTSIDE @scoped so a scope violation is traced too."""
    def decorator(fn):
        name = f"{prefix}:{role}"

        def _input(state):
            try:
                keys = reads(state) if reads else (sorted(state) if isinstance(state, dict) else [])
            except Exception:
                keys = []
            return {"read": sorted(keys)}

        def _output(s, result):
            result = result if isinstance(result, dict) else {}
            s.update(output={"wrote": sorted(set(result) - {"log", "events"}),
                             "log": result.get("log", [])})

        def _record(t0, ok):
            RUN_EVENTS.append({"role": role, "ms": round((time.perf_counter() - t0) * 1000, 1), "ok": ok})

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(state, *args, **kwargs):
                t0, ok = time.perf_counter(), True
                try:
                    with span(name, input=_input(state)) as s:
                        result = await fn(state, *args, **kwargs)
                        _output(s, result)
                        return result
                except BaseException as e:
                    ok = type(e).__name__ in PAUSE_EXCEPTIONS
                    raise
                finally:
                    _record(t0, ok)
            return awrapper

        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            t0, ok = time.perf_counter(), True
            try:
                with span(name, input=_input(state)) as s:
                    result = fn(state, *args, **kwargs)
                    _output(s, result)
                    return result
            except BaseException as e:
                ok = type(e).__name__ in PAUSE_EXCEPTIONS
                raise
            finally:
                _record(t0, ok)
        return wrapper
    return decorator


def flush() -> None:
    lf = langfuse_client()
    if lf is not None:
        try:
            lf.flush()
        except Exception:
            pass


def span_tree(spans: list[dict]) -> list[tuple[int, dict]]:
    """(depth, span) in start order, children under their parent."""
    ids = {x["span_id"] for x in spans}
    kids: dict = {}
    for x in spans:
        kids.setdefault(x["parent_id"] if x["parent_id"] in ids else None, []).append(x)
    order = lambda x: (x["started_at"], x.get("seq", 0))
    out = []

    def walk(parent, depth):
        for x in sorted(kids.get(parent, []), key=order):
            out.append((depth, x))
            walk(x["span_id"], depth + 1)
    walk(None, 0)
    return out


def read_trace(trace_key: str, path: Path | None = None) -> list[dict]:
    """All local spans recorded for one application (every run), oldest first."""
    path = path or TRACE_FILE
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    trace_ids = {r["trace_id"] for r in rows if (r.get("metadata") or {}).get("trace_key") == trace_key}
    return [r for r in rows if r["trace_id"] in trace_ids]
