"""
ML-Enhanced Real-Time Task Scheduler
------------------------------------
A self-contained Streamlit app that simulates a single-CPU real-time system and
compares three scheduling policies on the same workload:

    FIFO    first come, first served
    EDF     earliest absolute deadline first (never uses ML predictions)
    ML-EDF  EDF-inspired policy that uses ML-predicted execution time and
            ML-predicted deadline-miss probability

Run with:   streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SCHEDULERS = ["FIFO", "EDF", "ML-EDF"]
COLORS = {"FIFO": "#64748B", "EDF": "#2563EB", "ML-EDF": "#7C3AED"}
INK, MUTED, LINE = "#14213D", "#5A6785", "#E2E8F2"
GOOD, BAD, IDLE_C, SWITCH_C = "#1F9D63", "#E5484D", "#CBD5E1", "#F59F00"
DEFS = {
    "FIFO": "First come, first served. It ignores deadlines.",
    "EDF": "Always runs the task whose deadline is nearest.",
    "ML-EDF": "EDF plus machine-learning guesses of run time and lateness risk.",
}
SCENARIOS = {
    "Relaxed": ("Light", "Tasks arrive slowly and deadlines are generous. Most schedulers cope."),
    "Busy": ("Moderate", "The processor stays busy and deadlines are tight."),
    "Overloaded": ("Overloaded", "More work arrives than fits, so some deadlines must be missed."),
}
SCENARIO_OF = {v[0]: k for k, v in SCENARIOS.items()}

# Task types are workload LABELS. IO tasks are NOT simulated as blocking on I/O.
TASK_TYPES = ["CPU", "IO", "PERIODIC", "APERIODIC"]
TYPE_COST = {"CPU": 0.8, "IO": 0.4, "PERIODIC": 0.5, "APERIODIC": 1.2}
TYPE_CODE = {t: i for i, t in enumerate(TASK_TYPES)}

REQUIRED_COLS = ["task_id", "arrival_time", "execution_time", "deadline"]
WORKLOAD_COLS = REQUIRED_COLS + ["priority", "task_type", "input_size"]
REG_FEATURES = ["input_size", "type_code", "priority"]
CLF_FEATURES = REG_FEATURES + ["rel_deadline", "local_load"]

# (offered load, slack factor min, slack factor max)
DIFFICULTY = {
    "Light": (0.5, 2.5, 4.0),
    "Moderate": (0.85, 1.5, 3.0),
    "Overloaded": (1.25, 1.2, 2.0),
}

ML_VARIANTS = {
    "slack_risk": "Slack - risk (original ML-EDF formula)",
    "risk_deadline": "Risk-adjusted deadline (static key, EDF-like)",
}

# (key, label, which direction is better)
METRICS = [
    ("completed_tasks", "Completed tasks", "equal"),
    ("total_simulation_time", "Total simulation time", "lower"),
    ("deadline_misses", "Deadline misses", "lower"),
    ("deadline_miss_rate_pct", "Deadline-miss rate (%)", "lower"),
    ("avg_waiting_time", "Avg waiting time", "lower"),
    ("avg_turnaround_time", "Avg turnaround time", "lower"),
    ("avg_response_time", "Avg response time", "lower"),
    ("avg_tardiness", "Avg tardiness", "lower"),
    ("max_tardiness", "Max tardiness", "lower"),
    ("cpu_utilization_pct", "CPU utilization (%)", "context"),
    ("throughput", "Throughput (tasks / time unit)", "higher"),
    ("context_switches", "Context switches", "lower"),
    ("context_switch_time", "Context-switch time", "lower"),
    ("preemptions", "Preemptions", "context"),
    ("idle_time", "Idle time", "context"),
]
LABELS = {k: l for k, l, _ in METRICS}


# --------------------------------------------------------------------------- #
# Custom Min-Heap (used as the ready queue)
# --------------------------------------------------------------------------- #
class MinHeap:
    """Binary min-heap.

    Complexity: push O(log n), pop O(log n), peek O(1), size O(1),
    is_empty O(1), clear O(1), heapify O(n).
    Items are compared with `<`, so tuples work as (key, payload).
    """

    def __init__(self, items=None):
        self._a: list = []
        if items:
            self.heapify(items)

    def __len__(self) -> int:
        return len(self._a)

    def size(self) -> int:
        return len(self._a)

    def is_empty(self) -> bool:
        return not self._a

    def clear(self) -> None:
        self._a = []

    def peek(self):
        if not self._a:
            raise IndexError("peek from an empty heap")
        return self._a[0]

    def push(self, item) -> None:
        a = self._a
        a.append(item)
        i = len(a) - 1
        while i > 0:
            p = (i - 1) // 2
            if a[i] < a[p]:
                a[i], a[p] = a[p], a[i]
                i = p
            else:
                break

    def pop(self):
        a = self._a
        if not a:
            raise IndexError("pop from an empty heap")
        top = a[0]
        last = a.pop()
        if a:
            a[0] = last
            self._sift_down(0)
        return top

    def heapify(self, items) -> None:
        self._a = list(items)
        for i in range(len(self._a) // 2 - 1, -1, -1):
            self._sift_down(i)

    def _sift_down(self, i: int) -> None:
        a = self._a
        n = len(a)
        while True:
            left, right, m = 2 * i + 1, 2 * i + 2, i
            if left < n and a[left] < a[m]:
                m = left
            if right < n and a[right] < a[m]:
                m = right
            if m == i:
                break
            a[i], a[m] = a[m], a[i]
            i = m


# --------------------------------------------------------------------------- #
# Workload generation, features and CSV validation
# --------------------------------------------------------------------------- #
def generate_workload(n_tasks: int, load: float, slack_min: float, slack_max: float, seed: int) -> pd.DataFrame:
    """Reproducible synthetic workload. Execution time is hidden truth that
    depends on input_size and task_type (plus noise)."""
    rng = np.random.default_rng(seed)
    input_size = rng.uniform(10, 100, n_tasks).round(1)
    task_type = rng.choice(TASK_TYPES, n_tasks)
    priority = rng.integers(1, 6, n_tasks)  # 1 = most important

    cost = np.array([TYPE_COST[t] for t in task_type])
    base = input_size * cost / 8.0
    execution = np.maximum(1, np.round(base * rng.normal(1.0, 0.15, n_tasks))).astype(int)

    gaps = rng.exponential(execution.mean() / load, n_tasks)
    arrival = np.floor(np.cumsum(gaps)).astype(int)
    arrival -= arrival[0]

    slack_factor = rng.uniform(slack_min, slack_max, n_tasks)
    deadline = arrival + np.ceil(execution * slack_factor).astype(int)

    return pd.DataFrame(
        {
            "task_id": np.arange(1, n_tasks + 1),  # starts at 1 so it differs from the row index
            "arrival_time": arrival,
            "execution_time": execution,
            "deadline": deadline,
            "priority": priority,
            "task_type": task_type,
            "input_size": input_size,
        }
    )[WORKLOAD_COLS]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["type_code"] = out["task_type"].map(TYPE_CODE)
    out["rel_deadline"] = out["deadline"] - out["arrival_time"]
    a = np.sort(out["arrival_time"].to_numpy())
    x = out["arrival_time"].to_numpy()
    # tasks that arrived in the 15 time units up to and including this arrival
    out["local_load"] = np.searchsorted(a, x, side="right") - np.searchsorted(a, x - 15, side="left")
    return out


def workload_fingerprint(df: pd.DataFrame) -> str:
    text = df[WORKLOAD_COLS].to_csv(index=False)
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def _rows(mask) -> str:
    idx = np.flatnonzero(np.asarray(mask))
    shown = ", ".join(str(i) for i in idx[:5])
    return shown + (" ..." if len(idx) > 5 else "")


def validate_workload(raw: pd.DataFrame | None):
    """Return (clean_df | None, errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    if raw is None or len(raw) == 0:
        return None, ["The file contains no rows."], []

    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return None, [f"Missing required column(s): {', '.join(missing)}. Required: {', '.join(REQUIRED_COLS)}."], []

    for c in REQUIRED_COLS:
        num = pd.to_numeric(df[c], errors="coerce")
        if num.isna().any():
            errors.append(f"Column '{c}' has empty or non-numeric values at row index {_rows(num.isna())}.")
        elif (num != np.floor(num)).any():
            errors.append(f"Column '{c}' must hold whole numbers because the simulator uses integer time units.")
        df[c] = num
    if errors:
        return None, errors, warnings

    for c in REQUIRED_COLS:
        df[c] = df[c].astype(int)

    if df["task_id"].duplicated().any():
        errors.append(f"task_id values must be unique; duplicates at row index {_rows(df['task_id'].duplicated())}.")
    if (df["arrival_time"] < 0).any():
        errors.append(f"arrival_time must be >= 0; invalid at row index {_rows(df['arrival_time'] < 0)}.")
    if (df["execution_time"] <= 0).any():
        errors.append(f"execution_time must be > 0; invalid at row index {_rows(df['execution_time'] <= 0)}.")
    if (df["deadline"] <= df["arrival_time"]).any():
        errors.append(f"deadline must be later than arrival_time; invalid at row index {_rows(df['deadline'] <= df['arrival_time'])}.")

    if "priority" in df.columns:
        pr = pd.to_numeric(df["priority"], errors="coerce")
        if pr.isna().any() or (pr != np.floor(pr.fillna(0))).any():
            errors.append("priority must be a whole number (lower value = higher priority).")
        else:
            df["priority"] = pr.astype(int)
    else:
        df["priority"] = 3
        warnings.append("No 'priority' column: using 3 for every task.")

    if "task_type" in df.columns:
        df["task_type"] = df["task_type"].astype(str).str.strip().str.upper()
        bad = ~df["task_type"].isin(TASK_TYPES)
        if bad.any():
            errors.append(f"task_type must be one of {', '.join(TASK_TYPES)}; invalid at row index {_rows(bad)}.")
    else:
        df["task_type"] = "CPU"
        warnings.append("No 'task_type' column: using CPU for every task.")

    if "input_size" in df.columns:
        sz = pd.to_numeric(df["input_size"], errors="coerce")
        if sz.isna().any() or (sz <= 0).any():
            errors.append("input_size must be a positive number.")
        else:
            df["input_size"] = sz.astype(float)
    else:
        df["input_size"] = 55.0
        warnings.append("No 'input_size' column: using 55 for every task, so ML predictions will carry little information.")

    infeasible = df["deadline"] < df["arrival_time"] + df["execution_time"]
    if infeasible.any():
        warnings.append(f"{int(infeasible.sum())} task(s) cannot meet their deadline even on an idle CPU. This is allowed.")

    if errors:
        return None, errors, warnings
    return df[WORKLOAD_COLS].reset_index(drop=True), [], warnings


# --------------------------------------------------------------------------- #
# Simulation engine
# --------------------------------------------------------------------------- #
class SimResult:
    def __init__(self, tasks: pd.DataFrame, trace: pd.DataFrame, summary: dict, fingerprint: str):
        self.tasks = tasks
        self.trace = trace
        self.summary = summary
        self.fingerprint = fingerprint


TRACE_COLS = ["start_time", "end_time", "task_id", "event_type"]


def simulate(
    df: pd.DataFrame,
    policy: str,
    preemptive: bool = True,
    cs_cost: int = 0,
    risk_weight: float = 2.0,
    ml_variant: str = "slack_risk",
) -> SimResult:
    """Simulate one policy on a single CPU with a virtual clock (1 time unit per step).

    Conventions
    - Arrivals at time t are admitted before any scheduling decision at t.
    - The ready queue is a MinHeap. The running task is held outside the heap.
      In preemptive mode it is compared with heap.peek() every tick and is
      displaced only by a STRICTLY smaller key.
    - A context switch happens when the CPU dispatches a task different from the
      last one that ran. The first dispatch and a dispatch after an idle gap are free.
      A switch costs `cs_cost` time units, is recorded as CONTEXT_SWITCH, and is atomic.
    - A preemption is a switch away from an unfinished task.
    - Actual execution_time is always used to run tasks. Predictions only affect keys.
    - A task finishing exactly at its deadline is on time.
    """
    if policy not in SCHEDULERS:
        raise ValueError(f"Unknown policy: {policy}")
    fingerprint = workload_fingerprint(df) if len(df) else "empty"
    n = len(df)
    if n == 0:
        empty = pd.DataFrame(columns=WORKLOAD_COLS)
        return SimResult(empty, pd.DataFrame(columns=TRACE_COLS), {k: 0 for k, _, _ in METRICS}, fingerprint)

    ids = df["task_id"].to_numpy()
    arr = df["arrival_time"].to_numpy()
    ex = df["execution_time"].to_numpy()
    dl = df["deadline"].to_numpy()
    prio = df["priority"].to_numpy()
    if policy == "ML-EDF":
        if not {"pred_exec", "miss_risk"}.issubset(df.columns):
            raise ValueError("ML-EDF requires 'pred_exec' and 'miss_risk' columns.")
        pred = df["pred_exec"].to_numpy(dtype=float)
        risk = df["miss_risk"].to_numpy(dtype=float)

    remaining = ex.copy()
    start = np.full(n, -1, dtype=int)
    finish = np.full(n, -1, dtype=int)

    def key(i: int):
        if policy == "FIFO":
            return (arr[i], ids[i])
        if policy == "EDF":
            return (dl[i], prio[i], arr[i], ids[i])
        # ML-EDF. The current time t is the same for every waiting task, so it
        # cancels out of comparisons and is omitted from the stored key.
        rem_pred = max(0.5, pred[i] - (ex[i] - remaining[i]))
        if ml_variant == "slack_risk":
            score = dl[i] - rem_pred - risk_weight * risk[i]  # (D - t - C_hat) - w*P, minus t
        else:
            score = dl[i] - risk_weight * risk[i] * pred[i]  # static key: does not change while the task runs
        return (score, dl[i], prio[i], arr[i], ids[i])

    order = np.lexsort((ids, arr))
    ptr = 0
    heap = MinHeap()
    t = t0 = int(arr.min())
    events: list[list] = []
    busy = ctx = preempt = done = 0
    cur = None
    last = None

    def add_event(s, e, tid, typ):
        if typ == "RUN" and events and events[-1][3] == "RUN" and events[-1][2] == tid and events[-1][1] == s:
            events[-1][1] = e
        else:
            events.append([s, e, tid, typ])

    def admit(now):
        nonlocal ptr
        while ptr < n and arr[order[ptr]] <= now:
            i = int(order[ptr])
            heap.push((key(i), i))
            ptr += 1

    while done < n:
        admit(t)

        if cur is None and heap.is_empty():  # idle: jump to next arrival
            nxt = int(arr[order[ptr]])
            add_event(t, nxt, -1, "IDLE")
            t = nxt
            last = None
            continue

        if cur is None:
            cur = heap.pop()[1]
        elif preemptive and not heap.is_empty() and heap.peek()[0] < key(cur):
            nxt_item = heap.pop()
            heap.push((key(cur), cur))
            cur = nxt_item[1]

        if cur != last:
            if last is not None:
                ctx += 1
                if remaining[last] > 0:
                    preempt += 1
                if cs_cost > 0:
                    add_event(t, t + cs_cost, int(ids[cur]), "CONTEXT_SWITCH")
                    t += cs_cost
                    admit(t)

        if start[cur] < 0:
            start[cur] = t
        remaining[cur] -= 1
        busy += 1
        add_event(t, t + 1, int(ids[cur]), "RUN")
        t += 1
        last = cur

        if remaining[cur] == 0:
            finish[cur] = t
            done += 1
            cur = None

    res = df[WORKLOAD_COLS + [c for c in ("pred_exec", "miss_risk") if c in df.columns]].copy().reset_index(drop=True)
    res["start_time"] = start
    res["completion_time"] = finish
    res["remaining_time"] = remaining
    res["turnaround_time"] = finish - arr
    res["waiting_time"] = res["turnaround_time"] - ex
    res["response_time"] = start - arr
    res["tardiness"] = np.maximum(0, finish - dl)
    res["deadline_missed"] = finish > dl

    trace = pd.DataFrame(events, columns=TRACE_COLS)
    dur = trace["end_time"] - trace["start_time"]
    t_end = int(t)
    span = max(1, t_end - t0)
    summary = {
        "completed_tasks": n,
        "total_simulation_time": t_end,
        "deadline_misses": int(res["deadline_missed"].sum()),
        "deadline_miss_rate_pct": round(100 * float(res["deadline_missed"].mean()), 2),
        "avg_waiting_time": round(float(res["waiting_time"].mean()), 3),
        "avg_turnaround_time": round(float(res["turnaround_time"].mean()), 3),
        "avg_response_time": round(float(res["response_time"].mean()), 3),
        "avg_tardiness": round(float(res["tardiness"].mean()), 3),
        "max_tardiness": int(res["tardiness"].max()),
        "cpu_utilization_pct": round(100 * busy / span, 2),
        "throughput": round(n / span, 4),
        "context_switches": int(ctx),
        "context_switch_time": int(dur[trace["event_type"] == "CONTEXT_SWITCH"].sum()),
        "preemptions": int(preempt),
        "idle_time": int(dur[trace["event_type"] == "IDLE"].sum()),
    }
    return SimResult(res, trace, summary, fingerprint)


def validate_run(workload: pd.DataFrame, res: SimResult, cs_cost: int) -> list[dict]:
    """Independent invariant checks on a simulation result."""
    checks: list[dict] = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    tr, tk = res.trace, res.tasks
    if len(workload) == 0:
        add("Empty workload produces an empty trace", len(tr) == 0)
        return checks

    w = workload.set_index("task_id")
    t = tk.set_index("task_id")
    add("Every task appears exactly once in the results",
        tk["task_id"].is_unique and set(tk["task_id"]) == set(w.index))
    t = t.loc[w.index]
    same = all((t[c].to_numpy() == w[c].to_numpy()).all() for c in ["arrival_time", "execution_time", "deadline"])
    add("Results use the same arrival, execution and deadline values as the workload table", same)
    add("Workload was not modified by the simulation", res.fingerprint == workload_fingerprint(workload))

    s = tr["start_time"].to_numpy()
    e = tr["end_time"].to_numpy()
    add("Every trace event has end_time > start_time", (e > s).all())
    add("Trace is chronological with no overlapping CPU intervals", (s[1:] >= e[:-1]).all())
    add("Trace has no unexplained gaps (idle time is recorded explicitly)", (s[1:] == e[:-1]).all())
    first_arrival = int(w["arrival_time"].min())
    add("Trace starts at the earliest arrival", s[0] == first_arrival)

    run = tr[tr["event_type"] == "RUN"]
    run_dur = (run["end_time"] - run["start_time"]).groupby(run["task_id"]).sum().reindex(w.index).fillna(0)
    add("Sum of RUN durations equals the actual execution time of every task",
        (run_dur.to_numpy() == w["execution_time"].to_numpy()).all())
    add("No task runs before its arrival",
        (run["start_time"].to_numpy() >= run["task_id"].map(w["arrival_time"]).to_numpy()).all())

    first_run = run.groupby("task_id")["start_time"].min().reindex(w.index).to_numpy()
    last_run = run.groupby("task_id")["end_time"].max().reindex(w.index).to_numpy()
    add("start_time and completion_time match the first and last RUN events",
        (first_run == t["start_time"].to_numpy()).all() and (last_run == t["completion_time"].to_numpy()).all())
    add("completion_time >= start_time + execution_time",
        (t["completion_time"].to_numpy() >= t["start_time"].to_numpy() + t["execution_time"].to_numpy()).all())
    add("start_time >= arrival_time", (t["start_time"].to_numpy() >= t["arrival_time"].to_numpy()).all())
    add("deadline_missed equals (completion_time > deadline)",
        (t["deadline_missed"].to_numpy() == (t["completion_time"].to_numpy() > t["deadline"].to_numpy())).all())

    d = tr["end_time"] - tr["start_time"]
    run_total = int(d[tr["event_type"] == "RUN"].sum())
    idle_total = int(d[tr["event_type"] == "IDLE"].sum())
    cs_events = d[tr["event_type"] == "CONTEXT_SWITCH"]
    makespan = int(e[-1])
    add("Total RUN time equals the sum of all actual execution times", run_total == int(w["execution_time"].sum()))
    add("Total simulation time = first arrival + RUN + IDLE + context-switch time",
        makespan == first_arrival + run_total + idle_total + int(cs_events.sum()),
        f"{makespan} vs {first_arrival} + {run_total} + {idle_total} + {int(cs_events.sum())}")
    add("Every context-switch event lasts exactly the configured cost", (cs_events == cs_cost).all())
    add("Number of context-switch events matches the reported count (when cost > 0)",
        cs_cost == 0 or len(cs_events) == res.summary["context_switches"])
    add("Reported total simulation time matches the trace", res.summary["total_simulation_time"] == makespan)
    util = 100 * run_total / max(1, makespan - first_arrival)
    add("CPU utilization recomputed from the trace matches the reported value",
        abs(util - res.summary["cpu_utilization_pct"]) < 0.01)
    return checks


def run_schedulers(work, active, preemptive, cs_cost, risk_weight, variant) -> dict[str, SimResult]:
    return {s: simulate(work, s, preemptive, cs_cost, risk_weight, variant) for s in active}


def summary_table(results: dict[str, SimResult]) -> pd.DataFrame:
    table = pd.DataFrame({s: r.summary for s, r in results.items()}).T
    table.index.name = "scheduler"
    return table


# --------------------------------------------------------------------------- #
# Machine learning
# --------------------------------------------------------------------------- #
def check_features(X: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise ValueError(f"Missing model input column(s): {', '.join(missing)}")
    vals = X[cols].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("Model inputs contain NaN or infinite values (check task_type and numeric columns).")


def sanitize_predictions(pred, risk, fallback_exec: float):
    """Replace non-finite values and clip to valid ranges. Returns (pred, risk, n_repaired)."""
    pred = np.asarray(pred, dtype=float).copy()
    risk = np.asarray(risk, dtype=float).copy()
    bad_p = ~np.isfinite(pred) | (pred <= 0)
    bad_r = ~np.isfinite(risk) | (risk < 0) | (risk > 1)
    pred[bad_p] = max(1.0, float(fallback_exec))
    risk[~np.isfinite(risk)] = 0.0
    risk = np.clip(risk, 0.0, 1.0)
    return pred, risk, int(bad_p.sum() + bad_r.sum())


@st.cache_resource(show_spinner="Training ML models...")
def train_models(n_workloads: int, tasks_per_workload: int, n_estimators: int, seed: int,
                 preemptive: bool, cs_cost: int) -> dict:
    """Train both models on synthetic workloads that are separate from the workload being scheduled.

    Miss labels come from simulating EDF (same preemption / context-switch settings)
    on each training workload. The train/test split is by workload so tasks that
    share a congested queue never straddle the split.
    """
    frames = []
    for k in range(n_workloads):
        s = seed + 1000 + k
        prng = np.random.default_rng(s + 7)
        w = generate_workload(tasks_per_workload, prng.uniform(0.6, 1.3), 1.2, prng.uniform(2.0, 4.0), s)
        sim = simulate(w, "EDF", preemptive, cs_cost)
        w = add_features(w)
        w["missed"] = sim.tasks["deadline_missed"].astype(int).to_numpy()
        w["workload"] = k
        frames.append(w)
    data = pd.concat(frames, ignore_index=True)

    n_test = max(2, int(0.2 * n_workloads))
    train = data[data["workload"] < n_workloads - n_test]
    test = data[data["workload"] >= n_workloads - n_test]

    reg = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    reg.fit(train[REG_FEATURES], train["execution_time"])
    reg_pred = reg.predict(test[REG_FEATURES])
    reg_metrics = {
        "MAE": float(mean_absolute_error(test["execution_time"], reg_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(test["execution_time"], reg_pred))),
        "R2": float(r2_score(test["execution_time"], reg_pred)),
    }

    clf, clf_metrics, cm, clf_imp = None, {}, None, None
    if train["missed"].nunique() == 2:
        clf = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1,
        )
        clf.fit(train[CLF_FEATURES], train["missed"])
        proba = clf.predict_proba(test[CLF_FEATURES])[:, list(clf.classes_).index(1)]
        label = (proba >= 0.5).astype(int)
        y = test["missed"]
        clf_metrics = {
            "Accuracy": float(accuracy_score(y, label)),
            "Precision": float(precision_score(y, label, zero_division=0)),
            "Recall": float(recall_score(y, label, zero_division=0)),
            "F1": float(f1_score(y, label, zero_division=0)),
            "Brier score": float(brier_score_loss(y, proba)),
        }
        if y.nunique() == 2:
            clf_metrics["ROC-AUC"] = float(roc_auc_score(y, proba))
            clf_metrics["PR-AUC"] = float(average_precision_score(y, proba))
        cm = confusion_matrix(y, label, labels=[0, 1])
        clf_imp = pd.Series(clf.feature_importances_, index=CLF_FEATURES)

    return {
        "reg": reg,
        "clf": clf,
        "reg_metrics": reg_metrics,
        "clf_metrics": clf_metrics,
        "confusion": cm,
        "reg_test": pd.DataFrame({"actual": test["execution_time"].to_numpy(), "predicted": reg_pred}),
        "reg_importance": pd.Series(reg.feature_importances_, index=REG_FEATURES),
        "clf_importance": clf_imp,
        "exec_fallback": float(train["execution_time"].mean()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_miss_rate": float(train["missed"].mean()),
        "test_miss_rate": float(test["missed"].mean()),
        "config": {"n_workloads": n_workloads, "tasks_per_workload": tasks_per_workload,
                   "n_estimators": n_estimators, "seed": seed, "preemptive": preemptive, "cs_cost": cs_cost},
    }


def add_predictions(df: pd.DataFrame, models: dict):
    """Attach pred_exec and miss_risk. Returns (dataframe, number_of_repaired_values)."""
    out = add_features(df)
    check_features(out, CLF_FEATURES)
    pred = models["reg"].predict(out[REG_FEATURES])
    if models["clf"] is not None:
        risk = models["clf"].predict_proba(out[CLF_FEATURES])[:, list(models["clf"].classes_).index(1)]
    else:
        risk = np.zeros(len(out))
    pred, risk, repaired = sanitize_predictions(pred, risk, models["exec_fallback"])
    out["pred_exec"] = pred
    out["miss_risk"] = risk
    return out, repaired


def prediction_analysis(work, results, risk_weight, variant) -> pd.DataFrame:
    a = work[["task_id", "arrival_time", "execution_time", "deadline", "pred_exec", "miss_risk"]].copy()
    a["prediction_error"] = a["pred_exec"] - a["execution_time"]
    a["abs_error"] = a["prediction_error"].abs()
    a["slack_at_arrival"] = a["deadline"] - a["arrival_time"] - a["pred_exec"]
    if variant == "slack_risk":
        a["ml_edf_score"] = a["slack_at_arrival"] - risk_weight * a["miss_risk"]
    else:
        a["ml_edf_score"] = a["deadline"] - risk_weight * a["miss_risk"] * a["pred_exec"]
    if "EDF" in results:
        a["missed_under_EDF"] = results["EDF"].tasks["deadline_missed"].to_numpy()
    return a


# --------------------------------------------------------------------------- #
# Plots and result helpers
# --------------------------------------------------------------------------- #
def style_fig(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        font=dict(family="Manrope, system-ui, sans-serif", color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.18, font=dict(size=12)),
        title=dict(font=dict(size=15, color=INK)),
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE)
    if height:
        fig.update_layout(height=height)
    return fig


def gantt_figure(res: SimResult, title: str, t_lo: int, t_hi: int, color: str = INK) -> go.Figure:
    tr = res.trace
    tr = tr[(tr["end_time"] > t_lo) & (tr["start_time"] < t_hi)]
    tk = res.tasks.set_index("task_id")
    run = tr[tr["event_type"] == "RUN"]
    idle = tr[tr["event_type"] == "IDLE"]
    csw = tr[tr["event_type"] == "CONTEXT_SWITCH"]
    run_labels = "Task " + run["task_id"].astype(str)
    missed = run["task_id"].map(tk["deadline_missed"]).astype(bool)

    fig = go.Figure()

    def bars(part, labels, name, col):
        if len(part) == 0:
            return
        fig.add_trace(go.Bar(
            x=part["end_time"] - part["start_time"], base=part["start_time"], y=labels,
            orientation="h", name=name, marker_color=col, marker_line_width=0,
            customdata=np.c_[part["start_time"], part["end_time"]],
            hovertemplate="%{y}<br>time %{customdata[0]} to %{customdata[1]}<extra></extra>",
        ))

    bars(run[~missed], run_labels[~missed], "Running, will finish on time", GOOD)
    bars(run[missed], run_labels[missed], "Running, will finish late", BAD)
    bars(idle, pd.Series("IDLE", index=idle.index), "Processor idle", IDLE_C)
    bars(csw, pd.Series("SWITCHING", index=csw.index), "Switching tasks", SWITCH_C)

    rows = (["IDLE"] if len(idle) else []) + (["SWITCHING"] if len(csw) else []) + list(dict.fromkeys(run_labels))
    ids = run["task_id"].drop_duplicates()
    dls = ids.map(tk["deadline"])
    keep = (dls >= t_lo) & (dls <= t_hi)
    if keep.any():
        fig.add_trace(go.Scatter(
            x=dls[keep], y="Task " + ids[keep].astype(str), mode="markers", name="Deadline",
            marker=dict(symbol="diamond", size=9, color=INK, line=dict(color="white", width=1)),
            hovertemplate="%{y}<br>deadline %{x}<extra></extra>",
        ))
    style_fig(fig)
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=color)), barmode="overlay",
        xaxis=dict(range=[t_lo, t_hi], title="Time", gridcolor=LINE),
        yaxis=dict(type="category", categoryorder="array", categoryarray=rows, autorange="reversed", title=None),
        height=int(min(900, max(300, 20 * len(rows) + 150))), bargap=0.25,
    )
    return fig


def best_by_metric(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, label, direction in METRICS:
        if direction not in ("lower", "higher") or k not in table.columns:
            continue
        col = table[k].astype(float)
        best = col.min() if direction == "lower" else col.max()
        winners = [s for s in table.index if np.isclose(col[s], best)]
        rows.append({"metric": label, "better when": direction,
                     "best": ", ".join(winners) if len(winners) < len(table) else "all tied", "value": best})
    return pd.DataFrame(rows)


def run_experiment(sizes, levels, n_seeds, base_seed, models, preemptive, cs_cost, risk_weight, variant, progress=None):
    rows = []
    active = SCHEDULERS if models else ["FIFO", "EDF"]
    total = len(sizes) * len(levels) * n_seeds
    done = 0
    for size in sizes:
        for level in levels:
            load, smin, smax = DIFFICULTY[level]
            for i in range(n_seeds):
                seed = base_seed + 5000 + i
                w = generate_workload(size, load, smin, smax, seed)
                work = add_predictions(w, models)[0] if models else w
                res = run_schedulers(work, active, preemptive, cs_cost, risk_weight, variant)
                for name, r in res.items():
                    ok = all(c["passed"] for c in validate_run(w, r, cs_cost))
                    rows.append({"tasks": size, "difficulty": level, "seed": seed, "scheduler": name,
                                 "integrity_ok": ok, **r.summary})
                done += 1
                if progress:
                    progress.progress(done / total)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Built-in self-tests (hand-computed expectations, independent of the simulator code)
# --------------------------------------------------------------------------- #
def _wl(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["task_id", "arrival_time", "execution_time", "deadline", "priority"])
    df["task_type"] = "CPU"
    df["input_size"] = 50.0
    df["pred_exec"] = df["execution_time"].astype(float)
    df["miss_risk"] = 0.0
    return df


def _fin(res: SimResult) -> dict:
    return dict(zip(res.tasks["task_id"], res.tasks["completion_time"]))


def run_self_tests() -> pd.DataFrame:
    out = []

    def case(name, fn):
        try:
            ok, detail = fn()
        except Exception as exc:  # report, never crash the app
            ok, detail = False, f"exception: {exc!r}"
        out.append({"test": name, "passed": bool(ok), "detail": detail})

    def t_heap():
        rng = np.random.default_rng(1)
        vals = list(rng.integers(0, 20, 200))  # duplicates included
        h = MinHeap()
        for v in vals:
            h.push(v)
        popped = [h.pop() for _ in range(len(vals))]
        h2 = MinHeap(vals)
        ok = popped == sorted(vals) and [h2.pop() for _ in range(len(vals))] == sorted(vals)
        h3 = MinHeap()
        try:
            h3.pop()
            raised = False
        except IndexError:
            raised = True
        h3.push(5)
        ok = ok and raised and h3.peek() == 5 and h3.size() == 1
        h3.clear()
        return ok and h3.is_empty(), "sorted pops, duplicates, heapify, empty/single/clear"

    def t_fifo():
        r = simulate(_wl([(1, 0, 3, 10, 3), (2, 1, 2, 4, 3)]), "FIFO")
        return _fin(r) == {1: 3, 2: 5} and r.summary["deadline_misses"] == 1, f"finish={_fin(r)}"

    def t_edf_pre():
        r = simulate(_wl([(1, 0, 3, 10, 3), (2, 1, 2, 4, 3)]), "EDF", True)
        s = r.summary
        return (_fin(r) == {1: 5, 2: 3} and s["preemptions"] == 1 and s["context_switches"] == 2
                and s["deadline_misses"] == 0), f"finish={_fin(r)}, preemptions={s['preemptions']}, switches={s['context_switches']}"

    def t_edf_nonpre():
        r = simulate(_wl([(1, 0, 3, 10, 3), (2, 1, 2, 4, 3)]), "EDF", False)
        return _fin(r) == {1: 3, 2: 5} and r.summary["preemptions"] == 0, f"finish={_fin(r)}"

    def t_idle():
        r = simulate(_wl([(1, 0, 1, 5, 3), (2, 4, 1, 9, 3)]), "FIFO")
        idle = r.trace[r.trace["event_type"] == "IDLE"]
        ok = len(idle) == 1 and idle.iloc[0]["start_time"] == 1 and idle.iloc[0]["end_time"] == 4
        return ok and _fin(r) == {1: 1, 2: 5}, f"idle={idle[['start_time','end_time']].values.tolist()}"

    def t_late_start():
        r = simulate(_wl([(7, 5, 2, 20, 3)]), "EDF")
        t = r.tasks.iloc[0]
        return (t["start_time"] == 5 and t["completion_time"] == 7
                and r.trace.iloc[0]["start_time"] == 5), "first arrival at 5 -> no leading idle, runs 5-7"

    def t_cs():
        r = simulate(_wl([(1, 0, 2, 20, 3), (2, 0, 2, 20, 3)]), "FIFO", True, cs_cost=1)
        s = r.summary
        return (_fin(r) == {1: 2, 2: 5} and s["context_switches"] == 1 and s["context_switch_time"] == 1
                and s["total_simulation_time"] == 5), f"finish={_fin(r)}, initial dispatch free"

    def t_tie():
        r = simulate(_wl([(1, 0, 2, 10, 3), (2, 0, 2, 10, 1), (3, 0, 2, 10, 1)]), "EDF", False)
        return _fin(r) == {2: 2, 3: 4, 1: 6}, f"finish={_fin(r)} (priority, then arrival, then id)"

    def t_no_needless_preempt():
        r = simulate(_wl([(1, 0, 4, 10, 3), (2, 1, 2, 10, 3)]), "EDF", True)
        return r.summary["preemptions"] == 0 and _fin(r) == {1: 4, 2: 6}, f"finish={_fin(r)}"

    def t_mled_equals_edf():
        ok = True
        for seed in range(5):
            w = generate_workload(40, 1.0, 1.2, 2.5, seed)
            rng = np.random.default_rng(seed)
            w["pred_exec"] = w["execution_time"] * rng.uniform(0.6, 1.4, len(w))
            w["miss_risk"] = rng.uniform(0, 1, len(w))
            a = simulate(w, "EDF", True, 1)
            b = simulate(w, "ML-EDF", True, 1, risk_weight=0.0, ml_variant="risk_deadline")
            ok &= (a.tasks["completion_time"].to_numpy() == b.tasks["completion_time"].to_numpy()).all()
        return ok, "risk-adjusted-deadline ML-EDF with weight 0 reproduces EDF"

    def t_independence():
        w = generate_workload(40, 1.0, 1.2, 2.5, 3)
        w["pred_exec"], w["miss_risk"] = w["execution_time"] * 1.0, 0.0
        w2 = w.copy()
        w2["pred_exec"], w2["miss_risk"] = 99.0, 1.0
        same = all((simulate(w, p).tasks["completion_time"].to_numpy()
                    == simulate(w2, p).tasks["completion_time"].to_numpy()).all() for p in ("FIFO", "EDF"))
        return same, "changing predictions never changes FIFO or EDF"

    def t_determinism():
        w = generate_workload(60, 1.1, 1.2, 2.5, 9)
        rng = np.random.default_rng(0)
        w["pred_exec"], w["miss_risk"] = w["execution_time"] * rng.uniform(0.7, 1.3, len(w)), rng.uniform(0, 1, len(w))
        a, b = simulate(w, "ML-EDF", True, 1, 2.0), simulate(w, "ML-EDF", True, 1, 2.0)
        return a.trace.equals(b.trace), "same inputs -> identical trace"

    def t_invariants():
        failures = []
        for seed in range(4):
            w = generate_workload(40, [0.5, 1.0, 1.3, 0.8][seed], 1.2, 2.5, seed)
            rng = np.random.default_rng(seed)
            w["pred_exec"] = np.maximum(1, w["execution_time"] * rng.uniform(0.7, 1.3, len(w)))
            w["miss_risk"] = rng.uniform(0, 1, len(w))
            for p in SCHEDULERS:
                for pre in (True, False):
                    for cs in (0, 2):
                        for variant in ML_VARIANTS:
                            r = simulate(w, p, pre, cs, 2.0, variant)
                            bad = [c["check"] for c in validate_run(w, r, cs) if not c["passed"]]
                            if bad:
                                failures.append((seed, p, pre, cs, bad[0]))
        return not failures, "48+ runs across schedulers, modes, cs costs" if not failures else str(failures[:2])

    def t_catches_corruption():
        w = _wl([(1, 0, 3, 10, 3), (2, 1, 2, 4, 3)])
        r = simulate(w, "FIFO")
        r.trace.loc[1, "start_time"] = 1  # overlap the second event
        bad = [c for c in validate_run(w, r, 0) if not c["passed"]]
        return len(bad) > 0, f"{len(bad)} check(s) flagged a corrupted trace"

    def t_reproducible():
        a, b = generate_workload(30, 1.0, 1.5, 3.0, 5), generate_workload(30, 1.0, 1.5, 3.0, 5)
        c = generate_workload(30, 1.0, 1.5, 3.0, 6)
        return a.equals(b) and not a.equals(c), "same seed -> same workload"

    def t_csv():
        good = pd.DataFrame({"task_id": [5, 9], "arrival_time": [1, 2], "execution_time": [3, 2], "deadline": [10, 3]})
        ok_df, errs, warns = validate_workload(good)
        checks = [ok_df is not None and not errs and len(warns) >= 1]
        bad_cases = [
            good.assign(task_id=[5, 5]),
            good.assign(arrival_time=[-1, 2]),
            good.assign(execution_time=[0, 2]),
            good.assign(deadline=[1, 3]),
            good.assign(execution_time=[1.5, 2]),
            good.drop(columns=["deadline"]),
            good.assign(task_type=["CPU", "GPU"]),
        ]
        for b in bad_cases:
            _, e, _ = validate_workload(b)
            checks.append(len(e) > 0)
        _, e, w = validate_workload(good.assign(deadline=[3, 3]))  # infeasible but allowed
        checks.append(not e and any("cannot meet" in x for x in w))
        return all(checks), f"{sum(checks)}/{len(checks)} validation cases behaved correctly"

    def t_empty():
        r = simulate(_wl([(1, 0, 1, 5, 3)]).iloc[0:0], "EDF")
        return len(r.trace) == 0 and len(r.tasks) == 0, "empty workload does not crash"

    def t_sanitize():
        p, r, n = sanitize_predictions([np.nan, -2.0, 5.0], [0.5, 1.7, np.nan], 4.0)
        return (list(p) == [4.0, 4.0, 5.0] and list(r) == [0.5, 1.0, 0.0] and n >= 3), f"repaired={n}"

    for name, fn in [
        ("Min-Heap: ordering, duplicates, empty, single, heapify, clear", t_heap),
        ("FIFO order (hand-computed)", t_fifo),
        ("EDF preemptive (hand-computed, preemptions=1, switches=2)", t_edf_pre),
        ("EDF non-preemptive (hand-computed)", t_edf_nonpre),
        ("Idle interval recorded", t_idle),
        ("Late first arrival: no leading idle, no early execution", t_late_start),
        ("Context-switch cost, initial dispatch free", t_cs),
        ("Equal-deadline tie-breaking", t_tie),
        ("No needless preemption on equal keys", t_no_needless_preempt),
        ("ML-EDF (risk-deadline, weight 0) equals EDF", t_mled_equals_edf),
        ("FIFO and EDF independent of ML predictions", t_independence),
        ("Deterministic output", t_determinism),
        ("Invariants hold on random workloads (all modes)", t_invariants),
        ("Validator detects a corrupted trace", t_catches_corruption),
        ("Workload generator reproducible", t_reproducible),
        ("CSV validation rules", t_csv),
        ("Empty workload", t_empty),
        ("Invalid ML predictions are repaired", t_sanitize),
    ]:
        case(name, fn)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Interface: styling and small HTML builders
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#F5F7FB;--surface:#FFFFFF;--ink:#14213D;--muted:#5A6785;--line:#E2E8F2;
--good:#1F9D63;--bad:#E5484D;--fifo:#64748B;--edf:#2563EB;--ml:#7C3AED;color-scheme:light}
html,body,.stApp{background:var(--bg)!important;font-family:'Manrope',system-ui,-apple-system,'Segoe UI',sans-serif}
:where(.stApp) :where(p,li,label,h1,h2,h3,h4,span,div){color:var(--ink);font-family:'Manrope',system-ui,sans-serif}
:where(.stApp) :where(h1,h2,h3,h4){font-weight:800;letter-spacing:-0.02em}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
.block-container{padding-top:1.4rem;max-width:1180px}
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
section[data-testid="stSidebar"] .block-container{padding-top:1rem}
[data-testid="stSlider"],[data-testid="stToggle"],[data-testid="stCheckbox"],[data-testid="stRadio"]{filter:hue-rotate(222deg) saturate(.8)}
button[data-baseweb="tab"]{font-weight:700;color:var(--muted)}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--ink)}
div[data-baseweb="tab-highlight"]{background:var(--ink)!important}
.stButton>button,.stDownloadButton>button{border-radius:10px;border:1px solid var(--line);font-weight:700;background:#fff;color:var(--ink)}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--ink);color:var(--ink)}
.stButton>button[kind="primary"],button[data-testid="stBaseButton-primary"]{background:var(--ink);color:#fff;border-color:var(--ink)}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:#fff}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}

.hero{background:linear-gradient(120deg,#0E1A44 0%,#1B2A7A 100%);border-radius:20px;padding:30px 34px 26px;margin-bottom:18px}
.hero h1{color:#fff!important;font-size:2.15rem;line-height:1.15;margin:0 0 8px 0;max-width:720px}
.hero p{color:#C9D3F5!important;font-size:1.02rem;line-height:1.55;margin:0 0 16px 0;max-width:680px}
.pills{display:flex;flex-wrap:wrap;gap:8px}
.pill{background:rgba(255,255,255,.12);color:#fff!important;border-radius:999px;padding:5px 12px;font-size:.82rem;font-weight:600}
.pill.ok{background:#D6F5E6;color:#0B5D39!important}
.pill.bad{background:#FFE0E0;color:#8E1B1B!important}

.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:0 0 22px 0}
.step{background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 16px;display:flex;gap:12px;align-items:flex-start}
.step b{background:var(--ink);color:#fff!important;border-radius:50%;min-width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-size:.85rem}
.step span{font-size:.92rem;line-height:1.45;color:var(--ink)}

.sc{background:#fff;border:1px solid var(--line);border-top:5px solid var(--c);border-radius:16px;padding:18px 20px 16px}
.sc-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.sc-name{font-weight:800;font-size:1.15rem;color:var(--c)!important}
.badge{background:var(--c);color:#fff!important;font-size:.72rem;font-weight:700;padding:3px 9px;border-radius:999px}
.badge.tie{background:#E6EAF2;color:var(--muted)!important}
.sc-def{color:var(--muted)!important;font-size:.85rem;line-height:1.4;min-height:2.4em}
.sc-big{font-size:2.5rem;font-weight:800;letter-spacing:-0.03em;margin:10px 0 2px 0;line-height:1}
.sc-big small{font-size:.95rem;font-weight:600;color:var(--muted)!important;letter-spacing:0}
.sc-bar{height:8px;background:#EDF1F8;border-radius:99px;overflow:hidden;margin:10px 0 10px 0}
.sc-bar i{display:block;height:100%;background:var(--c);border-radius:99px}
.sc-foot{font-size:.86rem;color:var(--muted)!important}

.panel{background:#fff;border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin:6px 0 14px 0}
.panel h4{margin:0 0 4px 0;font-size:1.02rem}
.panel .sub{color:var(--muted)!important;font-size:.88rem;margin-bottom:12px}
.strip{display:flex;gap:14px;align-items:flex-start;margin:8px 0}
.strip-name{min-width:70px;font-weight:800;padding-top:1px}
.cells{display:flex;flex-wrap:wrap}
.cells i{display:inline-block;width:14px;height:14px;border-radius:4px;margin:0 3px 3px 0}
.cells i.ok{background:var(--good)}
.cells i.late{background:var(--bad);background-image:repeating-linear-gradient(135deg,rgba(255,255,255,.6) 0 2px,transparent 2px 5px)}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:.84rem;color:var(--muted);margin-top:10px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.insights{margin:0;padding-left:18px}
.insights li{margin:6px 0;line-height:1.5}
.brand{font-weight:800;font-size:1.25rem;letter-spacing:-0.02em;margin-bottom:2px}
.brand-sub{color:var(--muted)!important;font-size:.82rem;margin-bottom:14px}
.side-h{font-weight:800;font-size:.95rem;margin:16px 0 4px 0}
.callout{border-left:4px solid var(--ink);background:#fff;border-radius:0 12px 12px 0;padding:12px 16px;margin:8px 0 14px 0;font-size:.93rem;line-height:1.5}
.explain{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:6px 0 14px 0}
.explain div{background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 16px;font-size:.9rem;line-height:1.5}
.explain b{display:block;margin-bottom:4px;font-size:.98rem}
@media (max-width:760px){.steps,.explain{grid-template-columns:1fr}.hero h1{font-size:1.6rem}.hero{padding:22px}}
</style>
"""

FRIENDLY_ROWS = [
    ("on_time", "Tasks finished on time", "higher"),
    ("deadline_misses", "Tasks finished late", "lower"),
    ("avg_waiting_time", "Average waiting time", "lower"),
    ("avg_response_time", "Average time until first run", "lower"),
    ("avg_tardiness", "Average lateness", "lower"),
    ("max_tardiness", "Worst lateness", "lower"),
    ("preemptions", "Times a running task was interrupted", "none"),
    ("context_switches", "Switches between tasks", "lower"),
    ("cpu_utilization_pct", "Processor busy (%)", "none"),
    ("total_simulation_time", "Time when the last task finished", "lower"),
]


def pills_html(items: list[str], verified: bool) -> str:
    chips = "".join(f"<span class='pill'>{t}</span>" for t in items)
    state = ("<span class='pill ok'>All integrity checks passed</span>" if verified
             else "<span class='pill bad'>An integrity check failed, see Verify and learn</span>")
    return f"<div class='pills'>{chips}{state}</div>"


def hero_html(items: list[str], verified: bool) -> str:
    return (
        "<div class='hero'><h1>Which scheduler meets the most deadlines?</h1>"
        "<p>The same set of tasks runs through three schedulers on one processor. "
        "See who finishes on time, and what machine learning changes.</p>"
        + pills_html(items, verified) + "</div>"
    )


def steps_html() -> str:
    steps = [
        "Tasks arrive, each with the time it needs and a deadline.",
        "Each scheduler decides which task runs next. All three get the identical tasks.",
        "Count the tasks that finished before their deadline. Most on time wins.",
    ]
    return "<div class='steps'>" + "".join(f"<div class='step'><b>{i}</b><span>{s}</span></div>"
                                           for i, s in enumerate(steps, 1)) + "</div>"


def scorecard_html(name: str, summary: dict, n: int, best: set, n_active: int) -> str:
    late = summary["deadline_misses"]
    on = n - late
    pct = 100 * on / max(1, n)
    if len(best) == n_active:
        badge = "<span class='badge tie'>All tied</span>"
    elif name in best:
        badge = "<span class='badge'>Most on time</span>"
    else:
        badge = ""
    return (
        f"<div class='sc' style='--c:{COLORS[name]}'>"
        f"<div class='sc-head'><span class='sc-name'>{name}</span>{badge}</div>"
        f"<div class='sc-def'>{DEFS[name]}</div>"
        f"<div class='sc-big'>{on}<small> of {n} on time</small></div>"
        f"<div class='sc-bar'><i style='width:{pct:.1f}%'></i></div>"
        f"<div class='sc-foot'>{late} late. Average wait {summary['avg_waiting_time']:.1f} time units.</div></div>"
    )


def strip_html(res: SimResult, name: str) -> str:
    t = res.tasks.sort_values(["arrival_time", "task_id"])
    cells = []
    for tid, fin, dl, late in zip(t["task_id"], t["completion_time"], t["deadline"], t["deadline_missed"]):
        state = "late" if late else "on time"
        cells.append(f"<i class='{'late' if late else 'ok'}' title='Task {tid}: finished at {fin}, deadline {dl} ({state})'></i>")
    return (f"<div class='strip'><div class='strip-name' style='color:{COLORS[name]}'>{name}</div>"
            f"<div class='cells'>{''.join(cells)}</div></div>")


def build_insights(table: pd.DataFrame, results: dict, n: int) -> list[str]:
    out = []
    misses = table["deadline_misses"].astype(int)
    best_val = misses.min()
    best = [s for s in table.index if misses[s] == best_val]
    on = n - best_val
    if len(best) == len(table):
        out.append(f"All schedulers finished the same number of tasks on time: {on} of {n}.")
    else:
        out.append(f"<b>{' and '.join(best)}</b> met the most deadlines: {on} of {n} tasks on time ({100 * on / n:.0f}%).")
    if "FIFO" in table.index and "EDF" in table.index:
        d = int(misses["FIFO"] - misses["EDF"])
        if d > 0:
            out.append(f"EDF finished {d} more tasks on time than FIFO, because it looks at deadlines and FIFO does not.")
        elif d < 0:
            out.append(f"FIFO finished {-d} more tasks on time than EDF on this workload.")
        else:
            out.append("FIFO and EDF finished the same number of tasks on time on this workload.")
    if "ML-EDF" in table.index and "EDF" in table.index:
        d = int(misses["EDF"] - misses["ML-EDF"])
        if d > 0:
            out.append(f"ML-EDF beat EDF here by {d} task{'s' if d != 1 else ''}. Check the Is it luck? tab to see whether that repeats.")
        elif d < 0:
            out.append(f"ML-EDF finished {-d} fewer task{'s' if d != -1 else ''} on time than EDF on this workload.")
        else:
            out.append("ML-EDF and EDF finished the same number of tasks on time.")
        pe, pm = int(table.loc["EDF", "preemptions"]), int(table.loc["ML-EDF", "preemptions"])
        if pm >= 10 and pm > 2 * pe:
            out.append(f"ML-EDF interrupted running tasks {pm} times against {pe} for EDF. "
                       "Every switch can cost time, which is how a clever idea can lose to a simple one.")
    out.append("This is one set of tasks. Open <b>Is it luck?</b> to repeat the test on many.")
    return out


def friendly_table(table: pd.DataFrame, n: int) -> pd.DataFrame:
    t = table.copy()
    t["on_time"] = n - t["deadline_misses"].astype(int)
    return pd.DataFrame({label: t[key] for key, label, _ in FRIENDLY_ROWS}).T


def styled_friendly(df: pd.DataFrame):
    directions = {label: d for _, label, d in FRIENDLY_ROWS}

    def hl(row):
        d = directions[row.name]
        if d == "none":
            return [""] * len(row)
        best = row.max() if d == "higher" else row.min()
        return ["background-color:#DDF5E8;font-weight:700" if np.isclose(v, best) else "" for v in row]

    def fmt(v):
        v = float(v)
        return f"{v:,.0f}" if v.is_integer() else f"{v:,.2f}"

    try:
        return df.style.apply(hl, axis=1).format(fmt)
    except Exception:
        return df


def panel_open(title: str, sub: str = "") -> None:
    st.markdown(f"<div class='panel'><h4>{title}</h4><div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Scheduler Lab: which scheduler meets the most deadlines?",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
st.session_state.setdefault("seed", 42)


def _new_seed():
    st.session_state["seed"] = int(np.random.default_rng().integers(0, 1_000_000))


# ---- Sidebar --------------------------------------------------------------- #
with st.sidebar:
    st.markdown("<div class='brand'>Scheduler Lab</div>"
                "<div class='brand-sub'>Compare how tasks get scheduled</div>", unsafe_allow_html=True)

    st.markdown("<div class='side-h'>1. Pick a scenario</div>", unsafe_allow_html=True)
    scenario = st.radio("Scenario", list(SCENARIOS), index=1, label_visibility="collapsed")
    st.caption(SCENARIOS[scenario][1])
    n_tasks = st.slider("Number of tasks", 10, 300, 60, step=10)
    st.button("Shuffle to a new set of tasks", on_click=_new_seed)

    st.markdown("<div class='side-h'>2. Set the rules</div>", unsafe_allow_html=True)
    preemptive = st.toggle("A more urgent task may interrupt the running one", value=True)

    with st.expander("Advanced settings"):
        source = st.radio("Where do tasks come from?", ["Generate", "Upload CSV"], horizontal=True)
        upload = None
        custom = False
        load, smin, smax = DIFFICULTY[SCENARIOS[scenario][0]]
        if source == "Generate":
            custom = st.checkbox("Set load and deadlines by hand")
            if custom:
                load = st.slider("How busy the processor is", 0.3, 1.8, float(load), step=0.05,
                                 help="1.0 means work arrives about as fast as it can be done.")
                smin, smax = st.slider("Deadline room (min, max)", 1.0, 5.0, (float(smin), float(smax)), step=0.1,
                                       help="deadline = arrival + run time x this number")
            st.number_input("Random seed", 0, 1_000_000, key="seed",
                            help="The same seed always gives the same tasks.")
        else:
            upload = st.file_uploader("CSV file", type="csv")
            st.caption("Needs: task_id, arrival_time, execution_time, deadline. "
                       "Optional: priority, task_type, input_size. Whole numbers only.")
        cs_cost = int(st.number_input("Cost of switching tasks (time units)", 0, 10, 0,
                                      help="Time lost every time the processor changes task."))
        ml_variant = st.selectbox("How ML-EDF ranks tasks", list(ML_VARIANTS), format_func=lambda k: ML_VARIANTS[k])
        risk_weight = st.slider("How much ML-EDF trusts its lateness risk", 0.0, 10.0, 2.0, step=0.5,
                                help="0 ignores the risk estimate completely.")
        st.caption("Machine-learning training")
        n_workloads = st.slider("Practice workloads", 10, 80, 30, step=5)
        tasks_per = st.slider("Tasks per practice workload", 50, 300, 150, step=25)
        n_estimators = st.slider("Trees per model", 20, 300, 100, step=20)

# ---- Workload -------------------------------------------------------------- #
seed = int(st.session_state["seed"])
if source == "Generate":
    workload = generate_workload(n_tasks, load, smin, smax, seed)
    workload_name = f"seed {seed}"
else:
    if upload is None:
        st.markdown(hero_html(["Waiting for your CSV file"], True), unsafe_allow_html=True)
        st.info("Upload a CSV file in Advanced settings, or choose Generate to use random tasks.")
        st.stop()
    try:
        raw = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"The CSV file could not be read: {exc}")
        st.stop()
    workload, errs, warns = validate_workload(raw)
    for w_ in warns:
        st.warning(w_)
    if errs:
        st.error("The uploaded tasks have problems. Fix these and upload again:\n\n" + "\n".join(f"- {e}" for e in errs))
        st.stop()
    workload_name = "your CSV"

# ---- Models, predictions, simulations -------------------------------------- #
models = None
try:
    models = train_models(n_workloads, tasks_per, n_estimators, seed, preemptive, cs_cost)
except Exception as exc:
    st.error(f"The machine-learning models could not be trained ({exc}). FIFO and EDF still run; ML-EDF is switched off.")

active = SCHEDULERS if models else ["FIFO", "EDF"]
repaired = 0
if models:
    try:
        work, repaired = add_predictions(workload, models)
    except Exception as exc:
        st.error(f"Predictions failed ({exc}). ML-EDF is switched off for this workload.")
        models, active, work = None, ["FIFO", "EDF"], workload
else:
    work = workload

results = run_schedulers(work, active, preemptive, cs_cost, risk_weight, ml_variant)
table = summary_table(results)
checks = {s: validate_run(workload, results[s], cs_cost) for s in active}
all_ok = all(c["passed"] for s in active for c in checks[s])
fingerprints = {r.fingerprint for r in results.values()}
same_workload = len(fingerprints) == 1
n = len(workload)
verified = all_ok and same_workload

mode = "Interruptions allowed" if preemptive else "No interruptions"
scenario_label = scenario if source == "Generate" else "Uploaded tasks"
st.markdown(hero_html([f"{scenario_label} scenario", f"{n} tasks", mode,
                       f"Task set {next(iter(fingerprints))}"], verified), unsafe_allow_html=True)
st.markdown(steps_html(), unsafe_allow_html=True)
if repaired:
    st.warning(f"{repaired} unusable machine-learning prediction(s) were replaced with safe defaults.")

tab_res, tab_time, tab_tasks, tab_ml, tab_exp, tab_learn = st.tabs(
    ["Results", "Timeline", "Tasks", "Machine learning", "Is it luck?", "Verify and learn"]
)

# ---- Results --------------------------------------------------------------- #
with tab_res:
    misses = table["deadline_misses"].astype(int)
    best = {s for s in table.index if misses[s] == misses.min()}
    cols = st.columns(len(active))
    for col, s in zip(cols, active):
        col.markdown(scorecard_html(s, results[s].summary, n, best, len(active)), unsafe_allow_html=True)

    strips = "".join(strip_html(results[s], s) for s in active)
    st.markdown(
        "<div class='panel'><h4>Every task, one square each</h4>"
        "<div class='sub'>Squares run in order of arrival. Hover a square for its details.</div>"
        + strips +
        "<div class='legend'><span><i style='background:#1F9D63'></i>Finished on time</span>"
        "<span><i style='background:#E5484D;background-image:repeating-linear-gradient(135deg,rgba(255,255,255,.6) 0 2px,transparent 2px 5px)'></i>Finished late</span></div></div>",
        unsafe_allow_html=True)

    items = "".join(f"<li>{t}</li>" for t in build_insights(table, results, n))
    st.markdown(f"<div class='panel'><h4>What happened</h4><ul class='insights'>{items}</ul></div>",
                unsafe_allow_html=True)

    st.markdown("#### Side by side")
    st.dataframe(styled_friendly(friendly_table(table, n)), height=390)
    st.caption("Green cells mark the best value in each row. Rows without colour have no better or worse direction. "
               "No single number decides which scheduler is best.")

    with st.expander("See the numbers as charts"):
        chart_keys = [("deadline_misses", "Tasks finished late"), ("avg_waiting_time", "Average waiting time"),
                      ("avg_response_time", "Average time until first run"), ("context_switches", "Switches between tasks"),
                      ("preemptions", "Interruptions"), ("cpu_utilization_pct", "Processor busy (%)")]
        gcols = st.columns(3)
        for i, (k, label) in enumerate(chart_keys):
            fig = px.bar(table.reset_index(), x="scheduler", y=k, color="scheduler", color_discrete_map=COLORS, title=label)
            style_fig(fig, 280).update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
            gcols[i % 3].plotly_chart(fig, key=f"cmp_{k}")
    with st.expander("All technical metrics"):
        st.dataframe(table.rename(columns=LABELS).T)

# ---- Timeline -------------------------------------------------------------- #
with tab_time:
    st.markdown("<div class='callout'>Each bar is a task using the processor. "
                "<b>Green</b> tasks finish on time, <b>red</b> tasks finish late, grey is idle time, "
                "orange is time lost switching, and a diamond marks a deadline. "
                "If a task appears in several pieces, it was interrupted.</div>", unsafe_allow_html=True)
    t_min = int(workload["arrival_time"].min())
    t_max = int(max(r.summary["total_simulation_time"] for r in results.values()))
    t_lo, t_hi = st.slider("Zoom to a time window", t_min, t_max, (t_min, min(t_max, t_min + 100)))
    for s in active:
        st.plotly_chart(gantt_figure(results[s], f"{s}: last task finishes at time {results[s].summary['total_simulation_time']}",
                                     t_lo, t_hi, COLORS[s]), key=f"gantt_{s}")
        order = results[s].trace[results[s].trace["event_type"] == "RUN"].drop_duplicates("task_id")["task_id"].tolist()
        with st.expander(f"{s}: the order tasks first ran in"):
            st.code(" -> ".join(str(x) for x in order))

# ---- Tasks ----------------------------------------------------------------- #
with tab_tasks:
    st.markdown("#### The tasks every scheduler received")
    st.caption(f"Task set {next(iter(fingerprints))}. This code is a fingerprint of the table below, "
               "and every result stores the fingerprint of the tasks it was given.")
    shown = workload.rename(columns={
        "task_id": "Task", "arrival_time": "Arrives at", "execution_time": "Needs (time units)",
        "deadline": "Deadline", "priority": "Priority", "task_type": "Type", "input_size": "Input size"})
    st.dataframe(shown, hide_index=True, height=300)
    st.download_button("Download these tasks as CSV", workload.to_csv(index=False), "tasks.csv", "text/csv")

    st.markdown("#### What each scheduler did with them")
    c1, c2 = st.columns([1, 1])
    sched = c1.radio("Scheduler", active, horizontal=True, key="task_sched")
    only_late = c2.checkbox("Show only late tasks")
    d = results[sched].tasks.copy()
    d["late_by"] = d["tardiness"]
    d["result"] = np.where(d["deadline_missed"], "Late", "On time")
    cols_ = {"task_id": "Task", "arrival_time": "Arrives", "execution_time": "Needed", "deadline": "Deadline",
             "start_time": "Started", "completion_time": "Finished", "waiting_time": "Waited",
             "late_by": "Late by", "result": "Result"}
    if "pred_exec" in d:
        d["pred_exec"] = d["pred_exec"].round(1)
        d["risk_pct"] = (d["miss_risk"] * 100).round(0)
        cols_.update({"pred_exec": "AI guess of run time", "risk_pct": "AI lateness risk (%)"})
    view = d[list(cols_)].rename(columns=cols_)
    if only_late:
        view = view[view["Result"] == "Late"]
    st.dataframe(view, hide_index=True, height=360)
    st.download_button("Download these results as CSV", view.to_csv(index=False), f"{sched}_results.csv", "text/csv")
    with st.expander("Raw execution trace"):
        st.dataframe(results[sched].trace, hide_index=True)

# ---- Machine learning ------------------------------------------------------ #
with tab_ml:
    if models is None:
        st.warning("The machine-learning models are unavailable, so ML-EDF is switched off.")
    else:
        st.markdown(
            "<div class='explain'>"
            "<div><b>1. Learn from practice</b>Two models study many made-up workloads. One learns how long a task really runs. "
            "The other learns which tasks end up late.</div>"
            "<div><b>2. Guess for each task</b>For your tasks, the models guess the run time and the chance of being late. "
            "The scheduler never sees the true run time.</div>"
            "<div><b>3. Rank the tasks</b>ML-EDF mixes the guesses with each deadline to pick the next task. "
            "FIFO and EDF ignore the guesses completely.</div></div>", unsafe_allow_html=True)

        m, cm_ = models["reg_metrics"], models["clf_metrics"]
        typical = float(workload["execution_time"].mean())
        a, b = st.columns(2)
        with a:
            st.markdown("#### Guessing how long a task runs")
            st.markdown(f"<div class='callout'>On practice tasks the models had never seen, the guess was off by "
                        f"<b>{m['MAE']:.1f} time units</b> on average. A typical task here needs about {typical:.0f}, "
                        f"so that is roughly {100 * m['MAE'] / max(typical, 1):.0f}% of a task. R2 = {m['R2']:.2f} (1.0 is perfect).</div>",
                        unsafe_allow_html=True)
            pa = prediction_analysis(work, results, risk_weight, ml_variant)
            fig = px.scatter(pa, x="execution_time", y="pred_exec", opacity=0.65,
                             labels={"execution_time": "Real run time", "pred_exec": "Guessed run time"},
                             title="Your tasks: guess against reality")
            fig.update_traces(marker=dict(color=COLORS["ML-EDF"], size=8))
            top = float(pa["execution_time"].max())
            fig.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(dash="dash", color="#94A3B8"))
            st.plotly_chart(style_fig(fig, 320), key="ml_cur_scatter")
            st.caption("Dots on the dashed line are perfect guesses.")
        with b:
            st.markdown("#### Guessing which tasks end up late")
            if models["clf"] is None:
                st.warning("The practice workloads never produced both outcomes, so no lateness model was trained "
                           "and ML-EDF treats every task as equally safe.")
            else:
                auc = cm_.get("ROC-AUC")
                if auc is not None:
                    st.markdown(f"<div class='callout'>Shown a late task and an on-time task, the model rated the late "
                                f"one as riskier <b>{auc * 100:.0f}% of the time</b> (50% would be a coin flip). "
                                "That is useful but far from certain, so a risk score is a hint, not a promise.</div>",
                                unsafe_allow_html=True)
                if "missed_under_EDF" in pa:
                    fig = px.box(pa.assign(outcome=np.where(pa["missed_under_EDF"], "Was late under EDF", "On time under EDF")),
                                 x="outcome", y="miss_risk", points="all", color="outcome",
                                 color_discrete_map={"Was late under EDF": BAD, "On time under EDF": GOOD},
                                 labels={"miss_risk": "Predicted lateness risk", "outcome": ""},
                                 title="Predicted risk against what happened under EDF")
                    st.plotly_chart(style_fig(fig, 320).update_layout(showlegend=False), key="ml_cur_box")
                    st.caption("Good risk scores put the red box above the green box.")

        with st.expander("Technical scores and confusion matrix"):
            x1, x2, x3 = st.columns(3)
            x1.metric("MAE", f"{m['MAE']:.2f}")
            x2.metric("RMSE", f"{m['RMSE']:.2f}")
            x3.metric("R2", f"{m['R2']:.3f}")
            if models["clf"] is not None:
                mc = st.columns(4)
                for i, (k, v) in enumerate(cm_.items()):
                    mc[i % 4].metric(k, f"{v:.3f}")
                cmx = models["confusion"]
                st.dataframe(pd.DataFrame(cmx, index=["Actually on time", "Actually late"],
                                          columns=["Predicted on time", "Predicted late"]))
                st.caption(f"Share of practice tasks that ended late under EDF: train {models['train_miss_rate']:.0%}, "
                           f"test {models['test_miss_rate']:.0%}. The labels come from simulating EDF, "
                           "so the lateness model describes risk under EDF, not under ML-EDF. "
                           "Everything here uses made-up tasks, so the scores say how well the models learned this "
                           "generator, not how well they would do on real programs.")
            st.dataframe(pa.round(3), hide_index=True)
            report = {"regression": models["reg_metrics"], "classifier": models["clf_metrics"],
                      "confusion_matrix": None if models["confusion"] is None else models["confusion"].tolist(),
                      "train_miss_rate": models["train_miss_rate"], "test_miss_rate": models["test_miss_rate"],
                      "config": models["config"]}
            st.download_button("Download ML report (JSON)", json.dumps(report, indent=2),
                               "ml_evaluation.json", "application/json")

# ---- Experiments ----------------------------------------------------------- #
with tab_exp:
    st.markdown("<div class='callout'>One set of tasks can flatter any scheduler. Here the same test runs on many "
                "different sets, at several difficulty levels, so the result does not depend on luck.</div>",
                unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    sizes = e1.multiselect("Number of tasks", [30, 60, 120, 200], default=[60])
    levels = e2.multiselect("Scenarios", list(SCENARIOS), default=list(SCENARIOS))
    n_seeds = e3.slider("Task sets per setting", 3, 30, 10)
    st.caption("Uses the rules from the sidebar: interruptions, switching cost and ML-EDF settings.")
    if st.button("Run the experiment", type="primary"):
        if not sizes or not levels:
            st.warning("Pick at least one number of tasks and one scenario.")
        else:
            bar = st.progress(0.0)
            raw_exp = run_experiment(sizes, [SCENARIOS[x][0] for x in levels], n_seeds, seed, models, preemptive,
                                     cs_cost, risk_weight, ml_variant, bar)
            bar.empty()
            raw_exp["scenario"] = raw_exp["difficulty"].map(SCENARIO_OF)
            st.session_state["experiment"] = raw_exp
            st.session_state["experiment_cfg"] = {"interruptions": preemptive, "switch_cost": cs_cost,
                                                  "ml_formula": ml_variant, "risk_weight": risk_weight}
    exp = st.session_state.get("experiment")
    if exp is None:
        st.info("Choose your settings and run the experiment. It takes a few seconds.")
    else:
        if not exp["integrity_ok"].all():
            st.error(f"{int((~exp['integrity_ok']).sum())} simulation(s) failed integrity checks.")
        if {"EDF", "ML-EDF"}.issubset(set(exp["scheduler"])):
            piv = exp.pivot_table(index=["tasks", "scenario", "seed"], columns="scheduler",
                                  values="deadline_misses").reset_index()
            fewer, same, more = int((piv["ML-EDF"] < piv["EDF"]).sum()), int((piv["ML-EDF"] == piv["EDF"]).sum()), int((piv["ML-EDF"] > piv["EDF"]).sum())
            st.markdown(f"<div class='panel'><h4>ML-EDF against EDF, task set by task set</h4>"
                        f"<div class='sub'>{len(piv)} task sets compared</div>"
                        f"<ul class='insights'><li>ML-EDF finished <b>more</b> tasks on time in <b>{fewer}</b></li>"
                        f"<li>They finished the <b>same</b> number in <b>{same}</b></li>"
                        f"<li>ML-EDF finished <b>fewer</b> tasks on time in <b>{more}</b></li></ul></div>",
                        unsafe_allow_html=True)
        metric_choice = st.radio("Show", ["Late tasks", "Average waiting time", "Interruptions"], horizontal=True)
        col = {"Late tasks": "deadline_misses", "Average waiting time": "avg_waiting_time",
               "Interruptions": "preemptions"}[metric_choice]
        agg = exp.groupby(["tasks", "scenario", "scheduler"])[col].agg(["mean", "std"]).reset_index()
        size_pick = st.selectbox("Number of tasks", sorted(exp["tasks"].unique()))
        sub = agg[agg["tasks"] == size_pick]
        fig = px.bar(sub, x="scenario", y="mean", error_y="std", color="scheduler", barmode="group",
                     color_discrete_map=COLORS, category_orders={"scenario": list(SCENARIOS), "scheduler": SCHEDULERS},
                     labels={"mean": metric_choice, "scenario": ""},
                     title=f"{metric_choice}, average over task sets (lower is better; whiskers show spread)")
        st.plotly_chart(style_fig(fig, 380), key="exp_chart")
        with st.expander("Full numbers and downloads"):
            keys = ["deadline_misses", "avg_waiting_time", "avg_turnaround_time", "avg_response_time",
                    "cpu_utilization_pct", "context_switches", "preemptions"]
            full = exp.groupby(["tasks", "scenario", "scheduler"])[keys].agg(["mean", "std"]).round(3)
            full.columns = [f"{a}_{b}" for a, b in full.columns]
            full = full.reset_index()
            st.dataframe(full, hide_index=True)
            d1, d2, d3 = st.columns(3)
            d1.download_button("Summary CSV", full.to_csv(index=False), "scheduler_comparison.csv", "text/csv")
            d2.download_button("Raw CSV", exp.to_csv(index=False), "experiment_raw.csv", "text/csv")
            d3.download_button("Raw JSON", exp.to_json(orient="records", indent=2), "experiment_raw.json", "application/json")
            st.caption(f"Settings used: {st.session_state.get('experiment_cfg')}")

# ---- Verify and learn ------------------------------------------------------ #
with tab_learn:
    st.markdown("#### How to read this app")
    st.markdown(
        "<div class='explain'>"
        "<div><b>FIFO</b>Runs tasks in the order they arrive, like a queue at a counter. Simple and fair, "
        "but an urgent task can wait behind a relaxed one.</div>"
        "<div><b>EDF</b>Earliest Deadline First. Whenever the processor is free it picks the task whose deadline is "
        "closest. A strong classic for real-time systems.</div>"
        "<div><b>ML-EDF</b>This project's idea. It adds machine-learning guesses of run time and lateness risk to the "
        "deadline. Whether that helps is exactly what the app tests.</div></div>", unsafe_allow_html=True)
    with st.expander("Words used in this app"):
        st.markdown(
            "- **Deadline**: the time by which a task must finish. Finishing exactly on it counts as on time.\n"
            "- **Waiting time**: time from arrival to finish, minus the time the task actually needed.\n"
            "- **Interruption (preemption)**: a running task is paused because a more urgent one arrived.\n"
            "- **Switch cost**: time the processor loses each time it changes task. Set in Advanced settings.\n"
            "- **Idle**: nothing was ready, so the processor waited for the next arrival.\n"
            "- **Lateness**: how long after its deadline a task finished.")
    with st.expander("How ML-EDF ranks tasks"):
        st.markdown(
            "Smaller score runs first. Ties fall back to the EDF order.\n\n"
            "- **Original formula**: `(deadline - now - guessed remaining time) - w x risk`\n"
            "- **Risk-adjusted deadline**: `deadline - w x risk x guessed run time`, which equals EDF when w is 0.\n\n"
            "`w` is the slider named *How much ML-EDF trusts its lateness risk*. "
            "The original formula behaves like Least Laxity First: the running task's score gets worse while it runs, "
            "so waiting tasks keep overtaking it and interruptions can multiply. "
            "The second formula keeps a task's score fixed while it runs.")
    with st.expander("How the simulator counts things"):
        st.markdown(
            "- Time moves in whole steps on a virtual clock, on one processor.\n"
            "- Arrivals are admitted before any decision at the same time.\n"
            "- A switch happens when the processor runs a different task from the last one. "
            "The first task and the first task after idle time are free.\n"
            "- The real run time is always what a task uses. Guesses only affect ML-EDF's ranking.\n"
            "- Processor busy (%) = running time divided by (end time minus first arrival).")
    st.markdown("<div class='callout'><b>What this is not.</b> It is a simulator inspired by real-time operating systems. "
                "No FreeRTOS or Zephyr is involved, no hardware is timed, IO tasks do not wait on devices and periodic "
                "tasks do not repeat. A hardware demonstration would be a separate project phase.</div>",
                unsafe_allow_html=True)

    st.markdown("#### Can the numbers be trusted?")
    st.caption("After every run the app re-checks each schedule using only its recorded trace.")
    for s in active:
        ck = pd.DataFrame(checks[s])
        ok = ck["passed"].all()
        with st.expander(f"{s}: {'all ' + str(len(ck)) + ' checks passed' if ok else 'some checks FAILED'}", expanded=not ok):
            ck["passed"] = ck["passed"].map({True: "pass", False: "FAIL"})
            st.dataframe(ck, hide_index=True)
    st.markdown("**Built-in self-tests.** Small hand-worked scenarios for the heap, schedulers, switching cost, idle time, "
                "CSV validation and prediction repair.")
    if st.button("Run self-tests"):
        st.session_state["self_tests"] = run_self_tests()
    tests = st.session_state.get("self_tests")
    if tests is not None:
        n_pass = int(tests["passed"].sum())
        (st.success if n_pass == len(tests) else st.error)(f"{n_pass} of {len(tests)} self-tests passed.")
        view = tests.copy()
        view["passed"] = view["passed"].map({True: "pass", False: "FAIL"})
        st.dataframe(view, hide_index=True)
