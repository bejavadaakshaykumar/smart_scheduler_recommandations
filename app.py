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
COLORS = {"FIFO": "#8d99ae", "EDF": "#2b6cb0", "ML-EDF": "#d9480f"}

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
def gantt_figure(res: SimResult, title: str, t_lo: int, t_hi: int) -> go.Figure:
    tr = res.trace
    tr = tr[(tr["end_time"] > t_lo) & (tr["start_time"] < t_hi)]
    tk = res.tasks.set_index("task_id")
    run = tr[tr["event_type"] == "RUN"]
    idle = tr[tr["event_type"] == "IDLE"]
    csw = tr[tr["event_type"] == "CONTEXT_SWITCH"]
    run_labels = "Task " + run["task_id"].astype(str)
    missed = run["task_id"].map(tk["deadline_missed"]).astype(bool)

    fig = go.Figure()

    def bars(part, labels, name, color):
        if len(part) == 0:
            return
        fig.add_trace(go.Bar(
            x=part["end_time"] - part["start_time"], base=part["start_time"], y=labels,
            orientation="h", name=name, marker_color=color,
            customdata=np.c_[part["start_time"], part["end_time"]],
            hovertemplate="%{y}<br>%{customdata[0]} to %{customdata[1]}<extra></extra>",
        ))

    bars(run[~missed], run_labels[~missed], "Run (on time)", "#2f9e44")
    bars(run[missed], run_labels[missed], "Run (task misses deadline)", "#e03131")
    bars(idle, pd.Series("IDLE", index=idle.index), "Idle", "#adb5bd")
    bars(csw, pd.Series("CONTEXT SWITCH", index=csw.index), "Context switch", "#f08c00")

    rows = (["IDLE"] if len(idle) else []) + (["CONTEXT SWITCH"] if len(csw) else []) + list(dict.fromkeys(run_labels))
    ids = run["task_id"].drop_duplicates()
    dls = ids.map(tk["deadline"])
    keep = (dls >= t_lo) & (dls <= t_hi)
    if keep.any():
        fig.add_trace(go.Scatter(
            x=dls[keep], y="Task " + ids[keep].astype(str), mode="markers", name="Deadline",
            marker=dict(symbol="diamond", size=8, color="#1c1c1c"),
            hovertemplate="%{y}<br>deadline %{x}<extra></extra>",
        ))
    fig.update_layout(
        title=title, barmode="overlay", xaxis_title="Virtual time", yaxis_title="Task",
        xaxis=dict(range=[t_lo, t_hi]),
        yaxis=dict(type="category", categoryorder="array", categoryarray=rows, autorange="reversed"),
        height=int(min(900, max(320, 20 * len(rows) + 150))),
        margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15),
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
# Page --------------------------------------------------------------------- #
st.set_page_config(page_title="Real-Time Task Scheduler", layout="wide")
st.session_state.setdefault("seed", 42)

def _new_seed():
    st.session_state["seed"] = int(np.random.default_rng().integers(0, 1_000_000))


# Quiet, consistent styling that keeps the data and controls easy to scan.
st.markdown(
    """
    <style>
    .stApp { background: #f6f7f8; color: #24323a; }
    [data-testid="stSidebar"] { background: #eef2f3; border-right: 1px solid #dce3e5; }
    .block-container { max-width: 1480px; padding-top: 1.6rem; padding-bottom: 2.5rem; }
    h1, h2, h3 { color: #193b46; letter-spacing: -0.02em; }
    h1 { font-weight: 650; }
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e0e6e8; border-radius: 10px;
        padding: 0.8rem 0.95rem; box-shadow: 0 1px 2px rgba(20, 45, 54, 0.04);
    }
    [data-testid="stMetricLabel"] { color: #586a70; }
    [data-testid="stTabs"] button { color: #39555d; }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #176b61; border-bottom-color: #176b61;
    }
    div[data-testid="stAlert"] { border-radius: 9px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="color:#24766d;font-size:.76rem;font-weight:700;letter-spacing:.11em;'
    'text-transform:uppercase;margin-bottom:.25rem">SCHEDULER LAB · SIMULATION WORKSPACE</div>',
    unsafe_allow_html=True,
)
st.title("Real-Time Task Scheduler")
st.caption(
    "See how different rules decide which task runs next. Create a sample workload or upload your own, "
    "then compare FIFO, EDF and ML-EDF on the same tasks."
)
st.info(
    "This is a virtual single-CPU simulation. It helps explain scheduling behavior; it does not control "
    "a real computer or RTOS."
)

# ---- Setup ----------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Set up a run")
    st.caption("Choose the tasks and scheduling conditions for this comparison.")

    st.markdown("**1 · Workload**")
    source = st.radio(
        "Choose task data",
        ["Generate", "Upload CSV"],
        horizontal=True,
        format_func=lambda value: "Create sample" if value == "Generate" else "Upload CSV",
    )
    upload = None
    n_tasks, load, smin, smax, difficulty = 60, 0.85, 1.5, 3.0, "Moderate"
    if source == "Generate":
        n_tasks = st.slider("Tasks in this run", 10, 300, 60, step=10)
        difficulty = st.selectbox(
            "Workload intensity",
            list(DIFFICULTY) + ["Custom"],
            index=1,
            help="Higher offered load means tasks arrive more quickly relative to the CPU's capacity.",
        )
        if difficulty == "Custom":
            load = st.slider(
                "Offered load",
                0.3,
                1.8,
                0.85,
                step=0.05,
                help="At 1.0, tasks arrive about as quickly as the CPU can process them.",
            )
            smin, smax = st.slider(
                "Deadline range",
                1.0,
                5.0,
                (1.5, 3.0),
                step=0.1,
                help="The deadline is set from arrival time plus execution time multiplied by this factor.",
            )
        else:
            load, smin, smax = DIFFICULTY[difficulty]
            st.caption(f"Offered load {load}; deadline factor {smin}–{smax}.")
        st.number_input("Random seed", 0, 1_000_000, key="seed")
        st.button("Create a new random sample", on_click=_new_seed)
    else:
        upload = st.file_uploader("Choose a workload CSV", type="csv")
        st.caption(
            "Required columns: task_id, arrival_time, execution_time and deadline. "
            "Priority, task_type and input_size are optional."
        )

    st.markdown("---")
    st.markdown("**2 · Scheduling rules**")
    preemptive = st.toggle(
        "Allow task preemption",
        value=True,
        help="When enabled, a newly arrived task can pause the current task if its scheduling priority is higher.",
    )
    cs_cost = int(
        st.number_input(
            "Context-switch time",
            0,
            10,
            0,
            help="Extra virtual time used when the CPU changes from one task to another.",
        )
    )
    ml_variant = st.selectbox(
        "ML-EDF decision rule",
        list(ML_VARIANTS),
        format_func=lambda key: ML_VARIANTS[key],
        help="Choose how ML-EDF combines predicted task length and deadline-miss risk.",
    )
    risk_weight = st.slider(
        "Miss-risk weight",
        0.0,
        10.0,
        2.0,
        step=0.5,
        help="How strongly ML-EDF uses its predicted chance of missing a deadline. Zero removes this risk term.",
    )

    with st.expander("Advanced · model training"):
        st.caption("These settings control the synthetic examples used to train and evaluate the ML models.")
        n_workloads = st.slider("Training workloads", 10, 80, 30, step=5)
        tasks_per = st.slider("Tasks per training workload", 50, 300, 150, step=25)
        n_estimators = st.slider("Trees in each model", 20, 300, 100, step=20)

# ---- Workload -------------------------------------------------------------- #
seed = int(st.session_state["seed"])
if source == "Generate":
    workload = generate_workload(n_tasks, load, smin, smax, seed)
    workload_name = f"generated (seed {seed})"
else:
    if upload is None:
        st.info("Choose a CSV file in the sidebar, or switch to Create sample.")
        st.stop()
    try:
        raw = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()
    workload, errs, warns = validate_workload(raw)
    for w_ in warns:
        st.warning(w_)
    if errs:
        st.error("The uploaded workload needs attention:\n\n" + "\n".join(f"- {e}" for e in errs))
        st.stop()
    workload_name = "uploaded CSV"

# ---- Models, predictions and simulations ---------------------------------- #
models = None
try:
    models = train_models(n_workloads, tasks_per, n_estimators, seed, preemptive, cs_cost)
except Exception as exc:
    st.error(f"ML models could not be trained ({exc}). FIFO and EDF can still run; ML-EDF is unavailable.")

active = SCHEDULERS if models else ["FIFO", "EDF"]
repaired = 0
if models:
    try:
        work, repaired = add_predictions(workload, models)
    except Exception as exc:
        st.error(f"Predictions failed ({exc}). ML-EDF is unavailable for this workload.")
        models, active, work = None, ["FIFO", "EDF"], workload
else:
    work = workload

results = run_schedulers(work, active, preemptive, cs_cost, risk_weight, ml_variant)
table = summary_table(results)
checks = {s: validate_run(workload, results[s], cs_cost) for s in active}
all_ok = all(c["passed"] for s in active for c in checks[s])

fingerprints = {r.fingerprint for r in results.values()}
same_workload = len(fingerprints) == 1

st.subheader("Your current run")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Tasks", len(workload))
m2.metric("Scheduling mode", "Preemptive" if preemptive else "Non-preemptive")
m3.metric("Switch time", f"{cs_cost} time units")
m4.metric("Workload ID", next(iter(fingerprints)))
m5.metric("Task data", workload_name)
if all_ok and same_workload:
    st.success("Ready to explore: every integrity check passed, and each scheduler received the same tasks.")
else:
    st.error("A simulation integrity check failed. Open Integrity & tests to review the details.")
if repaired:
    st.warning(f"{repaired} invalid model prediction value(s) were replaced with safe fallbacks.")

tab_overview, tab_work, tab_time, tab_tasks, tab_ml, tab_exp, tab_checks, tab_guide = st.tabs(
    ["Overview", "Workload", "Schedule timeline", "Task results", "ML model", "Experiments", "Integrity & tests", "How it works"]
)

# ---- Overview -------------------------------------------------------------- #
with tab_overview:
    st.write("Start with missed deadlines and waiting time. The table and charts show the same run in more detail.")
    st.markdown("#### Results by scheduling rule")
    st.caption(
        "FIFO follows arrival order · EDF gives priority to the closest deadline · "
        "ML-EDF also uses model estimates."
    )
    cols = st.columns(len(active))
    for col, scheduler in zip(cols, active):
        r = results[scheduler].summary
        with col:
            st.metric(
                scheduler,
                f"{r['deadline_misses']} missed",
                f"{r['deadline_miss_rate_pct']}% of tasks · ends at {r['total_simulation_time']}",
                delta_color="off",
            )

    st.markdown("#### Compare all reported measures")
    comparison = table.rename(columns=LABELS).T
    comparison.index.name = "Measure"
    st.dataframe(comparison)

    st.markdown("#### Which rule led on each measure?")
    st.dataframe(best_by_metric(table), hide_index=True)
    st.caption(
        "This result describes one workload. A rule that leads here may not lead on another workload. "
        "Use Experiments to compare across multiple sizes, difficulty levels and random seeds."
    )

    chart_keys = [
        "deadline_misses",
        "avg_waiting_time",
        "avg_turnaround_time",
        "cpu_utilization_pct",
        "throughput",
        "context_switches",
    ]
    gcols = st.columns(3)
    for i, key in enumerate(chart_keys):
        fig = px.bar(
            table.reset_index(),
            x="scheduler",
            y=key,
            color="scheduler",
            color_discrete_map=COLORS,
            title=LABELS[key],
        )
        fig.update_layout(
            showlegend=False,
            height=300,
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title=None,
            yaxis_title=None,
        )
        gcols[i % 3].plotly_chart(fig, key=f"cmp_{key}")

# ---- Workload -------------------------------------------------------------- #
with tab_work:
    st.subheader("The task list used in this run")
    st.write(
        "Every scheduling rule received this same task list. Task ID is the label used in results and timelines."
    )
    shown = workload.copy()
    shown.index.name = "row"
    st.dataframe(shown)
    st.download_button("Download task list (CSV)", workload.to_csv(index=False), "workload.csv", "text/csv")
    st.caption(
        f"Workload ID {next(iter(fingerprints))} is calculated from the task data. "
        "It helps confirm that the scheduler results use the same workload."
    )

# ---- Timelines ------------------------------------------------------------- #
with tab_time:
    st.subheader("When each task used the CPU")
    t_min = int(workload["arrival_time"].min())
    t_max = int(max(r.summary["total_simulation_time"] for r in results.values()))
    t_lo, t_hi = st.slider("Show virtual time from–to", t_min, t_max, (t_min, min(t_max, t_min + 100)))
    st.caption(
        "Green bars finish by the deadline; red bars finish late. Diamonds show deadlines. "
        "Gray sections mark idle time or a context switch."
    )
    for scheduler in active:
        st.plotly_chart(
            gantt_figure(
                results[scheduler],
                f"{scheduler} · finishes at time {results[scheduler].summary['total_simulation_time']}",
                t_lo,
                t_hi,
            ),
            key=f"gantt_{scheduler}",
        )
        order = (
            results[scheduler]
            .trace[results[scheduler].trace["event_type"] == "RUN"]
            .drop_duplicates("task_id")["task_id"]
            .tolist()
        )
        with st.expander(f"{scheduler} · task start order"):
            st.code(" → ".join(str(value) for value in order))

# ---- Task results ---------------------------------------------------------- #
with tab_tasks:
    st.subheader("Look up task outcomes")
    st.caption("Times use virtual time units. A task that finishes exactly at its deadline counts as on time.")
    scheduler = st.selectbox("Choose a scheduling rule", active, key="task_sched")
    only_missed = st.checkbox("Only show tasks that missed their deadline")
    cols_ = [
        "task_id",
        "arrival_time",
        "execution_time",
        "pred_exec",
        "miss_risk",
        "deadline",
        "start_time",
        "completion_time",
        "remaining_time",
        "waiting_time",
        "response_time",
        "turnaround_time",
        "tardiness",
        "deadline_missed",
    ]
    detail = results[scheduler].tasks[[column for column in cols_ if column in results[scheduler].tasks.columns]].copy()
    detail.index.name = "row"
    for column in ("pred_exec", "miss_risk"):
        if column in detail:
            detail[column] = detail[column].round(3)
    if only_missed:
        detail = detail[detail["deadline_missed"]]

    display_names = {
        "task_id": "Task ID",
        "arrival_time": "Arrives at",
        "execution_time": "Actual length",
        "pred_exec": "Predicted length",
        "miss_risk": "Predicted miss chance",
        "deadline": "Deadline",
        "start_time": "First start",
        "completion_time": "Finished at",
        "remaining_time": "Remaining length",
        "waiting_time": "Waiting",
        "response_time": "Response",
        "turnaround_time": "Turnaround",
        "tardiness": "Time past deadline",
        "deadline_missed": "Missed deadline",
    }
    st.dataframe(detail.rename(columns=display_names))
    st.download_button("Download these task results (CSV)", detail.to_csv(), f"{scheduler}_tasks.csv", "text/csv")
    with st.expander("View the raw execution trace"):
        st.caption("The trace lists each run segment, idle interval and context switch in time order.")
        st.dataframe(results[scheduler].trace)

# ---- ML model -------------------------------------------------------------- #
with tab_ml:
    if models is None:
        st.warning("The ML models are unavailable, so ML-EDF is not included in this run.")
    else:
        st.subheader("How the ML estimates were evaluated")
        st.write(
            f"The models were trained on {models['train_rows']:,} tasks and evaluated on "
            f"{models['test_rows']:,} held-out tasks from separate synthetic workloads. "
            "The deadline-risk labels come from EDF simulations, so the classifier estimates risk under EDF, "
            "not under ML-EDF. These scores show how well the models learned this synthetic task generator; "
            "they are not a measure of real-world performance."
        )
        a, b = st.columns(2)
        with a:
            st.markdown("#### Task-length estimate")
            st.caption("MAE is the average prediction error; RMSE gives larger errors extra weight; R² summarizes fit.")
            m = models["reg_metrics"]
            x1, x2, x3 = st.columns(3)
            x1.metric("MAE", f"{m['MAE']:.2f}")
            x2.metric("RMSE", f"{m['RMSE']:.2f}")
            x3.metric("R²", f"{m['R2']:.3f}")
            fig = px.scatter(
                models["reg_test"],
                x="actual",
                y="predicted",
                opacity=0.5,
                title="Estimated length compared with actual length",
            )
            lim = float(models["reg_test"].max().max())
            fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(dash="dash", color="#888"))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, key="ml_reg_test")
            st.plotly_chart(
                px.bar(models["reg_importance"].sort_values(), orientation="h", title="What influenced the estimate").update_layout(
                    showlegend=False,
                    height=220,
                    margin=dict(l=10, r=10, t=50, b=10),
                    xaxis_title=None,
                    yaxis_title=None,
                ),
                key="ml_reg_imp",
            )
        with b:
            st.markdown("#### Deadline-miss estimate")
            if models["clf"] is None:
                st.warning(
                    "The training workloads had only one outcome (all tasks met or all missed their deadline). "
                    "A classifier could not be trained, so ML-EDF uses a risk estimate of 0."
                )
            else:
                cm_ = models["clf_metrics"]
                mc = st.columns(4)
                for i, (key, value) in enumerate(cm_.items()):
                    mc[i % 4].metric(key, f"{value:.3f}")
                st.caption(
                    f"Tasks that missed under EDF: {models['train_miss_rate']:.1%} in training and "
                    f"{models['test_miss_rate']:.1%} in testing. Accuracy can be misleading when one outcome is rare. "
                    "Brier score reflects probability quality; accuracy checks the 0.5 cutoff."
                )
                fig = px.imshow(
                    models["confusion"],
                    text_auto=True,
                    x=["Predicted on time", "Predicted missed"],
                    y=["Actually on time", "Actually missed"],
                    color_continuous_scale="Blues",
                    title="Predictions compared with EDF outcomes (0.5 cutoff)",
                )
                fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, key="ml_cm")
                st.plotly_chart(
                    px.bar(
                        models["clf_importance"].sort_values(),
                        orientation="h",
                        title="What influenced the risk estimate",
                    ).update_layout(
                        showlegend=False,
                        height=220,
                        margin=dict(l=10, r=10, t=50, b=10),
                        xaxis_title=None,
                        yaxis_title=None,
                    ),
                    key="ml_clf_imp",
                )

        st.markdown("#### Estimates for the current task list")
        pa = prediction_analysis(work, results, risk_weight, ml_variant)
        st.caption(
            "These are estimates, not outcomes. A predicted miss chance is not a guarantee. "
            "ML-EDF ranks a task using the selected decision rule; a smaller score means higher priority. "
            + (
                "Slack–risk uses (deadline − arrival − predicted remaining length) − weight × miss chance."
                if ml_variant == "slack_risk"
                else "Risk-adjusted deadline uses deadline − weight × miss chance × predicted length."
            )
        )
        p1, p2 = st.columns(2)
        fig = px.scatter(
            pa,
            x="execution_time",
            y="pred_exec",
            opacity=0.6,
            title="Current tasks · actual and estimated length",
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=float(pa["execution_time"].max()),
            y1=float(pa["execution_time"].max()),
            line=dict(dash="dash", color="#888"),
        )
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
        p1.plotly_chart(fig, key="ml_cur_scatter")
        if "missed_under_EDF" in pa:
            fig = px.box(
                pa,
                x="missed_under_EDF",
                y="miss_risk",
                points="all",
                title="Estimated miss chance vs. EDF outcome",
            )
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
            p2.plotly_chart(fig, key="ml_cur_box")
        st.dataframe(pa.round(3), hide_index=True)

        import json
        report = {
            "regression": models["reg_metrics"],
            "classifier": models["clf_metrics"],
            "confusion_matrix": None if models["confusion"] is None else models["confusion"].tolist(),
            "train_miss_rate": models["train_miss_rate"],
            "test_miss_rate": models["test_miss_rate"],
            "config": models["config"],
        }
        st.download_button(
            "Download ML evaluation report (JSON)",
            json.dumps(report, indent=2),
            "ml_evaluation.json",
            "application/json",
        )

# ---- Experiments ----------------------------------------------------------- #
with tab_exp:
    st.subheader("Compare across many workloads")
    st.write(
        "Run each available scheduler on the same generated workloads at several sizes and difficulty levels. "
        "The results include averages and variation across random seeds."
    )
    e1, e2, e3 = st.columns(3)
    sizes = e1.multiselect("Task-list sizes", [30, 60, 120, 200], default=[60])
    levels = e2.multiselect("Workload intensities", list(DIFFICULTY), default=list(DIFFICULTY))
    n_seeds = e3.slider("Random seeds per setting", 3, 30, 10)
    if st.button("Run experiment", type="primary"):
        if not sizes or not levels:
            st.warning("Choose at least one task-list size and one workload intensity.")
        else:
            bar = st.progress(0.0)
            raw_exp = run_experiment(
                sizes,
                levels,
                n_seeds,
                seed,
                models,
                preemptive,
                cs_cost,
                risk_weight,
                ml_variant,
                bar,
            )
            bar.empty()
            st.session_state["experiment"] = raw_exp
            st.session_state["experiment_cfg"] = {
                "preemptive": preemptive,
                "cs_cost": cs_cost,
                "ml_variant": ml_variant,
                "risk_weight": risk_weight,
                "base_seed": seed,
            }
    exp = st.session_state.get("experiment")
    if exp is not None:
        st.caption(f"Settings used for this run: {st.session_state.get('experiment_cfg')}")
        if not exp["integrity_ok"].all():
            st.error(f"{int((~exp['integrity_ok']).sum())} simulation(s) failed integrity checks.")
        keys = [
            "deadline_misses",
            "deadline_miss_rate_pct",
            "avg_waiting_time",
            "avg_turnaround_time",
            "avg_response_time",
            "cpu_utilization_pct",
            "throughput",
            "context_switches",
            "preemptions",
        ]
        agg = exp.groupby(["tasks", "difficulty", "scheduler"])[keys].agg(["mean", "std"]).round(3)
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg = agg.reset_index()
        st.dataframe(agg, hide_index=True)

        if {"EDF", "ML-EDF"}.issubset(set(exp["scheduler"])):
            piv = exp.pivot_table(
                index=["tasks", "difficulty", "seed"],
                columns="scheduler",
                values="deadline_misses",
            ).reset_index()
            piv["ML-EDF fewer misses"] = (piv["ML-EDF"] < piv["EDF"]).astype(int)
            piv["tie"] = (piv["ML-EDF"] == piv["EDF"]).astype(int)
            piv["ML-EDF more misses"] = (piv["ML-EDF"] > piv["EDF"]).astype(int)
            cmp_ = piv.groupby(["tasks", "difficulty"])[
                ["ML-EDF fewer misses", "tie", "ML-EDF more misses"]
            ].sum().reset_index()
            st.markdown("#### ML-EDF compared with EDF, workload by workload")
            st.dataframe(cmp_, hide_index=True)

        pick = st.selectbox("Measure to chart", keys, format_func=lambda key: LABELS.get(key, key))
        size_pick = st.selectbox("Task-list size", sorted(exp["tasks"].unique()))
        sub = agg[agg["tasks"] == size_pick]
        fig = px.bar(
            sub,
            x="difficulty",
            y=f"{pick}_mean",
            error_y=f"{pick}_std",
            color="scheduler",
            barmode="group",
            color_discrete_map=COLORS,
            category_orders={"difficulty": list(DIFFICULTY), "scheduler": SCHEDULERS},
            title=f"{LABELS.get(pick, pick)} · average and standard deviation across seeds",
        )
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, key="exp_chart")
        d1, d2, d3 = st.columns(3)
        d1.download_button("Download summary (CSV)", agg.to_csv(index=False), "scheduler_comparison.csv", "text/csv")
        d2.download_button("Download all results (CSV)", exp.to_csv(index=False), "experiment_raw.csv", "text/csv")
        d3.download_button(
            "Download all results (JSON)",
            exp.to_json(orient="records", indent=2),
            "experiment_raw.json",
            "application/json",
        )
    else:
        st.info("Choose the experiment settings, then select Run experiment.")

# ---- Integrity & tests ----------------------------------------------------- #
with tab_checks:
    st.subheader("Checks on this simulation")
    st.write(
        "The app checks the execution trace for overlapping work, tasks starting before arrival, "
        "execution time matching the task list, correct deadline flags and consistent total time."
    )
    for scheduler in active:
        ck = pd.DataFrame(checks[scheduler])
        ok = ck["passed"].all()
        label = (
            f"{scheduler} · all {len(ck)} checks passed"
            if ok
            else f"{scheduler} · review failed checks"
        )
        with st.expander(label, expanded=not ok):
            ck["passed"] = ck["passed"].map({True: "Pass", False: "Review"})
            st.dataframe(ck, hide_index=True)

    st.markdown("#### Built-in self-tests")
    st.caption(
        "These small examples check the heap, scheduling rules, context-switch accounting, idle periods, "
        "CSV validation and safe handling of invalid ML predictions."
    )
    if st.button("Run self-tests"):
        st.session_state["self_tests"] = run_self_tests()
    tests = st.session_state.get("self_tests")
    if tests is not None:
        n_pass = int(tests["passed"].sum())
        (st.success if n_pass == len(tests) else st.error)(
            f"{n_pass} of {len(tests)} self-tests passed."
        )
        view = tests.copy()
        view["passed"] = view["passed"].map({True: "Pass", False: "Review"})
        st.dataframe(view, hide_index=True)

# ---- How it works ---------------------------------------------------------- #
with tab_guide:
    st.subheader("A plain-language guide")
    st.markdown(
        r"""
**What is being simulated?** Each task has an arrival time, an actual execution length, a deadline,
a priority, a task type and an input size. Time is measured in whole virtual time units.

**How do the scheduling rules choose the next task?**
- **FIFO** runs the task that arrived first. If arrivals tie, the task ID breaks the tie.
- **EDF** runs the ready task with the closest deadline. Priority, arrival time and task ID break ties.
  EDF does not use ML predictions.
- **ML-EDF** adjusts its ranking using the predicted task length and predicted deadline-miss chance.
  A smaller score means a task is ranked sooner.

**ML-EDF decision rules**
- *Slack – risk* (the original ML-EDF formula): $\text{score} = (D_i - t - \hat{C}_i^{rem}) - w \cdot P_i$
- *Risk-adjusted deadline*: $\text{score} = D_i - w \cdot P_i \cdot \hat{C}_i$. With $w=0$, this follows EDF.

Here $\hat{C}_i$ is the estimated execution length, $\hat{C}_i^{rem}$ is estimated remaining length,
$P_i$ is the estimated chance of a deadline miss, and $w$ is the risk weight. The slack–risk score changes
while a task is running, so it may cause more preemptions. Context-switch time can add to the total run time.

**How to read the measures**
- A task finishing exactly at its deadline is on time; a later finish counts as a missed deadline.
- Waiting time is turnaround time minus actual execution time. Response time is first start minus arrival.
- Turnaround time is completion minus arrival. Tardiness is time beyond the deadline, or zero when on time.
- CPU utilization is time spent running tasks divided by total simulation time after the first arrival.
  Idle and context-switch time are reported separately.
- Throughput is completed tasks per unit of simulation time.

**What this app does not model**
This is a simulator inspired by real-time operating systems. It does not run inside an RTOS and does not measure
hardware timing. Task types are labels: IO tasks do not block, and PERIODIC tasks do not repeat. Workloads and
ML training examples are synthetic; the classifier learns miss labels from EDF runs. ML-EDF is a project-specific
heuristic and is not guaranteed to outperform EDF.
"""
    )

