"""
history_file.py
===================================================================
Persist the per-step history to a JSONL file so it survives process
restarts. Without this, driver.history only covers the lifetime of the
current process (~5 s for a typical query script), so a turbine that
span for two minutes across several runs is invisible.

One JSON object per line, appended as the sim runs. Each sample carries
"epoch_s" (wall-clock seconds) so samples from different runs can be
ordered and stitched into one continuous timeline -- driver.elapsed
restarts at 0 every process, so "time_s" alone is NOT comparable
across runs.

Usage:
    # in the driver: append every recorded sample
    from history_file import append_sample
    append_sample(sample, "history.jsonl")

    # in a query script: load everything ever recorded
    from history_file import load_history
    from history_queries import spinning_intervals
    log = load_history("history.jsonl")
    print(spinning_intervals(log, "Farm_East_tall"))
===================================================================
"""

import json
import os
import time

DEFAULT_HISTORY_FILE = "history.jsonl"


class HistoryLog:
    """Thin wrapper exposing .history, so history_queries works on a file too."""

    def __init__(self, history):
        self.history = history

    def __len__(self):
        return len(self.history)


def append_sample(sample: dict, path: str = DEFAULT_HISTORY_FILE) -> None:
    """Append one sample as a JSON line. Adds epoch_s if not already present."""
    sample = dict(sample)
    sample.setdefault("epoch_s", time.time())
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(sample) + "\n")


def load_history(path: str = DEFAULT_HISTORY_FILE) -> HistoryLog:
    """Load every recorded sample, ordered, with a continuous time_s timeline.

    time_s is rewritten as seconds since the FIRST sample in the file, so
    intervals spanning several runs make sense (each run's own elapsed
    counter restarts at 0 and would otherwise jump backwards).
    """
    samples = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue          # skip a torn last line
    except FileNotFoundError:
        return HistoryLog([])

    samples.sort(key=lambda s: s.get("epoch_s", 0.0))
    if samples:
        t0 = samples[0].get("epoch_s", 0.0)
        for s in samples:
            s["time_s"] = s.get("epoch_s", t0) - t0
    return HistoryLog(samples)


def clear_history(path: str = DEFAULT_HISTORY_FILE) -> bool:
    """Delete the log (start a fresh experiment).

    Returns True if a file was actually removed, False if none was found.
    Prints the ABSOLUTE path so a working-directory mismatch is obvious
    (the old version failed silently when called from the wrong folder).
    """
    ap = os.path.abspath(path)
    if os.path.exists(ap):
        os.remove(ap)
        print(f"[clear_history] deleted {ap}")
        return True
    print(f"[clear_history] nothing to delete at {ap}")
    return False