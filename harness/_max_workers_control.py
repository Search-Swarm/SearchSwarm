"""Runtime-adjustable MAX_WORKERS control for run_multi_react.py.

The executor pool is sized to a fixed ceiling, while active concurrency is
gated by an adjustable semaphore. Edit the control file while a run is alive
to raise or lower the active cap without restarting the rollout process.
"""

import os
import threading
import time


_DEFAULT_INTERVAL = 5.0
_DEFAULT_POOL_CAP = 1024


class AdjustableSemaphore:
    """Semaphore with a runtime-adjustable active cap."""

    def __init__(self, initial):
        if initial < 1:
            raise ValueError("initial must be >= 1")
        self._sem = threading.Semaphore(initial)
        self._current = initial
        self._lock = threading.Lock()

    def __enter__(self):
        self._sem.acquire()
        return self

    def __exit__(self, *exc):
        self._sem.release()

    def adjust_to(self, new_max):
        if new_max < 1:
            raise ValueError("new_max must be >= 1")
        with self._lock:
            diff = new_max - self._current
            self._current = new_max
        if diff > 0:
            for _ in range(diff):
                self._sem.release()
        elif diff < 0:
            for _ in range(-diff):
                self._sem.acquire()
        return diff

    @property
    def current(self):
        return self._current


def pool_capacity():
    return int(os.environ.get("MAX_WORKERS_POOL_CAP", str(_DEFAULT_POOL_CAP)))


def init_control(initial_max, control_file=None, interval=None, pool_cap=None):
    if control_file is None:
        control_file = os.environ.get(
            "MAX_WORKERS_CONTROL_FILE", "max_workers_control.txt")
    control_file = os.path.abspath(control_file)
    if interval is None:
        interval = float(os.environ.get(
            "MAX_WORKERS_WATCH_INTERVAL_S", str(_DEFAULT_INTERVAL)))
    if pool_cap is None:
        pool_cap = pool_capacity()

    if initial_max > pool_cap:
        print(f"[max_workers_control] WARNING: initial={initial_max} > "
              f"pool_cap={pool_cap}; clamping initial to pool_cap. Set "
              f"MAX_WORKERS_POOL_CAP higher in env if needed.",
              flush=True)
        initial_max = pool_cap

    sem = AdjustableSemaphore(initial_max)

    try:
        dirname = os.path.dirname(control_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(control_file, "w", encoding="utf-8") as f:
            f.write(f"{initial_max}\n")
    except Exception as e:
        print(f"[max_workers_control] could not seed control file "
              f"{control_file}: {e}", flush=True)

    print(f"[max_workers_control] initial={initial_max} pool_cap={pool_cap} "
          f"poll={interval}s control_file={control_file}", flush=True)
    print(f"[max_workers_control] to change at runtime: "
          f"echo <N> > {control_file}", flush=True)

    def _watcher():
        last = initial_max
        while True:
            time.sleep(interval)
            try:
                with open(control_file, encoding="utf-8") as f:
                    raw = f.read().strip()
            except FileNotFoundError:
                continue
            except OSError as e:
                print(f"[max_workers_control] read error: {e}", flush=True)
                continue
            if not raw:
                continue
            try:
                val = int(raw)
            except ValueError:
                print(f"[max_workers_control] non-integer in control file "
                      f"({raw!r}); ignoring", flush=True)
                continue
            if val == last:
                continue
            if val < 1:
                print(f"[max_workers_control] ignoring invalid cap value "
                      f"{val} (<1)", flush=True)
                continue
            if val > pool_cap:
                print(f"[max_workers_control] WARNING: requested {val} > "
                      f"pool_cap={pool_cap}; clamping to {pool_cap}",
                      flush=True)
                val = pool_cap
            print(f"[max_workers_control] adjusting {last} -> {val} "
                  f"({'+' if val > last else ''}{val - last}); "
                  f"shrinks wait for running tasks", flush=True)
            sem.adjust_to(val)
            last = val
            print(f"[max_workers_control] adjusted: active cap = {val}",
                  flush=True)

    thread = threading.Thread(
        target=_watcher, daemon=True, name="max_workers_watcher")
    thread.start()
    return sem
