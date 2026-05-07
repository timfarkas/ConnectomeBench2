"""
Thread-safe profiling utilities for ConnectomeEnv.

Usage:
    from utils.profiler import Profiler, profiled

    # Mark functions to profile with decorator
    class MyClass:
        @profiled("load_data")
        def load_data(self):
            ...

        @profiled(lambda self: f"step{self.step}.process")  # Dynamic name
        def process(self):
            ...

    # Enable profiling and run
    Profiler.enable()
    obj.load_data()
    obj.process()

    # Print timing summary
    Profiler.summary()
"""

import threading
import time
import functools
from typing import Dict, List, Callable, Any, Optional, Set, Tuple, Union
from collections import defaultdict


class TimingEntry:
    """A single timing entry with its full call ancestry."""
    __slots__ = ('elapsed', 'parent', 'ancestry')

    def __init__(self, elapsed: float, parent: Optional[str] = None, ancestry: Tuple[str, ...] = ()):
        self.elapsed = elapsed
        self.parent = parent
        self.ancestry = ancestry  # Full call stack (grandparent, ..., parent)


class Profiler:
    """Global profiler for timing code sections."""

    _enabled: bool = False
    _timings: Dict[str, List[TimingEntry]] = defaultdict(list)
    _timings_lock: threading.Lock = threading.Lock()
    _counters: Dict[str, List[int]] = defaultdict(list)
    _counters_lock: threading.Lock = threading.Lock()
    _local: threading.local = threading.local()  # per-thread call stack
    _targets: Optional[Set[str]] = None  # None = all, Set = only these

    @classmethod
    def _get_stack(cls) -> List[str]:
        """Get the per-thread call stack."""
        if not hasattr(cls._local, 'stack'):
            cls._local.stack = []
        return cls._local.stack

    @classmethod
    def enable(cls, targets: Optional[List[str]] = None):
        """Enable profiling.

        Args:
            targets: Optional list of function names to profile.
                     If None, profiles all @profiled functions.
                     If list, only profiles functions whose names contain any target string.
        """
        cls._enabled = True
        with cls._timings_lock:
            cls._timings.clear()
        with cls._counters_lock:
            cls._counters.clear()
        cls._get_stack().clear()
        cls._targets = set(targets) if targets else None

    @classmethod
    def disable(cls):
        """Disable profiling."""
        cls._enabled = False

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled

    @classmethod
    def should_profile(cls, name: str) -> bool:
        """Check if a given name should be profiled."""
        if not cls._enabled:
            return False
        if cls._targets is None:
            return True
        # Check if name matches any target (substring match)
        return any(target in name for target in cls._targets)

    @classmethod
    def push(cls, name: str):
        """Push a timer onto the stack (called when entering a timed block)."""
        if cls._enabled:
            cls._get_stack().append(name)

    @classmethod
    def pop(cls):
        """Pop a timer from the stack (called when exiting a timed block)."""
        if cls._enabled:
            stack = cls._get_stack()
            if stack:
                stack.pop()

    @classmethod
    def get_current_parent(cls) -> Optional[str]:
        """Get the current parent from the stack."""
        stack = cls._get_stack()
        if stack:
            return stack[-1]
        return None

    @classmethod
    def get_current_ancestry(cls) -> Tuple[str, ...]:
        """Get the full current call stack as ancestry."""
        return tuple(cls._get_stack())

    @classmethod
    def record(cls, name: str, elapsed: float, parent: Optional[str] = None, ancestry: Tuple[str, ...] = ()):
        """Record a timing with its parent context and full ancestry."""
        if cls._enabled:
            with cls._timings_lock:
                cls._timings[name].append(TimingEntry(elapsed, parent, ancestry))

    @classmethod
    def count(cls, name: str, hit: int = 1):
        """Record a hit/miss event. Displayed as hits/total in summary."""
        if cls._enabled:
            with cls._counters_lock:
                cls._counters[name].append(hit)

    @classmethod
    def get_data(cls) -> Dict[str, Dict[str, Any]]:
        """Get timing data as dict with stats."""
        result = {}
        with cls._timings_lock:
            items = list(cls._timings.items())
        for name, entries in items:
            if entries:
                times = [e.elapsed for e in entries]
                result[name] = {
                    'count': len(entries),
                    'total': sum(times),
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'entries': entries,  # Keep full entries for hierarchy
                }
        return result

    @classmethod
    def get_call_tree(cls) -> List[Dict[str, Any]]:
        """Get timing data as a list of calls preserving order and hierarchy."""
        # Reconstruct call order from entries
        calls = []
        for name, entries in cls._timings.items():
            for entry in entries:
                calls.append({
                    'name': name,
                    'elapsed': entry.elapsed,
                    'parent': entry.parent,
                })
        return calls

    @classmethod
    def summary(cls, sort_by: str = 'total', group_by_step: bool = True):
        """Print a summary of timing data with hierarchy.

        Args:
            sort_by: Key to sort by ('total', 'mean', 'count', 'max', 'min')
            group_by_step: If True, group timings by step prefix (reset, step1, step2, etc.)
        """
        data = cls.get_data()
        if not data:
            print("No profiling data collected. Did you call Profiler.enable()?")
            return

        # Build per-ancestry aggregation: for each (name, ancestry) pair, aggregate stats
        # This lets us correctly show children only under their actual call path
        # Key: (name, ancestry_tuple) -> {'times': [...], 'count': ..., 'total': ..., etc}
        per_ancestry_stats: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = defaultdict(lambda: {'times': []})

        for name, stats in data.items():
            for entry in stats['entries']:
                key = (name, entry.ancestry)
                per_ancestry_stats[key]['times'].append(entry.elapsed)

        # Compute stats for each (name, ancestry) pair
        for key in per_ancestry_stats:
            times = per_ancestry_stats[key]['times']
            per_ancestry_stats[key].update({
                'count': len(times),
                'total': sum(times),
                'mean': sum(times) / len(times) if times else 0,
                'min': min(times) if times else 0,
                'max': max(times) if times else 0,
            })

        # Find root nodes (those with empty ancestry)
        roots: List[str] = []
        for (name, ancestry) in per_ancestry_stats:
            if len(ancestry) == 0 and name not in roots:
                roots.append(name)

        # Calculate total time from root nodes only
        total_time = sum(per_ancestry_stats[(name, ())]['total'] for name in roots if (name, ()) in per_ancestry_stats)

        print("\n" + "=" * 95)
        print("PROFILING SUMMARY (hierarchical)")
        print("=" * 95)

        def get_children_with_ancestry(parent_name: str, parent_ancestry: Tuple[str, ...]) -> List[Tuple[str, Tuple[str, ...]]]:
            """Find all children that have this exact ancestry (parent_ancestry + parent_name)."""
            expected_ancestry = parent_ancestry + (parent_name,)
            children = []
            for (name, ancestry) in per_ancestry_stats:
                if ancestry == expected_ancestry:
                    children.append((name, ancestry))
            return children

        def print_node(name: str, ancestry: Tuple[str, ...], depth: int, parent_total: float):
            """Print a node and its children under a specific ancestry context."""
            key = (name, ancestry)
            if key not in per_ancestry_stats:
                return

            stats = per_ancestry_stats[key]
            indent = "  " * depth
            pct = (stats['total'] / parent_total * 100) if parent_total > 0 else 0

            display_name = f"{indent}└─ {name}"
            print(f"{display_name:<45} {stats['count']:>5} {stats['total']:>8.3f}s {stats['mean']:>8.3f}s ({pct:>5.1f}%)")

            # Find and print children that were called from this node
            children = get_children_with_ancestry(name, ancestry)
            if children:
                # Sort by total time
                sorted_children = sorted(
                    children,
                    key=lambda c: per_ancestry_stats[c]['total'],
                    reverse=True
                )
                for child_name, child_ancestry in sorted_children:
                    print_node(child_name, child_ancestry, depth + 1, stats['total'])

        if group_by_step:
            # Group root nodes by step prefix
            groups: Dict[str, List[str]] = defaultdict(list)
            for name in roots:
                if name.startswith("reset"):
                    groups["reset"].append(name)
                elif name.startswith("step"):
                    dot_idx = name.find(".")
                    if dot_idx > 0:
                        prefix = name[:dot_idx]
                        groups[prefix].append(name)
                    else:
                        groups["other"].append(name)
                else:
                    groups["other"].append(name)

            def step_sort_key(key):
                if key == "reset":
                    return (0, 0)
                elif key.startswith("step"):
                    try:
                        return (1, int(key[4:]))
                    except ValueError:
                        return (2, 0)
                return (3, 0)

            sorted_groups = sorted(groups.keys(), key=step_sort_key)

            print(f"{'Operation':<45} {'Count':>5} {'Total':>9} {'Mean':>9} {'% of parent'}")
            print("-" * 95)

            for group in sorted_groups:
                group_roots = groups[group]
                group_total = sum(per_ancestry_stats.get((n, ()), {}).get('total', 0) for n in group_roots)
                group_pct = (group_total / total_time * 100) if total_time > 0 else 0

                print(f"\n[{group.upper()}] ({group_total:.3f}s, {group_pct:.1f}% of total)")

                sorted_roots = sorted(group_roots, key=lambda x: per_ancestry_stats.get((x, ()), {}).get('total', 0), reverse=True)
                for root in sorted_roots:
                    root_key = (root, ())
                    if root_key not in per_ancestry_stats:
                        continue
                    root_stats = per_ancestry_stats[root_key]
                    indent = "  "
                    pct = (root_stats['total'] / group_total * 100) if group_total > 0 else 0
                    print(f"{indent}└─ {root:<42} {root_stats['count']:>5} {root_stats['total']:>8.3f}s {root_stats['mean']:>8.3f}s ({pct:>5.1f}%)")

                    # Print children with ancestry = (root,)
                    children = get_children_with_ancestry(root, ())
                    sorted_children = sorted(
                        children,
                        key=lambda c: per_ancestry_stats[c]['total'],
                        reverse=True
                    )
                    for child_name, child_ancestry in sorted_children:
                        print_node(child_name, child_ancestry, 2, root_stats['total'])

        else:
            print(f"{'Operation':<45} {'Count':>5} {'Total':>9} {'Mean':>9} {'% of parent'}")
            print("-" * 95)

            sorted_roots = sorted(roots, key=lambda x: per_ancestry_stats.get((x, ()), {}).get('total', 0), reverse=True)
            for root in sorted_roots:
                root_key = (root, ())
                if root_key not in per_ancestry_stats:
                    continue
                root_stats = per_ancestry_stats[root_key]
                pct = (root_stats['total'] / total_time * 100) if total_time > 0 else 0
                print(f"{root:<45} {root_stats['count']:>5} {root_stats['total']:>8.3f}s {root_stats['mean']:>8.3f}s ({pct:>5.1f}%)")

                children = get_children_with_ancestry(root, ())
                sorted_children = sorted(
                    children,
                    key=lambda c: per_ancestry_stats[c]['total'],
                    reverse=True
                )
                for child_name, child_ancestry in sorted_children:
                    print_node(child_name, child_ancestry, 1, root_stats['total'])

        # counters section
        with cls._counters_lock:
            counters = dict(cls._counters)
        if counters:
            print("\n" + "-" * 95)
            print("COUNTERS")
            print("-" * 95)
            for name, values in sorted(counters.items()):
                total = len(values)
                hits = sum(values)
                pct = (hits / total * 100) if total > 0 else 0
                print(f"  {name:<50} {hits:>5}/{total:<5} ({pct:>5.1f}%)")

        print("\n" + "-" * 95)
        print(f"{'TOTAL':<45} {'':<5} {total_time:>8.3f}s")
        print("=" * 95 + "\n")

    @classmethod
    def clear(cls):
        """Clear all timing data."""
        cls._timings.clear()
        cls._counters.clear()
        cls._get_stack().clear()


class Timer:
    """Context manager for timing code blocks (for manual use)."""

    def __init__(self, name: str):
        self.name = name
        self.start = None
        self._should_profile = False
        self._parent = None
        self._ancestry: Tuple[str, ...] = ()

    def __enter__(self):
        self._should_profile = Profiler.should_profile(self.name)
        if self._should_profile:
            self._parent = Profiler.get_current_parent()  # Capture parent before pushing
            self._ancestry = Profiler.get_current_ancestry()
            Profiler.push(self.name)
            self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self._should_profile and self.start is not None:
            elapsed = time.perf_counter() - self.start
            Profiler.record(self.name, elapsed, self._parent, self._ancestry)
            Profiler.pop()


def profiled(name: Union[str, Callable[..., str]] = None):
    """
    Decorator to profile a function.

    Args:
        name: Either a static string name, or a callable that takes the same
              arguments as the decorated function and returns the profile name.
              If None, uses the function's qualified name.

    Examples:
        @profiled  # Uses function name
        def my_func():
            ...

        @profiled("custom_name")  # Static name
        def my_func():
            ...

        @profiled(lambda self: f"step{self.current_step}.process")  # Dynamic name
        def process(self):
            ...
    """
    # Store the name_arg to use in the decorator
    name_arg = name

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not Profiler.is_enabled():
                return func(*args, **kwargs)

            # Determine the profile name
            if name_arg is None:
                profile_name = func.__qualname__
            elif callable(name_arg):
                try:
                    profile_name = name_arg(*args, **kwargs)
                except Exception:
                    profile_name = func.__qualname__
            else:
                profile_name = name_arg

            # Check if we should profile this
            if not Profiler.should_profile(profile_name):
                return func(*args, **kwargs)

            # Capture parent and full ancestry before pushing onto stack
            parent = Profiler.get_current_parent()
            ancestry = Profiler.get_current_ancestry()
            Profiler.push(profile_name)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                Profiler.record(profile_name, elapsed, parent, ancestry)
                Profiler.pop()

        return wrapper

    # Handle @profiled without parentheses (bare decorator)
    # We detect this by checking if name is a function (not a lambda or string)
    # Lambdas have __name__ == '<lambda>', regular functions have their actual name
    if callable(name) and getattr(name, '__name__', '') != '<lambda>':
        # name is actually the function being decorated (bare @profiled usage)
        func = name
        name_arg = None
        return decorator(func)

    return decorator


# Keep old name for backwards compatibility
profile = profiled
