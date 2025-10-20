#!/usr/bin/env python3
"""
Latency benchmarking infrastructure for Pocket Whisper.
This script provides a skeleton for measuring end-to-end latency.
"""

import time
from dataclasses import dataclass
from typing import List, Dict
import statistics


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""

    component: str
    latency_ms: float
    timestamp: float


class LatencyProfiler:
    """Profile latency across different components."""

    def __init__(self):
        self.measurements: Dict[str, List[float]] = {}

    def record(self, component: str, latency_ms: float):
        """Record a latency measurement for a component."""
        if component not in self.measurements:
            self.measurements[component] = []
        self.measurements[component].append(latency_ms)

    def get_stats(self, component: str) -> Dict[str, float]:
        """Get statistics for a component."""
        if component not in self.measurements or not self.measurements[component]:
            return {}

        data = sorted(self.measurements[component])
        n = len(data)

        return {
            "count": n,
            "min": data[0],
            "max": data[-1],
            "avg": statistics.mean(data),
            "median": statistics.median(data),
            "p95": data[int(n * 0.95)] if n > 0 else 0,
            "p99": data[int(n * 0.99)] if n > 0 else 0,
        }

    def print_report(self):
        """Print a formatted latency report."""
        print("\n" + "=" * 70)
        print("LATENCY PROFILE REPORT")
        print("=" * 70)

        for component in sorted(self.measurements.keys()):
            stats = self.get_stats(component)
            if not stats:
                continue

            print(f"\n{component}:")
            print(f"  Count:  {stats['count']}")
            print(f"  Min:    {stats['min']:.1f}ms")
            print(f"  Avg:    {stats['avg']:.1f}ms")
            print(f"  Median: {stats['median']:.1f}ms")
            print(f"  P95:    {stats['p95']:.1f}ms")
            print(f"  P99:    {stats['p99']:.1f}ms")
            print(f"  Max:    {stats['max']:.1f}ms")

        print("\n" + "=" * 70)

    def check_targets(self, targets: Dict[str, float]) -> bool:
        """Check if measurements meet target thresholds."""
        all_passed = True

        print("\n" + "=" * 70)
        print("TARGET VALIDATION")
        print("=" * 70)

        for component, target in targets.items():
            if component not in self.measurements:
                print(f"\n❌ {component}: NO DATA")
                all_passed = False
                continue

            stats = self.get_stats(component)
            p95 = stats.get("p95", float("inf"))

            passed = p95 <= target
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"\n{status} {component}:")
            print(f"  Target: ≤{target}ms (P95)")
            print(f"  Actual: {p95:.1f}ms")

            if not passed:
                all_passed = False

        print("\n" + "=" * 70)
        return all_passed


def example_usage():
    """Example of how to use the latency profiler."""
    profiler = LatencyProfiler()

    # Simulate some measurements
    print("Simulating latency measurements...")

    for i in range(100):
        # Simulate VAD (should be <10ms)
        vad_latency = 5 + (i % 3)
        profiler.record("vad", vad_latency)

        # Simulate ASR (should be <150ms)
        asr_latency = 80 + (i % 40)
        profiler.record("asr_encoder", asr_latency)

        # Simulate LLM (should be <200ms)
        llm_latency = 120 + (i % 60)
        profiler.record("llm", llm_latency)

        # Simulate TTS (should be <100ms)
        tts_latency = 40 + (i % 30)
        profiler.record("tts", tts_latency)

        # Total (should be <350ms)
        total = vad_latency + asr_latency + llm_latency + tts_latency
        profiler.record("total", total)

    # Print report
    profiler.print_report()

    # Check against targets
    targets = {
        "vad": 10,
        "asr_encoder": 150,
        "llm": 200,
        "tts": 100,
        "total": 350,
    }

    passed = profiler.check_targets(targets)

    if passed:
        print("\n✅ ALL TARGETS MET!")
    else:
        print("\n❌ SOME TARGETS MISSED - OPTIMIZATION NEEDED")


if __name__ == "__main__":
    example_usage()
