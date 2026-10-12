# gpt-fast
با درنظرگرفتن زمان کل، انرژی، ریسک و هزینهٔ انتقال.
این سه مسئله باید با هم کار کنند، اما لازم نیست از ابتدا در یک فرمول بزرگ ادغام شوند. طراحی جداگانه و آزمایش‌پذیر معمولاً پیدا کردن خطاها و بهبود سیستم را آسان‌تر می‌کند.
قدم بعدی ما
این سند، نقشهٔ اولیهٔ GPT-Fast است. هنوز به پیاده‌سازی، تنظیم پارامترها و آزمایش واقعی نیاز دارد.
من می‌توانم نوشتن کد را بر عهده بگیرم؛ لازم نیست تو برنامه‌نویسی بلد باشی. برای شروع، هستهٔ زمان‌بندی و ثبت نتایج را می‌سازیم و بعد پالایش نویز را اضافه می‌کنیم تا بتوانیم با دادهٔ واقعی نشان دهیم آیا سیستم بهتر شده است یا نه.
به امید خدا، اول می‌سازیم، بعد آزمایش می‌کنیم و در نهایت فقط چیزی را ادعا می‌کنیم که بتوانیم نشان بدهی
from dataclasses import dataclass
from typing import Iterable


@dataclass
class Task:
    name: str
    kind: str
    priority: int = 1


@dataclass
class Device:
    name: str
    supported_tasks: tuple[str, ...]
    latency_ms: float
    energy_j: float
    transfer_ms: float = 0.0
    risk: float = 0.0
    available: bool = True


@dataclass
class Weights:
    latency: float = 1.0
    energy: float = 0.1
    transfer: float = 1.0
    risk: float = 100.0


def task_score(task: Task, device: Device,
               weights: Weights) -> float:
    return (
        weights.latency * device.latency_ms
        + weights.energy * device.energy_j
        + weights.transfer * device.transfer_ms
        + weights.risk * device.risk
    )


def choose_device(
    task: Task,
    devices: Iterable[Device],
    weights: Weights = Weights(),
) -> Device | None:
    candidates = [
        d for d in devices
        if d.available and task.kind in d.supported_tasks
    ]

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda d: task_score(task, d, weights),
    )


def filter_inputs(items: Iterable[str]) -> list[str]:
    """Remove empty and exact duplicate inputs only."""
    result = []
    seen = set()

    for item in items:
        value = item.strip()

        if not value or value in seen:
            continue

        seen.add(value)
        result.append(value)

    return result


if __name__ == "__main__":
    devices = [
        Device(
            name="local_cpu",
            supported_tasks=("text", "control"),
            latency_ms=30,
            energy_j=2.0,
            transfer_ms=0,
        ),
        Device(
            name="local_gpu",
            supported_tasks=("text", "vision"),
            latency_ms=12,
            energy_j=5.0,
            transfer_ms=3,
        ),
    ]

    task = Task(name="Example", kind="text")
    selected = choose_device(task, devices)

    print("Selected:", selected.name if selected else "No device")
    print("Inputs:", filter_inputs(
        ["hello", "hello", "  ", "world"]
    ))
    import unittest

from gpt_fast import (
    Device,
    Task,
    choose_device,
    filter_inputs,
)


class TestGPTFast(unittest.TestCase):

    def setUp(self):
        self.devices = [
            Device(
                "cpu", ("text",), 30, 2.0
            ),
            Device(
                "gpu", ("text",), 10, 5.0
            ),
        ]

    def test_selects_lowest_score(self):
        task = Task("T1", "text")
        result = choose_device(task, self.devices)
        self.assertEqual(result.name, "gpu")

    def test_unsupported_task(self):
        task = Task("T2", "audio")
        self.assertIsNone(choose_device(task, self.devices))

    def test_unavailable_device(self):
        devices = [
            Device("offline", ("text",), 1, 1, available=False)
        ]
        self.assertIsNone(
            choose_device(Task("T3", "text"), devices)
        )

    def test_filter_preserves_order(self):
        self.assertEqual(
            filter_inputs(["A", "A", " ", "B"]),
            ["A", "B"],
        )


if __name__ == "__main__":
    unittest.main()
    # GPT-Fast

A small experimental task-routing prototype.

## Goal

Choose a feasible execution device using estimated latency,
energy use, data-transfer overhead, and risk.

## Current features

- Device selection from supplied profiles
- Configurable scoring weights
- Basic duplicate and empty-input filtering
- Unit tests

## Limitations

- This is not a foundation language model.
- Device measurements are supplied manually.
- No real CPU/GPU benchmarking is implemented yet.
- No cloud, Azure, operating-system, or vehicle integration exists yet.
- The scheduler's decisions must be validated against real workloads.

## Requirements

Python 3.10 or newer. No external packages required.

## Run

    python gpt_fast.py

## Test

    python -m unittest -v

## Status

Prototype — experimental, not benchmark-validated.
# GPT-Fast: Mathematical Model

## 1. Feasible devices

For task i, define the feasible device set:

D_i = {d | available(d) and supports(d, i)}

If D_i is empty, return no device.

## 2. Estimated latency

L(i,d) = L_compute + L_queue
         + L_transfer + L_prepare + L_result

The current prototype uses a supplied latency estimate.
It does not measure these terms automatically.

## 3. Cost function

J(i,d) = a*L(i,d) + b*E(i,d)
         + c*T(i,d) + g*R(i,d)

Where:
- L = estimated latency
- E = estimated energy
- T = data-transfer overhead
- R = risk penalty
- a,b,c,g = configurable weights

## 4. Routing decision

d* = argmin J(i,d), for d in D_i

This minimizes the estimated score, not necessarily
real-world execution time.

## 5. Input filtering

Remove empty strings and exact duplicates.

Do not automatically discard unknown or unusual inputs.
They may contain important information.

## 6. Validation

Compare routing against a baseline using the same tasks.

Measure:
- End-to-end latency
- Energy consumption
- Output quality
- Failure rate
- Memory usage

Filtering is useful only if its total overhead is justified
by measured improvements.
