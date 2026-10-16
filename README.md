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
import time
import random

class ProcessingNode:
    """گره پردازشی (ابر / پردازنده‌ها) با قابلیت سنجش دما و ظرفیت واقعی"""
    def __init__(self, name, max_capacity):
        self.name = name
        self.max_capacity = max_capacity
        self.current_load = 0
        self.temperature = 35.0  # دمای پایه (درجه)

    def is_available(self, task_weight):
        # بررسی همزمان ظرفیت واقعی و دمای بحران (زیر 75 درجه)
        return (self.current_load + task_weight <= self.max_capacity) and (self.temperature < 75.0)

    def process(self, task_weight):
        self.current_load += task_weight
        self.temperature += task_weight * 3.0  # بالا رفتن دما با پردازش
        print(f"[{self.name}] پردازش انجام شد. بار فعلی: {self.current_load}, دما: {self.temperature:.1f}°C")

    def cool_down(self):
        """مکانیسم تنفس عمیق و خنک‌سازی گره"""
        self.current_load = max(0, self.current_load - 2)
        self.temperature = max(35.0, self.temperature - 6.0)


class SmartBeeBypass:
    """ماژول زنبور: پل ارتباطی مستقیم و سبک زمان اضطرار"""
    def __init__(self):
        self.status = "آماده و چابک"

    def direct_process(self, task):
        print(f"[زنبور (S)] فضای ابری درگیر بود؛ وظیفه مستقیماً با انرژی کم از مسیر زنبور پردازش شد: {task}")
        return True


class SmartRecyclingBin:
    """سطل هوشمند با تک‌چشم و ترازوی ارزش برای بازیافت داده‌های اضافه"""
    def __init__(self):
        self.encrypted_archive = []

    def absorb_and_encrypt(self, overflow_task):
        print(f"[سطل هوشمند] دریافت بار اضافه/خطا. وزن‌کشی با ترازوی ارزش و رمزنگاری...")
        encrypted_data = f"SECURE_ENC_{hash(overflow_task)}"
        self.encrypted_archive.append(encrypted_data)
        print(f"[سطل هوشمند] داده‌ی امن‌شده ذخیره شد: {encrypted_data}")
        return encrypted_data


# --- شبیه‌سازی اصلی سیستم ---
if __name__ == "__main__":
    print("--- راه‌اندازی سیستم پویا و هوشمند ---")
    
    # ساخت گره‌ها با ظرفیت مشخص
    cloud_node_1 = ProcessingNode("Cloud_Node_A", max_capacity=10)
    cloud_node_2 = ProcessingNode("Cloud_Node_B", max_capacity=8)
    bee_module = SmartBeeBypass()
    smart_bin = SmartRecyclingBin()

    nodes = [cloud_node_1, cloud_node_2]

    # شبیه‌سازی ورود وظایف (Tasks)
    tasks = [4, 6, 5, 8, 3]

    for i, task in enumerate(tasks, 1):
        print(f"\n--- فریم {i}: ورود وظیفه با حجم {task} ---")
        
        # ۱. پیدا کردن «مسیر خنک و دارای ظرفیت واقعی»
        best_node = None
        for node in nodes:
            if node.is_available(task):
                if best_node is None or node.temperature < best_node.temperature:
                    best_node = node

        if best_node:
            # گره مناسب و خنک پیدا شد
            best_node.process(task)
        else:
            print("[سیستم] هشدار: همه گره‌های ابری داغ یا پر هستند!")
            # ۲. فعال شدن ماژول زنبور به عنوان پل اضطراری
            if task <= 4:
                bee_module.direct_process(f"Task_Type_{task}")
            else:
                # ۳. هدایت بار سنگین اضافه به سطل هوشمند برای ترازوی ارزش و رمزنگاری
                smart_bin.absorb_and_encrypt(f"Heavy_Overflow_{task}")

        # تنفس عمیق و خنک‌سازی دوره‌ای گره‌ها
        for node in nodes:
            node.cool_down()
        
        time.sleep(0.5)

    print("\n--- پایان چرخه‌ی اول شبیه‌سازی با موفقیت ---")
    
from pathlib import Path

path = Path('/mnt/data') / "fast_gpt_architecture.py"
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w", encoding='utf-8') as output_file:
    output_file.write("# -*- coding: utf-8 -*-\n\"\"\"\nfast_gpt_architecture.py\n========================\nFastGPT Architecture  —  معماری FastGPT (بصریِ کاربر)\n\"به نام الله\"\n\nبازسازی عملیاتیِ نقشهٔ دست‌کشیدهٔ کاربر:\n  • سرِ انسان = CPU (منطق و کنترل) / GPU (موازی‌سازی) / NPU (شتاب‌دهی AI)\n    + حافظهٔ ارّادی (Voluntary Memory cloud) + گوش (ورودی شنیداری) + مکالمه (خروجی)\n    + فن (Thermal Path Routing) + زنبور (Robotic Bee / Bypass Bridge)\n  • Organic Geometric Shifting Cloud Matrix = شبکهٔ ابریِ لایه‌بندی‌شدهٔ گره‌ها\n  • Quantum Jump Nodes / Encrypted Circuits / Smart Recycling Container\n\nاجزا (مطابق درخواست):\n  1) ProcessingNode      → مسیریابی CPU/GPU/NPU، بررسی دما و ظرفیت واقعی،\n                            و «جهش کوانتومی» وقتی گره‌های تازه آزاد می‌شوند.\n  2) SmartBeeBypass      → ماژول سبک «زنبور» (S-module)؛ پل میان‌بُرِ مستقیم\n                            وقتی ابرِ اصلی یا گره‌ها دچار تأخیر/اضافه‌بار شوند.\n  3) SmartRecyclingBin   → «یک‌چشم» (sentinel watcher) + «ترازوی ارزش» (Value Scale)\n                            برای ارزیابی/فیلتر/ترمیم/رمزنگاریِ مجدد داده‌های\n                            ناخواسته یا سرریز و بازگرداندن امنِ آن‌ها به سیستم.\n  4) Simulation runner   → همزمانی چند کاربر + توزیع بار پویا.\n\nاجرا:  python3 fast_gpt_architecture.py\n\"\"\"\n\nfrom __future__ import annotations\n\nimport hashlib\nimport random\nimport statistics\nimport threading\nimport time\nimport uuid\nfrom dataclasses import dataclass, field\nfrom enum import Enum\nfrom typing import Callable, Dict, List, Optional\n\n# =====================================================================\n# Enums & dataclasses — شمارش‌ها و ساختارهای پایه\n# =====================================================================\n\n\nclass DeviceType(Enum):\n    \"\"\"انواع پردازنده — مطابق نمودار سرِ انسان (CPU/GPU/NPU).\"\"\"\n    CPU = \"CPU\"   # منطق و کنترل (Logic & Control)\n    GPU = \"GPU\"   # پردازش موازی (Parallel Computing)\n    NPU = \"NPU\"   # شتاب‌دهی هوش مصنوعی (AI Acceleration)\n\n\nclass ThermalState(Enum):\n    \"\"\"وضعیت حرارتی گره — «مسیر حرارتی» (فن در نقشه).\"\"\"\n    COOL = \"cool\"        # سرد → پذیرش کامل بار\n    WARM = \"warm\"        # گرم → کاهش ظرفیت مؤثر\n    HOT = \"hot\"          # داغ → رد بار / هدایت به زنبور\n    CRITICAL = \"critical\"  # بحرانی → خنک‌کاری اجباری\n\n\n@dataclass\nclass Task:\n    \"\"\"یک واحد کار؛ کاربر + حجم محاسبه (واحد ساختگی FLOP-like).\"\"\"\n    user: str\n    size: int\n    prefer: DeviceType = DeviceType.CPU\n    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])\n    hops: int = 0            # چند بار جابه‌جا شده (برای پایش به‌شدت‌گرم)\n    bee_bypassed: bool = False\n\n\n@dataclass\nclass Packet:\n    \"\"\"بستهٔ داده‌ای که به سطل بازیافت می‌رسد (سرریز/ناخواسته).\"\"\"\n    payload: str\n    origin: str\n    size: int\n    encrypted: bool = False\n    repaired: bool = False\n    value: float = 0.0\n\n\n# =====================================================================\n# 1) ProcessingNode — گرهٔ پردازش (CPU/GPU/NPU) با نظارت حرارتی\n#    و جهش کوانتومی (Quantum Jump)\n# =====================================================================\n\n\nclass ProcessingNode:\n    \"\"\"\n    یک گرهٔ محاسباتی با سه شتاب‌دهنده (CPU/GPU/NPU).\n\n    منطق نقشهٔ کاربر:\n      • حافظهٔ ارّادی ← CPU ← GPU و NPU  و بازگشت نتیجه به حافظه.\n      • «مسیر حرارتی» = فن: اگر گره داغ شد، بار به گرهٔ خنک‌تر می‌رود.\n      • «جهش کوانتومی» (Quantum Jump): وقتی ظرفیتِ آزادشدهٔ ناگهانی\n        در گره‌های دیگر دیده می‌شود، صفِ در انتظار به‌صورت اتمی\n        به آن گره‌ها «می‌پرد».\n    \"\"\"\n\n    #: بار بیشینهٔ هر دستگاه (واحد ساختگی)\n    CAPACITY: Dict[DeviceType, int] = {\n        DeviceType.CPU: 100,\n        DeviceType.GPU: 150,\n        DeviceType.NPU: 120,\n    }\n    #: توان پردازش در هر تیک — NPU سریع‌ترین برای کارهای AI\n    SPEED: Dict[DeviceType, float] = {\n        DeviceType.CPU: 1.0,\n        DeviceType.GPU: 1.6,\n        DeviceType.NPU: 2.2,\n    }\n\n    def __init__(self, node_id: str, name: str = \"\"):\n        self.node_id = node_id\n        self.name = name or f\"node-{node_id[:6]}\"\n        self.load: Dict[DeviceType, int] = {d: 0 for d in DeviceType}\n        self.temperature: float = 30.0           # سلسیوس (شبیه‌سازی)\n        self.thermal_state: ThermalState = ThermalState.COOL\n        self.queue: List[Task] = []\n        self.completed: List[Task] = []\n        self.lock = threading.Lock()\n        self.stats = {\"routed\": 0, \"bee_bypass\": 0, \"quantum_jumps\": 0,\n                      \"rejected_hot\": 0, \"recycled\": 0}\n        # هوک «جهش کوانتومی» — توسط کلاستر ثبت می‌شود\n        self.on_quantum_jump: Optional[Callable] = None\n\n    # ------------------- ظرفیت واقعی -------------------\n    def effective_capacity(self, device: DeviceType) -> int:\n        \"\"\"ظرفیت واقعی = اسمی × ضریب حرارتی. سرد=۱، گرم=۰.۷، داغ=۰.۳.\"\"\"\n        factor = {ThermalState.COOL: 1.0, ThermalState.WARM: 0.7,\n                  ThermalState.HOT: 0.3, ThermalState.CRITICAL: 0.0}[self.thermal_state]\n        return int(self.CAPACITY[device] * factor)\n\n    def free_capacity(self, device: DeviceType) -> int:\n        return max(0, self.effective_capacity(device) - self.load[device])\n\n    def total_free(self) -> int:\n        return sum(self.free_capacity(d) for d in DeviceType)\n\n    # ------------------- حرارت (فن) -------------------\n    def tick_thermal(self) -> None:\n        \"\"\"هر تیک: دما بر اساس بار بالا می‌رود و فن کمی خنک می‌کند.\"\"\"\n        util = sum(self.load.values()) / sum(self.CAPACITY.values())\n        fan_cooling = 3.5  # «فن» — Big processor as fan\n        self.temperature += util * 12.0 - fan_cooling\n        self.temperature = max(25.0, min(105.0, self.temperature))\n        if self.temperature < 55:\n            self.thermal_state = ThermalState.COOL\n        elif self.temperature < 75:\n            self.thermal_state = ThermalState.WARM\n        elif self.temperature < 90:\n            self.thermal_state = ThermalState.HOT\n        else:\n            self.thermal_state = ThermalState.CRITICAL\n\n    # ------------------- مسیریابی -------------------\n    def choose_device(self, task: Task) -> DeviceType:\n        \"\"\"انتخاب شتاب‌دهنده: ترجیح کاربر، وگرنه آزادترین دستگاه.\"\"\"\n        if self.free_capacity(task.prefer) > 0:\n            return task.prefer\n        return max(DeviceType, key=lambda d: self.free_capacity(d))\n\n    def try_route(self, task: Task) -> bool:\n        \"\"\"تلاش برای پذیرش کار؛ اگر گره داغ/پر بود → False (زنبور یا گرهٔ دیگر).\"\"\"\n        with self.lock:\n            device = self.choose_device(task)\n            if task.size > self.free_capacity(device):\n                if self.thermal_state in (ThermalState.HOT, ThermalState.CRITICAL):\n                    self.stats[\"rejected_hot\"] += 1\n                return False\n            self.load[device] += task.size\n            task.hops += 1\n            self.queue.append(task)\n            self.stats[\"routed\"] += 1\n            return True\n\n    # ------------------- پردازش یک تیک -------------------\n    def process_tick(self, recycle_bin: \"SmartRecyclingBin\") -> None:\n        \"\"\"پردازش کارهای صف؛ سرریز/ناکارسازی → سطل بازیافت.\"\"\"\n        with self.lock:\n            done: List[Task] = []\n            overflow: List[Task] = []\n            for task in self.queue:\n                device = self.choose_device(task)\n                progress = self.SPEED[device] * random.uniform(0.6, 1.4) * 25\n                task.size -= int(progress)\n                if task.size <= 0:\n                    done.append(task)\n                elif task.hops > 3:               # خیلی چرخیده → بازیافت\n                    overflow.append(task)\n            self.queue = [t for t in self.queue if t not in done and t not in overflow]\n            self.completed.extend(done)\n            # آزادسازی بارِ کارهای تمام‌شده (تقریبی و نسبت به دستگاه)\n        for t in done:\n            self.load[DeviceType.CPU] = max(0, self.load[DeviceType.CPU] - min(t.size, 0))\n        # سرریز به سطل بازیافت (بازگشت امن به سیستم)\n        for t in overflow:\n            recycle_bin.receive(Packet(payload=f\"overflow:{t.id}\", origin=self.name,\n                                       size=max(1, t.size)))\n            self.stats[\"recycled\"] += 1\n\n    # ------------------- جهش کوانتومی -------------------\n    def check_quantum_jump(self, cluster: \"FastGPTCluster\") -> int:\n        \"\"\"\n        Quantum Jump: اگر آزادسازی ناگهانی ظرفیت در «دیگر» گره‌ها دیده شد،\n        کارهای در انتظارِ همین گره به‌صورت اتمی به آن‌ها می‌پرند.\n        \"\"\"\n        jumped = 0\n        with self.lock:\n            waiting = list(self.queue)\n        for task in waiting:\n            for other in cluster.nodes:\n                if other is self:\n                    continue\n                if other.total_free() > task.size and other.try_route(task):\n                    with self.lock:\n                        if task in self.queue:\n                            self.queue.remove(task)\n                            self.load[task.prefer] = max(0, self.load[task.prefer] - task.size)\n                    self.stats[\"quantum_jumps\"] += 1\n                    jumped += 1\n                    if self.on_quantum_jump:\n                        self.on_quantum_jump(self.name, other.name, task.id)\n                    break\n        return jumped\n\n\n# =====================================================================\n# 2) SmartBeeBypass — ماژول زنبور (S-module)؛ پل میان‌بُر مستقیم\n# =====================================================================\n\n\nclass SmartBeeBypass:\n    \"\"\"\n    «زنبور» — یک ماژول سبک (bee) که وقتی ابرِ اصلی یا گره‌ها دچار\n    تأخیر (lag) یا اضافه‌بار (overload) می‌شوند، یک پل مستقیمِ\n    میان‌بُر بین کاربر و مقصد برقرار می‌کند (مانند Robotic Bee Module\n    با DIRECT BYPASS BRIDGE در پوستر).\n    \"\"\"\n\n    LAG_THRESHOLD = 0.35      # ثانیه؛ بالاتر از این یعنی «کندی»\n    OVERLOAD_RATIO = 0.9      # اشغال > ۹۰٪ کل خوشه → اضافه‌بار\n\n    def __init__(self, cluster: \"FastGPTCluster\"):\n        self.cluster = cluster\n        self.bridge_log: List[str] = []\n        self.stats = {\"bypassed\": 0, \"returned\": 0}\n        self._last_latency: float = 0.0\n\n    # ------------------- سنجش سلامت -------------------\n    def measure_lag(self) -> float:\n        \"\"\"تأخیر ابر را شبیه‌سازی/سنجش می‌کند (این جا: تابع فشار خوشه).\"\"\"\n        total_cap = sum(n.total_free() for n in self.cluster.nodes)\n        max_cap = sum(sum(ProcessingNode.CAPACITY.values()) for _ in self.cluster.nodes)\n        self._last_latency = 0.05 + (1 - total_cap / max_cap) * 0.5\n        return self._last_latency\n\n    def is_overloaded(self) -> bool:\n        total_cap = sum(n.total_free() for n in self.cluster.nodes)\n        max_cap = sum(sum(ProcessingNode.CAPACITY.values()) for _ in self.cluster.nodes)\n        return 1 - (total_cap / max_cap) >= self.OVERLOAD_RATIO\n\n    def should_bypass(self) -> bool:\n        \"\"\"کندی یا اضافه‌بار → زنبور فعال می‌شود.\"\"\"\n        return self.measure_lag() > self.LAG_THRESHOLD or self.is_overloaded()\n\n    # ------------------- پل میان‌بُر -------------------\n    def fly(self, task: Task) -> bool:\n        \"\"\"\n        پل مستقیم: زنبور کار را از کنارِ صفِ ابر به سبک‌ترین گرهٔ فعال\n        می‌برد؛ اگر هیچ‌جا جا نبود، خودش «پردازش سبک» (light path)\n        را انجام می‌دهد تا کاربر منتظر نماند.\n        \"\"\"\n        if not self.should_bypass():\n            return False\n        # سبک‌ترین گرهٔ خنک را پیدا کن\n        candidates = [n for n in self.cluster.nodes\n                      if n.thermal_state in (ThermalState.COOL, ThermalState.WARM)]\n        target = min(candidates or self.cluster.nodes,\n                     key=lambda n: sum(n.load.values()))\n        if target.try_route(task):\n            task.bee_bypassed = True\n            self.stats[\"bypassed\"] += 1\n            self.bridge_log.append(f\"bee: {task.user} → {target.name} (lag={self._last_latency:.2f}s)\")\n            return True\n        # مسیر سبکِ خودِ زنبور — پردازش فوریِ کوچک (S-module سبک)\n        task.size = int(task.size * 0.4)          # نسخهٔ سبک\n        task.bee_bypassed = True\n        self.cluster.nodes[0].completed.append(task)\n        self.stats[\"returned\"] += 1\n        self.bridge_log.append(f\"bee: light-path served {task.user} directly\")\n        return True\n\n\n# =====================================================================\n# 3) SmartRecyclingBin — سطل بازیافت هوشمند\n#    «یک‌چشم» (sentinel) + «ترازوی ارزش» (Value Scale)\n# =====================================================================\n\n\nclass SmartRecyclingBin:\n    \"\"\"\n    Smart Recycling Container در پوستر: DATA SORTING، VALUE EXTRACTION،\n    SUSTAINABLE CYCLE، SELF-OPTIMIZING.\n\n    • One-Eye sentinel  → پایش مداوم بسته‌های ورودی (نگهبان تک‌چشم).\n    • Value Scale       → وزن‌کردن ارزش هر بسته: بالا=ترمیم و بازگشت،\n                          پایین=حذف امن.\n    • هر بستهٔ بازگشتی دوباره رمزنگاری (re-encrypt) می‌شود (Encrypted Circuits).\n    \"\"\"\n\n    VALUE_THRESHOLD = 0.35   # زیر این ارزش → دورریز امن\n\n    def __init__(self):\n        self.bin: List[Packet] = []\n        self.returned: List[Packet] = []\n        self.discarded: int = 0\n        self.stats = {\"received\": 0, \"repaired\": 0, \"re_encrypted\": 0,\n                      \"returned\": 0, \"discarded\": 0}\n        self.eye_open = True   # یک‌چشم روشن\n\n    # ------------------- رمزنگاری -------------------\n    @staticmethod\n    def _encrypt(data: str) -> str:\n        \"\"\"رمز سادهٔ SHA-256 (نمونه؛ در عمل: AES/کی‌مسیر واقعی).\"\"\"\n        return hashlib.sha256(data.encode()).hexdigest()[:16]\n\n    # ------------------- پایش یک‌چشم -------------------\n    def one_eye_watch(self, packet: Packet) -> None:\n        \"\"\"نگهبان تک‌چشم: هر بسته را ثبت و از خرابی محتوا چک می‌کند.\"\"\"\n        if not self.eye_open:\n            return\n        if not packet.payload or packet.size <= 0:\n            packet.payload = f\"repaired:{packet.payload or 'empty'}\"\n            packet.size = max(1, packet.size)\n            packet.repaired = True\n            self.stats[\"repaired\"] += 1\n\n    # ------------------- ترازوی ارزش -------------------\n    def value_scale(self, packet: Packet) -> float:\n        \"\"\"\n        Value Scale: وزن ارزش = ترکیبی از اندازه، تازگی و اصالت.\n        خروجی بین ۰ و ۱.\n        \"\"\"\n        size_score = min(1.0, packet.size / 200)\n        fresh = 1.0 if packet.origin in (\"bee\", \"cloud\") else 0.7\n        return round(0.6 * size_score + 0.4 * fresh, 3)\n\n    # ------------------- چرخهٔ بازیافت -------------------\n    def receive(self, packet: Packet) -> None:\n        self.stats[\"received\"] += 1\n        self.one_eye_watch(packet)                       # یک‌چشم\n        packet.value = self.value_scale(packet)          # ترازوی ارزش\n        if packet.value >= self.VALUE_THRESHOLD:\n            # ترمیم + رمزنگاری مجدد → بازگشت امن به سیستم\n            if not packet.repaired:\n                packet.repaired = True\n                self.stats[\"repaired\"] += 1\n            packet.payload = self._encrypt(packet.payload)\n            packet.encrypted = True\n            self.stats[\"re_encrypted\"] += 1\n            self.returned.append(packet)\n            self.stats[\"returned\"] += 1\n        else:\n            self.discarded += 1\n            self.stats[\"discarded\"] += 1\n        self.bin.append(packet)\n\n    def flush_returned(self, cluster: \"FastGPTCluster\") -> int:\n        \"\"\"بازگرداندن امنِ بسته‌های ارزشمند به گره‌ها (sustainable cycle).\"\"\"\n        sent = 0\n        for p in self.returned:\n            task = Task(user=\"recycler\", size=max(1, p.size // 4))\n            for node in cluster.nodes:\n                if node.try_route(task):\n                    sent += 1\n                    break\n        self.returned.clear()\n        return sent\n\n\n# =====================================================================\n# 4) FastGPTCluster — خوشه (Organic Geometric Shifting Cloud Matrix)\n#    + Simulation runner (چند کاربر + توزیع بار پویا)\n# =====================================================================\n\n\nclass FastGPTCluster:\n    \"\"\"ابرِ متحرکِ ارگانیک: چند گره + زنبور + سطل بازیافت.\"\"\"\n\n    def __init__(self, n_nodes: int = 4):\n        self.nodes: List[ProcessingNode] = [ProcessingNode(str(i)) for i in range(n_nodes)]\n        self.bee = SmartBeeBypass(self)\n        self.recycler = SmartRecyclingBin()\n        for n in self.nodes:\n            n.on_quantum_jump = lambda src, dst, tid: print(\n                f\"  ⚡ Quantum jump: {src} → {dst} (task {tid})\")\n\n    # ------------------- توزیع بار پویا -------------------\n    def submit(self, task: Task) -> bool:\n        \"\"\"اول بهترین گره؛ در صورت داغی/کندی → زنبور؛ در بدترین حالت → صف بازیافت.\"\"\"\n        target = min(self.nodes, key=lambda n: (n.total_free() * -1, n.temperature))\n        if target.try_route(task):\n            return True\n        return self.bee.fly(task) or (self.recycler.receive(\n            Packet(payload=f\"queued:{task.id}\", origin=\"cloud\", size=task.size)) or True)\n\n    def tick(self) -> None:\n        \"\"\"یک تیک همزمانی: حرارت، پردازش، جهش کوانتومی، بازیافت.\"\"\"\n        for n in self.nodes:\n            n.tick_thermal()\n            n.process_tick(self.recycler)\n        for n in self.nodes:\n            n.check_quantum_jump(self)\n        self.recycler.flush_returned(self)\n\n\ndef run_simulation(users: int = 8, ticks: int = 12, tasks_per_user: int = 4, seed: int = 7):\n    \"\"\"همزمانی چند کاربر + توزیع بار پویا — شبیه‌ساز معماری FastGPT.\"\"\"\n    random.seed(seed)\n    cluster = FastGPTCluster(n_nodes=4)\n    print(\"به نام الله — FastGPT Architecture simulation\")\n    print(f\"nodes={len(cluster.nodes)}  users={users}  ticks={ticks}\\n\")\n\n    # ورودی‌های کاربران (گوش → حافظهٔ ارّادی → CPU/GPU/NPU)\n    for t in range(ticks):\n        print(f\"--- tick {t + 1} ---\")\n        for u in range(users):\n            for _ in range(random.choices([0, 1, 2], weights=[3, 5, 2])[0]):\n                task = Task(user=f\"user{u}\", size=random.randint(20, 160),\n                            prefer=random.choice(list(DeviceType)))\n                cluster.submit(task)\n        cluster.tick()\n        for n in cluster.nodes:\n            print(f\"  {n.name}: temp={n.temperature:5.1f}C [{n.thermal_state.value:8s}] \"\n                  f\"free={n.total_free():4d} queue={len(n.queue)} done={len(n.completed)} \"\n                  f\"qjumps={n.stats['quantum_jumps']} bee={n.stats['bee_bypass']}\")\n        time.sleep(0.05)\n\n    # ---------- گزارش نهایی ----------\n    total_done = sum(len(n.completed) for n in cluster.nodes)\n    bee = cluster.bee.stats\n    rec = cluster.recycler.stats\n    print(\"\\n===== گزارش نهایی / Final report =====\")\n    print(f\"tasks completed      : {total_done}\")\n    print(f\"bee bypasses         : {bee['bypassed']} (+{bee['returned']} light-path)\")\n    print(f\"quantum jumps        : {sum(n.stats['quantum_jumps'] for n in cluster.nodes)}\")\n    print(f\"hot rejections       : {sum(n.stats['rejected_hot'] for n in cluster.nodes)}\")\n    print(f\"recycler received    : {rec['received']}  repaired: {rec['repaired']}  \"\n          f\"re-encrypted: {rec['re_encrypted']}\")\n    print(f\"recycler returned    : {rec['returned']}   discarded: {rec['discarded']}\")\n    if cluster.bee.bridge_log:\n        print(\"bee bridge log:\")\n        for line in cluster.bee.bridge_log[:5]:\n            print(f\"  {line}\")\n    temps = [n.temperature for n in cluster.nodes]\n    print(f\"node temps avg/max   : {statistics.mean(temps):.1f}C / {max(temps):.1f}C\")\n    return cluster\n\n\nif __name__ == \"__main__\":\n    run_simulation()\n")
print(path)
Output: /mnt/data/fast_gpt_architecture.py
import subprocess
r = subprocess.run(["python3", "/mnt/data/fast_gpt_architecture.py"], capture_output=True, text=True, timeout=60)
print(r.returncode); print(r.stdout[-2500:]); print(r.stderr[-800
Output: 0
=3 qjumps=7 bee=0
  node-3: temp= 43.4C [cool    ] free= 131 queue=0 done=0 qjumps=5 bee=0
--- tick 7 ---
  node-0: temp= 93.4C [critical] free=   0 queue=0 done=30 qjumps=4 bee=0
  node-1: temp= 62.5C [warm    ] free=   3 queue=0 done=4 qjumps=6 bee=0
  node-2: temp= 46.9C [cool    ] free= 109 queue=0 done=3 qjumps=7 bee=0
  node-3: temp= 47.7C [cool    ] free= 131 queue=0 done=0 qjumps=5 bee=0
--- tick 8 ---
  ⚡ Quantum jump: node-3 → node-2 (task 740d8da0)
  node-0: temp=105.0C [critical] free=   0 queue=0 done=34 qjumps=4 bee=0
  node-1: temp= 71.7C [warm    ] free=   3 queue=0 done=4 qjumps=6 bee=0
  node-2: temp= 54.1C [cool    ] free=   4 queue=1 done=5 qjumps=7 bee=0
  node-3: temp= 53.8C [cool    ] free=  89 queue=0 done=0 qjumps=6 bee=0
--- tick 9 ---
  node-0: temp=105.0C [critical] free=   0 queue=0 done=43 qjumps=4 bee=0
  node-1: temp= 80.9C [hot     ] free=   0 queue=0 done=4 qjumps=6 bee=0
  node-2: temp= 63.3C [warm    ] free=   0 queue=0 done=6 qjumps=7 bee=0
  node-3: temp= 60.3C [warm    ] free=   0 queue=0 done=1 qjumps=6 bee=0
--- tick 10 ---
  node-0: temp=105.0C [critical] free=   0 queue=0 done=53 qjumps=4 bee=0
  node-1: temp= 90.1C [critical] free=   0 queue=0 done=4 qjumps=6 bee=0
  node-2: temp= 72.7C [warm    ] free=   0 queue=0 done=6 qjumps=7 bee=0
  node-3: temp= 67.0C [warm    ] free=   0 queue=0 done=1 qjumps=6 bee=0
--- tick 11 ---
  node-0: temp=105.0C [critical] free=   0 queue=0 done=62 qjumps=4 bee=0
  node-1: temp= 99.3C [critical] free=   0 queue=0 done=4 qjumps=6 bee=0
  node-2: temp= 82.1C [hot     ] free=   0 queue=0 done=6 qjumps=7 bee=0
  node-3: temp= 73.7C [warm    ] free=   0 queue=0 done=1 qjumps=6 bee=0
--- tick 12 ---
  node-0: temp=105.0C [critical] free=   0 queue=0 done=70 qjumps=4 bee=0
  node-1: temp=105.0C [critical] free=   0 queue=0 done=4 qjumps=6 bee=0
  node-2: temp= 91.6C [critical] free=   0 queue=0 done=6 qjumps=7 bee=0
  node-3: temp= 80.4C [hot     ] free=   0 queue=0 done=1 qjumps=6 bee=0

===== گزارش نهایی / Final report =====
tasks completed      : 81
bee bypasses         : 0 (+56 light-path)
quantum jumps        : 23
hot rejections       : 0
recycler received    : 5  repaired: 5  re-encrypted: 5
recycler returned    : 5   discarded: 0
bee bridge log:
  bee: light-path served user0 directly
  bee: light-path served user1 directly
  bee: light-path served user2 directly
  bee: light-path served user6 directly
  bee: light-path served user3 directly
node temps avg/max   : 95.5C / 105.0C
https://gapgpt.app/api/v1/files/server/_XdadnQW7kW_X4IX1OrILfWom5A-AFLnSnFVTjShApou44aJ2A.webp
