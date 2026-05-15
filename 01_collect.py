# ============================================================
# 01_collect.py
# 本地LLM推理性能数据采集
# 正交表L9(3^4)，每组重复20次，共180条记录
# ============================================================

import time
import os
import csv
import threading
import psutil
import numpy as np
import pandas as pd
from llama_cpp import Llama

# ─────────────────────────────────────────
# 0. 路径配置
# ─────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "results.csv")

MODEL_PATHS = {
    "Q6K":  os.path.join(MODEL_DIR, "qwen2.5-3b-instruct-q6_k.gguf"),
    "Q4KM": os.path.join(MODEL_DIR, "qwen2.5-3b-instruct-q4_k_m.gguf"),
    "Q2K":  os.path.join(MODEL_DIR, "qwen2.5-3b-instruct-q2_k.gguf"),
}

# ─────────────────────────────────────────
# 1. 正交表 L9(3^4)
# ─────────────────────────────────────────
ORTHOGONAL_TABLE = [
    [0, 0, 0, 0],
    [0, 1, 1, 1],
    [0, 2, 2, 2],
    [1, 0, 1, 2],
    [1, 1, 2, 0],
    [1, 2, 0, 1],
    [2, 0, 2, 1],
    [2, 1, 0, 2],
    [2, 2, 1, 0],
]

LEVEL_A = ["Q6K", "Q4KM", "Q2K"]
LEVEL_B = [1, 2, 4]
LEVEL_C = [10, 40, 70]
LEVEL_D = [0, 2, 4]

REPEATS    = 20
MAX_TOKENS = 20
PROMPT     = "请用一句话解释什么是机器学习。"
COOL_DOWN  = 0.2   # 每次推理后冷却秒数

# ─────────────────────────────────────────
# 2. 获取CPU核心数（动态配置线程）
# ─────────────────────────────────────────
PHYSICAL_CORES = psutil.cpu_count(logical=False) or 4
LOGICAL_CORES  = psutil.cpu_count(logical=True)  or 4

# LLM使用物理核心数，最少4最多保留2个给系统
LLM_THREADS    = max(4, PHYSICAL_CORES - 2)
# 压力线程使用逻辑核心的一半
STRESS_CORES   = max(1, LOGICAL_CORES // 2)

print(f"[系统] 物理核心: {PHYSICAL_CORES} | 逻辑核心: {LOGICAL_CORES}")
print(f"[配置] LLM线程: {LLM_THREADS} | 压力核心: {STRESS_CORES}")

# ─────────────────────────────────────────
# 3. CPU负载施压工具
# ─────────────────────────────────────────
class CPULoadGenerator:

    def __init__(self):
        self._stop_event = threading.Event()
        self._threads    = []

    def start(self, target_pct: int):
        self.stop()
        if target_pct <= 0:
            return

        n_cores = STRESS_CORES
        per_core_pct = min(target_pct / n_cores, 95) / 100

        self._stop_event.clear()
        self._threads = []
        for _ in range(n_cores):
            t = threading.Thread(
                target=self._burn,
                args=(per_core_pct,),
                daemon=True
            )
            t.start()
            self._threads.append(t)
        time.sleep(1.5)   # 等待负载稳定

    def _burn(self, duty_cycle: float):
        interval = 0.01
        while not self._stop_event.is_set():
            end_busy = time.perf_counter() + interval * duty_cycle
            while time.perf_counter() < end_busy:
                pass
            sleep_time = interval * (1 - duty_cycle)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=1)
        self._threads = []

# ─────────────────────────────────────────
# 4. 内存压力工具
# ─────────────────────────────────────────
class MemoryPressure:

    def __init__(self):
        self._blocks = []

    def apply(self, gb: float):
        self.release()
        if gb <= 0:
            return
        chunk_bytes = 256 * 1024 * 1024   # 256MB每块
        n_chunks    = int(gb * 1024 / 256)
        print(f"    [内存] 申请 {gb}GB...", end="", flush=True)
        for _ in range(n_chunks):
            block = bytearray(chunk_bytes)
            for i in range(0, len(block), 4096):
                block[i] = 1
            self._blocks.append(block)
        print(f" 完成")

    def release(self):
        self._blocks.clear()

# ─────────────────────────────────────────
# 5. 单次推理
# ─────────────────────────────────────────
def run_single_inference(llm: Llama, batch_size: int) -> float:
    if batch_size == 1:
        t0 = time.perf_counter()
        llm(PROMPT, max_tokens=MAX_TOKENS,
            echo=False, temperature=0.0)
        t1 = time.perf_counter()
        time.sleep(COOL_DOWN)
        return (t1 - t0) * 1000
    else:
        t0 = time.perf_counter()
        for _ in range(batch_size):
            llm(PROMPT, max_tokens=MAX_TOKENS,
                echo=False, temperature=0.0)
        t1 = time.perf_counter()
        time.sleep(COOL_DOWN)
        return (t1 - t0) * 1000 / batch_size

# ─────────────────────────────────────────
# 6. 主采集流程
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("  本地LLM推理性能采集")
    print(f"  L9(3^4) 正交表，9组合 × {REPEATS}次 = {9*REPEATS}条记录")
    print("=" * 60)

    # 检查模型文件
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"[错误] 找不到模型：{path}")
            return
        size_gb = os.path.getsize(path) / 1024**3
        print(f"  ✓ {name}: {size_gb:.1f} GB")

    cpu_gen = CPULoadGenerator()
    mem_gen = MemoryPressure()

    all_records        = []
    current_model_name = None
    llm                = None

    total_start = time.time()

    try:
        for combo_idx, row in enumerate(ORTHOGONAL_TABLE):
            a_idx, b_idx, c_idx, d_idx = row

            model_name = LEVEL_A[a_idx]
            batch_size = LEVEL_B[b_idx]
            cpu_load   = LEVEL_C[c_idx]
            mem_gb     = LEVEL_D[d_idx]

            combo_start = time.time()
            elapsed     = (combo_start - total_start) / 60

            print(f"\n{'─'*60}")
            print(f"  组合 {combo_idx+1}/9 | "
                  f"精度={model_name} | Batch={batch_size} | "
                  f"CPU={cpu_load}% | 内存={mem_gb}GB")
            print(f"  已用时: {elapsed:.1f} 分钟")
            print(f"{'─'*60}")

            # ── 切换模型 ──
            if model_name != current_model_name:
                # 先停止压力，再切换模型
                cpu_gen.stop()
                mem_gen.release()

                if llm is not None:
                    del llm
                    llm = None
                    time.sleep(2)   # 等待内存释放

                print(f"  [模型] 加载 {model_name}...", flush=True)
                load_t0 = time.time()
                llm = Llama(
                    model_path        = MODEL_PATHS[model_name],
                    n_ctx             = 1024,
                    n_threads         = LLM_THREADS,
                    n_threads_batch   = LLM_THREADS,
                    n_gpu_layers      = 0,
                    verbose           = False,
                )
                load_t1 = time.time()
                current_model_name = model_name
                print(f"  [模型] 加载完成，耗时 {load_t1-load_t0:.1f}s")

                # 预热2次，不计入结果
                print(f"  [预热] 执行2次预热...", end="", flush=True)
                for _ in range(2):
                    llm(PROMPT, max_tokens=MAX_TOKENS,
                        echo=False, temperature=0.0)
                print(" 完成")

            # ── 施加环境压力 ──
            cpu_gen.stop()
            mem_gen.release()
            time.sleep(0.5)

            mem_gen.apply(mem_gb)
            cpu_gen.start(cpu_load)

            actual_cpu = psutil.cpu_percent(interval=1)
            actual_mem = psutil.virtual_memory()
            print(f"  [状态] CPU: {actual_cpu:.1f}% | "
                  f"内存已用: {actual_mem.percent:.1f}%")

            # ── 20次重复采集 ──
            latencies = []
            print(f"  [采集] ", end="", flush=True)

            for rep in range(REPEATS):
                lat_ms = run_single_inference(llm, batch_size)
                latencies.append(lat_ms)
                print(f".", end="", flush=True)

            print(f" 完成")

            # ── 统计量 ──
            lat_arr  = np.array(latencies)
            mean_lat = float(np.mean(lat_arr))
            std_lat  = float(np.std(lat_arr, ddof=1))
            cv       = (std_lat / mean_lat * 100) if mean_lat > 0 else 0.0

            combo_elapsed = (time.time() - combo_start) / 60
            print(f"  [结果] 均值={mean_lat:.1f}ms | "
                  f"SD={std_lat:.1f}ms | "
                  f"CV={cv:.2f}% | "
                  f"组合耗时={combo_elapsed:.1f}min")

            # ── 写入记录 ──
            for rep, lat_ms in enumerate(latencies):
                all_records.append({
                    "combo":         combo_idx + 1,
                    "rep":           rep + 1,
                    "A_quant":       model_name,
                    "B_batch":       batch_size,
                    "C_cpu_pct":     cpu_load,
                    "D_mem_gb":      mem_gb,
                    "y1_latency_ms": round(lat_ms, 3),
                    "y2_cv_pct":     round(cv, 4),
                })

            # 每个组合结束后实时保存（防止意外中断丢数据）
            df_tmp = pd.DataFrame(all_records)
            df_tmp.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            print(f"  [保存] 已写入 {len(df_tmp)} 条记录")

    finally:
        cpu_gen.stop()
        mem_gen.release()
        if llm is not None:
            del llm

    # ── 最终保存与汇总 ──
    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    total_min = (time.time() - total_start) / 60
    print(f"\n{'='*60}")
    print(f"  采集完成！共 {len(df)} 条记录")
    print(f"  总耗时：{total_min:.1f} 分钟")
    print(f"  输出：{OUTPUT_CSV}")
    print(f"{'='*60}")

    summary = df.groupby(
        ["A_quant", "B_batch", "C_cpu_pct", "D_mem_gb"]
    ).agg(
        mean_lat=("y1_latency_ms", "mean"),
        cv=("y2_cv_pct", "first")
    ).reset_index()

    print("\n各组合汇总：")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()