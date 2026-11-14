"""统一网格重采样：所有序列落到 1 分钟桶。

规则：
- 采样率高于桶（interval < 60s）→ 桶内 mean/max/min + 计数，source='agg'。
- 采样率低于桶（interval >= 60s）或该段上报间隔被拉长（降级）→ hold 沿用，carried=True。
  hold 只在「前后原始格相距 <= 3 格」且「前后采样点间距 > 1.5 桶」时发生；
  更长空缺留给缺口模块判定，1s 设备偶尔丢一格也不会被静默补值。
- 首格之前没有前值 → 不反向外推（保持空）。

时基标记：输入点可带第三个元素 basis（corrected/raw，来自分段校时）；
每格记录 time_basis：corrected（已校时）/ raw（原时刻）/ mixed（段交界同格混排）/
None（无原始点，由校时模块按桶归属补标）。hold 沿用格继承前值格的时基。
"""
from __future__ import annotations

from . import settings


def bucket_of(ts: float) -> int:
    return int(ts // settings.BUCKET_SECONDS)


def _bucket_for(point: tuple) -> int:
    """点的归桶：校正点按「最近分钟桶」归位（校正时刻含拟合残差，round 避免
    落在分钟边界上的点被亚秒级误差甩到相邻桶）；原始点按所在桶（floor）。"""
    if len(point) > 2 and point[2] == "corrected":
        return int((point[0] + settings.BUCKET_SECONDS / 2) // settings.BUCKET_SECONDS)
    return bucket_of(point[0])


def bucket_start_iso(bucket: int) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(bucket * settings.BUCKET_SECONDS, tz=_tz()).isoformat(timespec="seconds")


def _tz():
    from datetime import timedelta, timezone

    return timezone(timedelta(hours=8))


def _cell(bucket: int, **kwargs) -> dict:
    base = {
        "bucket": bucket,
        "bucket_start": bucket_start_iso(bucket),
        "value": None,
        "mean": None,
        "max": None,
        "min": None,
        "samples": 0,
        "source": "empty",   # agg | sample | hold | interpolated | empty
        "carried": False,
        "filled": False,     # 缺口插补或沿用后是否为「非原始」值
        "first_ts": None,
        "last_ts": None,
        "time_basis": None,  # corrected | raw | mixed | None（无原始点）
    }
    base.update(kwargs)
    return base


def _basis_of(samples: list[tuple]) -> str | None:
    """一格内原始点的时基：全 corrected → corrected；全 raw → raw；混排 → mixed。"""
    bases = {p[2] for p in samples if len(p) > 2}
    if not bases:
        return None
    if len(bases) == 1:
        return bases.pop()
    return "mixed"


def resample(
    points: list[tuple[float, float]],
    interval_s: int,
    start_bucket: int,
    end_bucket: int,
) -> list[dict]:
    """把一条序列重采样到 [start_bucket, end_bucket] 的 1 分钟格。

    points 元素为 (ts, value) 或 (ts, value, basis)；basis 仅用于时基标记。
    """
    grouped: dict[int, list[tuple]] = {}
    for point in points:
        b = _bucket_for(point)
        if b < start_bucket or b > end_bucket:
            continue
        grouped.setdefault(b, []).append(point)

    high_rate = interval_s < settings.BUCKET_SECONDS
    cells: list[dict] = []
    for bucket in range(start_bucket, end_bucket + 1):
        samples = grouped.get(bucket)
        if not samples:
            cells.append(_cell(bucket))
            continue
        values = [p[1] for p in samples]
        basis = _basis_of(samples)
        if high_rate:
            cells.append(
                _cell(
                    bucket,
                    first_ts=min(p[0] for p in samples),
                    last_ts=max(p[0] for p in samples),
                    value=sum(values) / len(values),
                    mean=sum(values) / len(values),
                    max=max(values),
                    min=min(values),
                    samples=len(values),
                    source="agg",
                    time_basis=basis,
                )
            )
        else:
            last_ts, last_value = max(samples, key=lambda p: p[0])[:2]
            cells.append(
                _cell(
                    bucket,
                    first_ts=min(p[0] for p in samples),
                    last_ts=last_ts,
                    value=last_value,
                    mean=last_value,
                    max=last_value,
                    min=last_value,
                    samples=len(samples),
                    source="sample",
                    time_basis=basis,
                )
            )
    _apply_hold(cells)
    return cells


def _apply_hold(cells: list[dict]) -> None:
    """低频/降级采样段：hold 沿用前值并计 carried。

    触发条件（两者同时满足）：
    - 前后原始格相距 <= HOLD_MAX_BRACKET_BUCKETS 格（更长的空缺交给缺口模块判定）；
    - 前后两个原始采样点的时间间距 > 1.5 桶（说明该段上报间隔确实稀疏，
      1s 设备偶尔丢一格时不静默补值）。
    """
    anchor_idx = [i for i, c in enumerate(cells) if c["source"] in ("agg", "sample")]
    if len(anchor_idx) < 2:
        return
    min_spacing_s = 1.5 * settings.BUCKET_SECONDS
    for prev, nxt in zip(anchor_idx, anchor_idx[1:]):
        distance = cells[nxt]["bucket"] - cells[prev]["bucket"]
        if distance <= 1 or distance > settings.HOLD_MAX_BRACKET_BUCKETS:
            continue
        prev_ts = cells[prev]["last_ts"]
        next_ts = cells[nxt]["first_ts"]
        if prev_ts is None or next_ts is None:
            continue
        if next_ts - prev_ts <= min_spacing_s:
            continue
        hold_value = cells[prev]["value"]
        for i in range(prev + 1, nxt):
            cells[i]["value"] = hold_value
            cells[i]["mean"] = hold_value
            cells[i]["max"] = hold_value
            cells[i]["min"] = hold_value
            cells[i]["source"] = "hold"
            cells[i]["carried"] = True
            cells[i]["filled"] = True
            cells[i]["time_basis"] = cells[prev]["time_basis"]  # 沿用前值的时基


def stats(cells: list[dict]) -> dict:
    counts = {"agg": 0, "sample": 0, "hold": 0, "empty": 0}
    for cell in cells:
        counts[cell["source"]] = counts.get(cell["source"], 0) + 1
    raw = counts["agg"] + counts["sample"]
    return {
        "cells": len(cells),
        "agg_cells": counts["agg"],
        "sample_cells": counts["sample"],
        "held_cells": counts["hold"],
        "empty_cells": counts["empty"],
        "raw_cells": raw,
        "carried_cells": counts["hold"],
    }
