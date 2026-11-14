"""分段校时：把漂移估计的 (a, b) 真正落到时间轴上，再进入网格/缺口/摘要/覆盖率。

规则（与 README「核心规则」一致）：
1. 仅当漂移判定成立（|offset| > 阈值）时才校时；基准设备与时钟正常的序列原样通过，
   以单段「无需校时」标注，保证每条序列都有明确的分段结论。
2. 按「锚点是否充分」把窗口切成连续时段：相邻内点锚点（含窗口首/尾到最近锚点）
   间距 <= CORRECT_MAX_ANCHOR_GAP_S 的时段 → 锚点充分 → 按 t_ref = a·t_dev + b 校正；
   更长的空档（断采、降级上报、信号平坦）→ 锚点不足 → 保留设备原时刻并明确标注。
3. 段与段交界不重不漏：设备时间轴上的分段无缝拼接，每个原始点恰好落入一段；
   网格每一格也恰好归属一段（按格中点落在哪段的参考时间范围判定，段缝取最近段），
   全部段一起发布为同一次运行结果。
4. 校正后时刻裁剪进数据窗口（仅影响窗口边缘亚秒级越界，防止首/末点掉出网格）。

纯函数、无随机性：同一输入重跑得到完全一致的分段与网格。
"""
from __future__ import annotations

from . import grid, settings

# 时段状态
CORRECTED = "corrected"   # 已按 a·t_dev + b 校时
RAW = "raw"               # 锚点不足 / 无需校时，保留设备原时刻


def device_segments(
    anchor_times: list[float],
    span_s: float,
    max_gap_s: float = settings.CORRECT_MAX_ANCHOR_GAP_S,
) -> list[dict]:
    """按锚点充分度把 [0, span_s]（设备时钟，相对窗口起点）切成连续时段。

    返回的分段依次首尾相接、并集恰好覆盖整个窗口（不重不漏）；
    校正段与保留段必然交替。每段记录 anchors（段内锚点数）与成因。
    """
    anchors = sorted(t for t in anchor_times if 0.0 <= t <= span_s)
    cuts = [0.0] + anchors + [span_s]
    atoms: list[dict] = []
    for start, end in zip(cuts, cuts[1:]):
        if end - start <= 0:
            continue
        gap = end - start
        atoms.append({"start_s": start, "end_s": end, "corrected": gap <= max_gap_s, "gap_s": gap})
    if not atoms:
        atoms = [{"start_s": 0.0, "end_s": span_s, "corrected": False, "gap_s": span_s}]

    # 合并相邻同状态的原子段（校正段之间只隔着锚点，必被并成一段）
    merged: list[dict] = []
    for atom in atoms:
        if merged and merged[-1]["corrected"] == atom["corrected"]:
            merged[-1]["end_s"] = atom["end_s"]
            merged[-1]["gap_s"] = max(merged[-1]["gap_s"], atom["gap_s"])
        else:
            merged.append(dict(atom))

    for seg in merged:
        if seg["corrected"]:
            seg["anchors"] = sum(1 for a in anchors if seg["start_s"] <= a <= seg["end_s"])
            seg["reason"] = (
                f"锚点充分（{seg['anchors']} 个，相邻间距 ≤ {max_gap_s:g}s），按 t_ref = a·t_dev + b 校时"
            )
        else:
            seg["anchors"] = sum(1 for a in anchors if seg["start_s"] < a < seg["end_s"])
            seg["reason"] = (
                f"相邻锚点间距 {seg['gap_s']:.0f}s > 上限 {max_gap_s:g}s，锚点不足，保留设备原时刻"
            )
    return merged


def _segment_at(segments: list[dict], rel_s: float) -> dict:
    """设备时刻 rel_s 所属的段（分段不重不漏；越界时刻钳到首/末段）。"""
    for seg in segments:
        if seg["start_s"] <= rel_s < seg["end_s"]:
            return seg
    if rel_s < segments[0]["start_s"]:
        return segments[0]
    return segments[-1]


def _ref_ranges(segments: list[dict], a: float, b: float) -> list[tuple[float, float, dict]]:
    """各段在参考时间轴上的范围：校正段 [s,e) → [a·s+b, a·e+b)，保留段恒等映射。"""
    ranges = []
    for seg in segments:
        if seg["corrected"]:
            ranges.append((a * seg["start_s"] + b, a * seg["end_s"] + b, seg))
        else:
            ranges.append((seg["start_s"], seg["end_s"], seg))
    return ranges


def _segment_for_ref(rel_s: float, ref_ranges: list[tuple[float, float, dict]]) -> dict:
    """参考时刻 rel_s 所属的段：精确包含优先，段缝（< 1 桶）取时间上最近的段。"""
    best = None
    best_dist = None
    for start, end, seg in ref_ranges:
        if start <= rel_s < end:
            return seg
        dist = min(abs(rel_s - start), abs(rel_s - end))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = seg
    return best


def correct_points(
    points: list[tuple[float, float]],
    window_start: float,
    a: float,
    b: float,
    segments: list[dict],
    span_s: float,
) -> list[tuple[float, float, str]]:
    """逐点校时：校正段 → t_ref = a·t_dev + b（裁剪进窗口）；保留段 → 原时刻。

    返回 (ts, value, basis) 三元组，basis ∈ {corrected, raw}，供网格标记每格时基。
    """
    out: list[tuple[float, float, str]] = []
    for ts, value in points:
        rel = ts - window_start
        seg = _segment_at(segments, rel)
        if seg["corrected"]:
            corrected = a * rel + b
            corrected = min(max(corrected, 0.0), span_s - 1e-6)
            out.append((window_start + corrected, value, CORRECTED))
        else:
            out.append((ts, value, RAW))
    return out


def bucket_labels(
    segments: list[dict],
    a: float,
    b: float,
    start_bucket: int,
    end_bucket: int,
) -> list[str]:
    """每个网格桶的时基标签：按桶中点落在哪段的「参考时间范围」判定（不重不漏）。"""
    ref_ranges = _ref_ranges(segments, a, b)
    labels: list[str] = []
    for bucket in range(start_bucket, end_bucket + 1):
        mid_rel = (bucket - start_bucket) * settings.BUCKET_SECONDS + settings.BUCKET_SECONDS / 2
        seg = _segment_for_ref(mid_rel, ref_ranges)
        labels.append(CORRECTED if seg["corrected"] else RAW)
    return labels


def label_cells(cells: list[dict], labels: list[str], start_bucket: int) -> None:
    """把无原始点的格（空缺/插补）按桶标签补上时基；有点格与沿用格保留实测时基。"""
    for cell in cells:
        if cell.get("time_basis"):
            continue
        idx = cell["bucket"] - start_bucket
        if 0 <= idx < len(labels):
            cell["time_basis"] = labels[idx]


def build_correction(
    points: list[tuple[float, float]],
    drift_result: dict,
    reference_device: str | None,
    window_start: float,
    start_bucket: int,
    end_bucket: int,
) -> dict:
    """校时决策 + 执行：返回分段、校时后的点与发布用摘要。

    三种不校时的情形（基准设备 / 漂移估计不可用 / 偏移未超阈值）同样给出
    覆盖全窗的单段结论，保证结果页、质量报告与导出都能写明每段状态。
    """
    span_s = (end_bucket + 1) * settings.BUCKET_SECONDS - start_bucket * settings.BUCKET_SECONDS
    available = bool(drift_result.get("available"))
    drifted = bool(drift_result.get("drifted"))

    if not available or not drifted:
        if not available:
            status_label = "基准设备不校时"
            reason = drift_result.get("reason") or "本设备是该指标的时钟基准，时间视为可信"
        else:
            status_label = "无需校时"
            reason = (
                f"|偏移| {abs(drift_result.get('offset_s') or 0.0):.2f}s ≤ 阈值 "
                f"{drift_result.get('threshold_s', 0):g}s，保留设备原时刻"
            )
        segment = _display_segment(
            index=0,
            start_bucket=start_bucket,
            end_bucket=end_bucket,
            corrected=False,
            anchors=0,
            basis="设备原时刻",
            reason=reason,
        )
        return {
            "applied": False,
            "status": "none",
            "status_label": status_label,
            "reference_device": reference_device,
            "model": None,
            "max_anchor_gap_s": settings.CORRECT_MAX_ANCHOR_GAP_S,
            "segments": [segment],
            "points": [(ts, value, RAW) for ts, value in points],
            "bucket_labels": [RAW] * (end_bucket - start_bucket + 1),
            "corrected_points": 0,
            "raw_points": len(points),
            "note": f"{status_label}：{reason}。网格、缺口、摘要与覆盖率均按设备原时刻计算。",
        }

    a = drift_result["a"]
    b = drift_result["b"]
    segments = device_segments(drift_result.get("anchor_dev_times") or [], span_s)
    corrected = correct_points(points, window_start, a, b, segments, span_s)
    labels = bucket_labels(segments, a, b, start_bucket, end_bucket)
    ref_ranges = _ref_ranges(segments, a, b)

    # 发布用分段：按桶标签把相邻同状态桶并段（段与段不重不漏，并集覆盖全网格）
    display: list[dict] = []
    for offset, label in enumerate(labels):
        bucket = start_bucket + offset
        is_corrected = label == CORRECTED
        if display and display[-1]["corrected"] == is_corrected:
            display[-1]["end_bucket"] = bucket
        else:
            display.append({"corrected": is_corrected, "start_bucket": bucket, "end_bucket": bucket})

    basis_text = f"{reference_device} · t_ref = a·t_dev + b"
    seg_payloads = []
    for index, seg in enumerate(display):
        # 用与桶标签完全相同的判定找到该段对应的设备时段，取其锚点计数与成因
        mid_rel = (seg["start_bucket"] - start_bucket) * settings.BUCKET_SECONDS + settings.BUCKET_SECONDS / 2
        dev_seg = _segment_for_ref(mid_rel, ref_ranges)
        seg_payloads.append(
            _display_segment(
                index=index,
                start_bucket=seg["start_bucket"],
                end_bucket=seg["end_bucket"],
                corrected=seg["corrected"],
                anchors=dev_seg["anchors"],
                basis=basis_text if seg["corrected"] else "设备原时刻（锚点不足）",
                reason=dev_seg["reason"],
            )
        )

    corrected_points = sum(1 for _, _, basis_flag in corrected if basis_flag == CORRECTED)
    corrected_segments = sum(1 for s in seg_payloads if s["corrected"])
    if corrected_segments == 0:
        status, status_label = "none", "锚点不足未校时"
    elif corrected_segments == len(seg_payloads):
        status, status_label = "full", "全窗校时"
    else:
        status, status_label = "partial", "分段校时"
    return {
        "applied": corrected_segments > 0,
        "status": status,
        "status_label": status_label,
        "reference_device": reference_device,
        "model": {"a": a, "b": b, "formula": "t_ref = a · t_dev + b"},
        "max_anchor_gap_s": settings.CORRECT_MAX_ANCHOR_GAP_S,
        "segments": seg_payloads,
        "points": corrected,
        "bucket_labels": labels,
        "corrected_points": corrected_points,
        "raw_points": len(corrected) - corrected_points,
        "note": (
            f"相对基准 {reference_device} 校时：{len(seg_payloads)} 段中 {corrected_segments} 段已按 "
            f"t_ref = a·t_dev + b 校正，{len(seg_payloads) - corrected_segments} 段锚点不足保留原时刻；"
            "段与段首尾相接、不重不漏，网格/缺口/摘要/覆盖率按校时后时刻重算。"
        ),
    }


def _display_segment(
    index: int,
    start_bucket: int,
    end_bucket: int,
    corrected: bool,
    anchors: int,
    basis: str,
    reason: str,
) -> dict:
    return {
        "index": index,
        "start_bucket": start_bucket,
        "end_bucket": end_bucket,
        "start_time": grid.bucket_start_iso(start_bucket),
        "end_time": grid.bucket_start_iso(end_bucket),
        "cells": end_bucket - start_bucket + 1,
        "corrected": corrected,
        "status_label": "已校时" if corrected else "保留原时刻",
        "anchors": anchors,
        "basis": basis,
        "reason": reason,
    }


def cell_stats(cells: list[dict]) -> dict:
    """网格时基统计：已校时 / 原时刻 / 混合（段交界同格混排）。"""
    stats = {"corrected_cells": 0, "raw_cells": 0, "mixed_cells": 0, "unknown_cells": 0}
    for cell in cells:
        basis = cell.get("time_basis")
        if basis == CORRECTED:
            stats["corrected_cells"] += 1
        elif basis == RAW:
            stats["raw_cells"] += 1
        elif basis == "mixed":
            stats["mixed_cells"] += 1
        else:
            stats["unknown_cells"] += 1
    return stats
