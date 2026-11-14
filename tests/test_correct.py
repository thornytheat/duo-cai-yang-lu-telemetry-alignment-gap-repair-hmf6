"""分段校时：锚点充分度分段、逐段校正、交界不重不漏、窗口边缘裁剪、确定性。"""
from __future__ import annotations

import pytest

from app import correct, drift, grid, settings

START = 1_750_000_000.0 // 60 * 60  # 对齐到整分钟的窗口起点
SPAN_S = 3600.0
B0 = int(START // 60)


def triangle(t: float, period: float = 480.0, amp: float = 4.0, base: float = 21.0) -> float:
    phase = (t % period) / period
    return base + amp * (4 * abs(phase - 0.5) - 1)


def make_series(offset_s: float, interval_s: int, span_s: int = 7200) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    t = 0.0
    while t < span_s:
        points.append((START + t + offset_s, triangle(t)))
        t += interval_s
    return points


def drift_result_for(dev_points, ref_points=None, threshold=5.0):
    ref_points = ref_points if ref_points is not None else make_series(0.0, 1)
    result = drift.estimate_drift(ref_points, dev_points, START, threshold)
    result["available"] = True
    return result


def test_device_segments_tile_without_overlap_or_gap():
    anchors = [100.0, 400.0, 700.0, 2500.0, 2800.0, 3100.0]
    segments = correct.device_segments(anchors, SPAN_S, max_gap_s=900.0)
    # 不重不漏：依次首尾相接，并集恰好覆盖 [0, span]
    assert segments[0]["start_s"] == 0.0
    assert segments[-1]["end_s"] == SPAN_S
    for prev, nxt in zip(segments, segments[1:]):
        assert nxt["start_s"] == prev["end_s"]
    # 700 → 2500 的间距 1800s > 900s：中间夹一个锚点不足段
    statuses = [s["corrected"] for s in segments]
    assert statuses == [True, False, True]
    insufficient = segments[1]
    assert insufficient["anchors"] == 0
    assert "锚点不足" in insufficient["reason"]


def test_no_anchors_marks_whole_window_raw():
    segments = correct.device_segments([], SPAN_S)
    assert len(segments) == 1
    assert segments[0]["corrected"] is False


def test_correct_points_shifts_supported_and_keeps_unsupported():
    segments = correct.device_segments([100.0, 400.0, 2500.0, 2800.0], SPAN_S, max_gap_s=900.0)
    a, b = 0.9999, -37.0
    points = [(START + 50.0, 1.0), (START + 1500.0, 2.0), (START + 2600.0, 3.0)]
    out = correct.correct_points(points, START, a, b, segments, SPAN_S)
    assert out[0][2] == "corrected"
    assert out[0][0] == pytest.approx(START + a * 50.0 + b)
    assert out[1][2] == "raw"          # 1500s 落在锚点不足段（700→2500 空档）
    assert out[1][0] == START + 1500.0  # 保留原时刻
    assert out[2][2] == "corrected"


def test_corrected_points_are_clamped_into_window():
    segments = [{"start_s": 0.0, "end_s": SPAN_S, "corrected": True, "anchors": 4, "reason": ""}]
    # b 让首点校正后略小于窗口起点（拟合残差导致），应被裁剪回窗口而不是丢失
    points = [(START + 37.0, 1.0), (START + 97.0, 2.0)]
    out = correct.correct_points(points, START, a=1.0, b=-37.01, segments=segments, span_s=SPAN_S)
    assert out[0][0] == START
    assert grid.bucket_of(out[0][0]) == B0


def test_bucket_labels_cover_every_bucket_exactly_once():
    anchors = [100.0, 400.0, 2500.0, 2800.0, 3550.0]
    segments = correct.device_segments(anchors, SPAN_S, max_gap_s=900.0)
    labels = correct.bucket_labels(segments, a=0.9999, b=-37.0, start_bucket=B0, end_bucket=B0 + 59)
    assert len(labels) == 60
    assert set(labels) <= {correct.CORRECTED, correct.RAW}
    # 与段结构一致：前段 corrected，中段（锚点空档）raw，尾段 corrected
    assert labels[0] == correct.CORRECTED
    mid_bucket = int((1500.0 + 30.0) // 60)  # 1500s 落在锚点不足段
    assert labels[mid_bucket] == correct.RAW
    assert labels[-1] == correct.CORRECTED


def test_drifted_series_corrects_supported_points_and_marks_outage():
    ref = make_series(0.0, 1)
    dev = make_series(37.0, 60)
    # 挖掉 30 分钟（t=1800..3600）制造锚点不足段
    dev = [p for p in dev if not (START + 1800 + 37 <= p[0] < START + 3600 + 37)]
    drift_result = drift_result_for(dev, ref)
    assert drift_result["drifted"] is True

    end_bucket = int((START + 7200) // 60)
    corr = correct.build_correction(dev, drift_result, "REF", START, B0, end_bucket)
    assert corr["applied"] is True
    assert corr["status"] == "partial"
    raw_segments = [s for s in corr["segments"] if not s["corrected"]]
    assert raw_segments, "断采 30 分钟应产生锚点不足段"
    # 断采段内的点保留原时刻；充分段的点被校正约 -37s
    for ts, value, basis in corr["points"]:
        rel = ts - START
        if basis == "corrected":
            assert abs((rel % 60)) < 1.0 or abs((rel % 60) - 60) < 1.0  # 校正后回到整分钟附近
        else:
            assert pytest.approx(rel % 60, abs=0.6) == 37.0 % 60        # 原时刻仍是 +37s 相位


def test_correction_is_deterministic():
    ref = make_series(0.0, 1)
    dev = make_series(37.0, 60)
    drift_result = drift_result_for(dev, ref)
    end_bucket = int((START + 7200) // 60)
    first = correct.build_correction(dev, drift_result, "REF", START, B0, end_bucket)
    second = correct.build_correction(dev, drift_result, "REF", START, B0, end_bucket)
    assert first["points"] == second["points"]
    assert first["segments"] == second["segments"]
    assert first["bucket_labels"] == second["bucket_labels"]


def test_not_drifted_series_passes_through_with_single_segment():
    ref = make_series(0.0, 1)
    dev = make_series(1.0, 60)  # 偏移 1s < 阈值 5s
    drift_result = drift_result_for(dev, ref)
    assert drift_result["drifted"] is False
    corr = correct.build_correction(dev, drift_result, "REF", START, B0, B0 + 119)
    assert corr["applied"] is False
    assert corr["status_label"] == "无需校时"
    assert len(corr["segments"]) == 1
    assert corr["segments"][0]["cells"] == 120
    assert all(basis == correct.RAW for _, _, basis in corr["points"])


def test_grid_marks_time_basis_and_hold_inherits():
    # 60s 设备：校正段两点（校正后落在整分钟附近）+ 锚点不足段两点，中间一格 hold
    points = [
        (START + 0.0, 20.0, "corrected"),
        (START + 60.0, 21.0, "corrected"),
        (START + 1800.0, 22.0, "raw"),
        (START + 1920.0, 23.0, "raw"),
    ]
    cells = grid.resample(points, interval_s=60, start_bucket=B0, end_bucket=B0 + 32)
    assert cells[0]["time_basis"] == "corrected"
    assert cells[1]["time_basis"] == "corrected"
    assert cells[30]["time_basis"] == "raw"
    assert cells[31]["time_basis"] == "raw"   # hold 沿用，继承前值格时基
    assert cells[31]["carried"] is True
    assert cells[32]["time_basis"] == "raw"


def test_corrected_points_land_in_nearest_bucket():
    # 校正后时刻带亚秒残差：59.994 应归入第 1 桶（最近桶），而不是被 floor 到第 0 桶
    points = [(START + 59.994, 5.0, "corrected"), (START + 120.002, 6.0, "corrected")]
    cells = grid.resample(points, interval_s=60, start_bucket=B0, end_bucket=B0 + 2)
    assert cells[0]["source"] == "empty"
    assert cells[1]["samples"] == 1
    assert cells[1]["value"] == pytest.approx(5.0)
    assert cells[2]["samples"] == 1
    assert cells[2]["value"] == pytest.approx(6.0)
