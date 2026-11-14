"""API 端到端：导入、对齐、结果页、错误输入 4xx、质量报告与 CSV 导出。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        # 主流程第一步：幂等导入样例遥测
        resp = test_client.post("/api/import")
        assert resp.status_code == 200, resp.text
        yield test_client


def align(client: TestClient, **payload):
    return client.post("/api/align", json=payload)


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["app"] == "tsgrid"


def test_import_is_idempotent(client: TestClient):
    first = client.post("/api/import").json()
    second = client.post("/api/import").json()
    assert first["dataset_id"] == second["dataset_id"]
    assert first["sha256"] == second["sha256"]
    assert first["rows"] > 20000
    assert len(first["devices"]) == 3
    assert {d["device_id"] for d in first["devices"]} == {"LINE-A-01", "LINE-A-02", "LINE-A-03"}
    assert first["window"]["cells"] == 360


def test_drifted_device_offset_is_about_37_seconds(client: TestClient):
    resp = align(client, device_id="LINE-A-03", metric="temperature", offset_threshold=5)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["drift"]["available"] is True
    assert abs(data["drift"]["offset_s"] - 37.0) < 1.0
    assert data["drift"]["drifted"] is True
    assert data["drift"]["rmse_s"] < 1.0
    assert data["reference"]["device_id"] == "LINE-A-01"
    # 温度序列：3 格插补缺口 + 5 格未修复缺口 = 2 处
    assert data["gaps"]["count"] == 2
    assert data["gaps"]["interpolated_cells"] == 3
    assert data["gaps"]["unrepaired_cells"] == 5
    assert data["coverage"]["grade"] == "A"
    assert data["coverage"]["expected_cells"] == 360
    assert data["interval_s"] == 60
    assert data["run_id"].startswith("run-")


def test_low_rate_grades_and_carried_counts(client: TestClient):
    temperature = align(client, device_id="LINE-A-02", metric="temperature").json()
    assert temperature["drift"]["drifted"] is False          # 5s 设备时钟正常
    assert temperature["coverage"]["grade"] == "B"
    assert temperature["coverage"]["carried_cells"] == 30    # 3 分钟降级段被 hold
    assert temperature["gaps"]["count"] == 3

    pressure = align(client, device_id="LINE-A-03", metric="pressure").json()
    assert pressure["coverage"]["grade"] == "C"
    assert abs(pressure["drift"]["offset_s"] - 37.0) < 1.0
    assert pressure["gaps"]["unrepaired_cells"] == 30


def test_reference_device_reports_no_self_comparison(client: TestClient):
    data = align(client, device_id="LINE-A-01", metric="temperature").json()
    assert data["reference"]["self_reference"] is True
    assert data["drift"]["available"] is False
    assert data["drift"]["offset_s"] is None


def test_threshold_change_flips_verdict(client: TestClient):
    strict = align(client, device_id="LINE-A-02", metric="temperature", offset_threshold=0.005).json()
    assert strict["drift"]["drifted"] is True


def test_unknown_device_returns_4xx_with_reason(client: TestClient):
    resp = align(client, device_id="LINE-A-99", metric="temperature")
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "LINE-A-99" in detail
    assert "可用" in detail


def test_metric_not_on_device_returns_4xx(client: TestClient):
    resp = align(client, device_id="LINE-A-01", metric="pressure")
    assert resp.status_code == 400
    assert "pressure" in resp.json()["detail"]


def test_bad_threshold_returns_4xx(client: TestClient):
    resp = align(client, device_id="LINE-A-03", metric="temperature", offset_threshold="abc")
    assert resp.status_code == 400
    assert "阈值" in resp.json()["detail"]

    resp_negative = align(client, device_id="LINE-A-03", metric="temperature", offset_threshold=-5)
    assert resp_negative.status_code == 400


def test_missing_fields_return_4xx(client: TestClient):
    assert client.post("/api/align", json={"metric": "temperature"}).status_code == 400
    assert client.post("/api/align", json={"device_id": "LINE-A-03"}).status_code == 400


def test_result_page_renders_evidence(client: TestClient):
    data = align(client, device_id="LINE-A-03", metric="temperature").json()
    resp = client.get(f"/result/{data['run_id']}")
    assert resp.status_code == 200
    html = resp.text
    for expected in ["时钟漂移估计", "校正建议", "校时分段", "已校时", "保留原时刻", "1 分钟网格预览", "缺口明细", "小时摘要", "覆盖率与分级", "37."]:
        assert expected in html


def test_unknown_run_returns_404(client: TestClient):
    assert client.get("/result/run-doesnotexist").status_code == 404
    assert client.get("/api/runs/run-doesnotexist").status_code == 404


def test_pages_render(client: TestClient):
    for path in ["/", "/workbench", "/review"]:
        resp = client.get(path)
        assert resp.status_code == 200, path
        assert "遥测" in resp.text or "对齐" in resp.text


def test_report_and_csv_export(client: TestClient):
    report = client.get("/api/report").json()
    assert len(report["rows"]) == 5
    grades = {row["coverage"]["grade"] for row in report["rows"]}
    assert grades == {"A", "B", "C"}
    assert len(report["devices"]) == 3
    assert report["summary"]["total_gaps"] >= 5

    csv_resp = client.get("/api/report.csv")
    assert csv_resp.status_code == 200
    assert csv_resp.headers["content-type"].startswith("text/csv")
    lines = [line for line in csv_resp.text.strip().splitlines() if line]
    assert lines[0].startswith("device_id,metric,interval_s")
    assert "time_correction" in lines[0]
    assert "corrected_cells" in lines[0]
    assert len(lines) == 6  # 表头 + 5 条序列
    # 漂移设备的导出行写明校时状态与基准
    drifted_row = [line for line in lines if line.startswith("LINE-A-03,temperature")][0]
    assert "分段校时" in drifted_row
    assert "LINE-A-01" in drifted_row


def test_alignment_pipeline_stages_recorded(client: TestClient):
    data = align(client, device_id="LINE-A-03", metric="temperature").json()
    names = [stage["name"] for stage in data["stages"]]
    assert names == ["数据导入", "漂移估计", "分段校时", "网格重采样", "缺口判定", "小时摘要", "质量评分"]
    assert data["total_ms"] > 0
    assert len(data["hourly"]) == 6
    assert len(data["preview"]) == 60


def test_correction_segments_tile_window_and_mark_status(client: TestClient):
    data = align(client, device_id="LINE-A-03", metric="temperature").json()
    corr = data["correction"]
    assert corr["applied"] is True
    assert corr["status_label"] == "分段校时"
    assert corr["reference_device"] == "LINE-A-01"
    assert corr["model"]["formula"] == "t_ref = a · t_dev + b"

    segments = corr["segments"]
    assert [s["corrected"] for s in segments] == [True, False, True]
    # 段与段不重不漏：首尾相接，并集恰好覆盖整个网格
    assert segments[0]["start_bucket"] == data["preview"][0]["bucket"]
    assert segments[-1]["end_bucket"] == data["preview"][0]["bucket"] + data["window"]["cells"] - 1
    for prev, nxt in zip(segments, segments[1:]):
        assert nxt["start_bucket"] == prev["end_bucket"] + 1
    assert sum(s["cells"] for s in segments) == data["window"]["cells"]
    # 锚点不足段（05:00–05:50 降级上报）保留原时刻并明确标注
    raw_seg = segments[1]
    assert raw_seg["status_label"] == "保留原时刻"
    assert "锚点不足" in raw_seg["reason"]
    assert raw_seg["anchors"] == 0
    assert segments[0]["anchors"] > 0
    # 每格都有时基标注，校正格 + 原时刻格 = 全网格
    cells = corr["cells"]
    assert cells["corrected_cells"] + cells["raw_cells"] + cells["mixed_cells"] == data["window"]["cells"]
    assert cells["unknown_cells"] == 0


def test_correction_keeps_raw_times_in_insufficient_segment(client: TestClient):
    data = align(client, device_id="LINE-A-03", metric="pressure").json()
    corr = data["correction"]
    assert corr["applied"] is True
    assert len(corr["segments"]) == 5  # 断采段 + 降级段 两处锚点不足
    raw_segments = [s for s in corr["segments"] if not s["corrected"]]
    assert len(raw_segments) == 2
    assert all("锚点不足" in s["reason"] for s in raw_segments)


def test_reference_and_normal_devices_report_no_correction(client: TestClient):
    ref = align(client, device_id="LINE-A-01", metric="temperature").json()
    assert ref["correction"]["applied"] is False
    assert ref["correction"]["status_label"] == "基准设备不校时"
    assert len(ref["correction"]["segments"]) == 1

    normal = align(client, device_id="LINE-A-02", metric="temperature").json()
    assert normal["correction"]["applied"] is False
    assert normal["correction"]["status_label"] == "无需校时"


def test_rerun_same_data_produces_identical_result(client: TestClient):
    first = align(client, device_id="LINE-A-03", metric="temperature").json()
    second = align(client, device_id="LINE-A-03", metric="temperature").json()
    assert first["run_id"] != second["run_id"]  # 各落一条记录，旧版保留
    for key in ["correction", "grid", "gaps", "coverage", "hourly", "preview", "drift"]:
        assert first[key] == second[key], key


def test_failed_publish_keeps_existing_runs(client: TestClient, monkeypatch):
    before = client.get("/api/runs/" + align(client, device_id="LINE-A-03", metric="temperature").json()["run_id"])
    assert before.status_code == 200
    runs_url = "/api/report"
    report_before = client.get(runs_url).json()

    def boom(*args, **kwargs):
        raise RuntimeError("磁盘已满（模拟发布失败）")

    monkeypatch.setattr("app.main.store.save_run", boom)
    resp = align(client, device_id="LINE-A-03", metric="temperature")
    assert resp.status_code == 500
    assert "发布失败" in resp.json()["detail"]
    monkeypatch.undo()

    # 旧版仍在：既有运行记录可回看，质量报告未被破坏
    again = client.get("/api/runs/" + before.json()["run_id"])
    assert again.status_code == 200
    assert again.json()["run_id"] == before.json()["run_id"]
    assert client.get(runs_url).json() == report_before


def test_legacy_run_without_correction_still_renders(client: TestClient):
    """校时功能上线前的运行记录（payload 无 correction）：降级展示而不是 500。"""
    from app import pipeline, store
    from app.main import STATE

    dataset = STATE["dataset"]
    payload = pipeline.run_alignment(dataset, "LINE-A-03", "temperature")
    legacy = {k: v for k, v in payload.items() if k != "correction"}
    run_id = store.save_run(dataset.dataset_id, legacy)
    resp = client.get(f"/result/{run_id}")
    assert resp.status_code == 200
    assert "旧版结果未校时" in resp.text
