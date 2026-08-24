"""skill_entry 集成测试：标准 JSON 结构与三种输出格式。"""
from __future__ import annotations

from pathlib import Path

from conftest import make_synthetic_pack
from skill_entry import REQUIRED_KEYS, run_safety_inspection_skill


def test_skill_returns_standard_schema(tmp_path: Path):
    csv_path = tmp_path / "pack.csv"
    make_synthetic_pack(n_cells=4, n_cycles=3).to_csv(csv_path, index=False)
    report = run_safety_inspection_skill(str(csv_path), sensitivity=0.95, out_format="none")
    for key in REQUIRED_KEYS:
        assert key in report, f"缺少字段 {key}"
    assert isinstance(report["pack_health_status"], str)
    assert isinstance(report["abnormal_cells_list"], list)
    assert isinstance(report["max_dT_dt"], (int, float))
    assert isinstance(report["root_cause_prediction"], str)


def test_skill_writes_all_formats(tmp_path: Path):
    csv_path = tmp_path / "pack.csv"
    make_synthetic_pack(n_cells=4, n_cycles=3).to_csv(csv_path, index=False)
    out = tmp_path / "out"
    run_safety_inspection_skill(str(csv_path), out_format="json", out_dir=out)
    run_safety_inspection_skill(str(csv_path), out_format="md", out_dir=out)
    run_safety_inspection_skill(str(csv_path), out_format="pdf", out_dir=out)
    assert (out / "safety_report.json").exists()
    assert (out / "safety_report.md").exists()
    assert (out / "safety_report.pdf").exists()
    assert (out / "safety_report.pdf").stat().st_size > 1000
