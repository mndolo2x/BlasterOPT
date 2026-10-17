import pytest
import os
from src.engine.bench_model import BenchConfig, build_bench_scene, render_preview

def test_build_bench_scene():
    cfg = BenchConfig(
        bench_length_m=50.0,
        bench_width_m=30.0,
        bench_height_m=12.0,
        rows=3,
        holes_per_row=4,
    )
    scene, holes, metadata = build_bench_scene(cfg)
    assert len(holes) == 12
    assert metadata["geometry_count"] == 13
    assert metadata["total_holes"] == 12

def test_render_preview(tmp_path):
    cfg = BenchConfig()
    scene, holes, metadata = build_bench_scene(cfg)
    out_img = os.path.join(tmp_path, "preview.png")
    res = render_preview(scene, out_img)
    assert os.path.exists(res)
