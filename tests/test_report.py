"""test_report.py - 报告生成器的冒烟测试（D7）。

用手造的小成绩表跑 render()，断言：报告 md 里三行总结都在、热力图真的出了图。
不依赖任何真实模型/数据——纯 DataFrame 进、文件出。
运行：python -m pytest tests/ -v
"""

import pandas as pd

from rvkit.harness import report


def _fake_df():
    """仿 runner 主表：clean_day 基准 + night 大掉分 + rain 小掉分。"""
    rows = [
        {"condition": "clean_day", "map": 0.4000, "map50": 0.5500, "p": 0.60, "r": 0.50, "rel_drop": 0.0},
        {"condition": "night",     "map": 0.3000, "map50": 0.4400, "p": 0.50, "r": 0.42, "rel_drop": -25.0},
        {"condition": "rain",      "map": 0.3800, "map50": 0.5200, "p": 0.58, "r": 0.49, "rel_drop": -5.0},
    ]
    df = pd.DataFrame(rows)
    df.attrs["counts"] = {"clean_day": 300, "night": 300, "rain": 300}  # 样本数走元数据
    return df


def _fake_perclass():
    """仿每类长表：person 夜里掉 50%（最狠），car 只掉 12%。"""
    rows = []
    for cond, person, car in [("clean_day", 0.60, 0.50), ("night", 0.30, 0.44), ("rain", 0.57, 0.49)]:
        for cls_id, cls, ap in [(0, "person", person), (2, "car", car)]:
            rows.append({"condition": cond, "cls_id": cls_id, "cls": cls, "ap": ap})
    return pd.DataFrame(rows)


def test_render_writes_report_and_heatmap(tmp_path):
    out = tmp_path / "results" / "mini_report.md"
    report.render(_fake_df(), _fake_perclass(), model="yolo11s.pt", mode="bdd8coco", out=out)

    md = out.read_text(encoding="utf-8")
    assert "最狠的条件：night（mAP50-95 掉 25.0%，0.4000 → 0.3000）" in md   # 挑对了最狠的行
    assert "掉最狠的类：person" in md                                        # 类别判定正确（-27.5% vs -7%）
    assert "![掉分热力图](figures/mini_report_heatmap.png)" in md            # 图引用写进报告

    fig = tmp_path / "results" / "figures" / "mini_report_heatmap.png"
    assert fig.exists() and fig.stat().st_size > 1000                        # 热力图真的出了图


def test_small_sample_gets_caveat(tmp_path):
    """样本数 < 100 的条件（如雾），总结里必须带"仅供参考"括号。"""
    df = _fake_df()
    rows = df.to_dict("records")
    rows[2]["rel_drop"] = -80.0            # 让"雾"成为最狠条件，触发免责括号
    df = pd.DataFrame(rows)
    # 注意：attrs 不跟随 DataFrame 重建，必须在重建之后重新挂样本数
    df.attrs["counts"] = {"clean_day": 300, "night": 300, "rain": 9}

    out = tmp_path / "r.md"
    report.render(df, _fake_perclass(), model="m.pt", mode="bdd8coco", out=out)
    assert "仅 9 张，仅供参考" in out.read_text(encoding="utf-8")


def test_df_to_markdown_shape():
    md = report.df_to_markdown(_fake_df())
    assert md.splitlines()[0] == "| condition | map | map50 | p | r | rel_drop |"
    assert "| night |" in md
