"""test_convert.py — convert.py 的回归测试（防以后改动悄悄改坏转换逻辑）。

fixture = 手写小输入 + 手算好答案，不依赖真实大 JSON。
运行：python -m pytest tests/ -v
"""

import json
import sys
from collections import Counter
from pathlib import Path

# 让 tests/ 能 import 到 data/convert.py（rvkit/tests -> rvkit/data）
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA_DIR))

import convert


def _run(record, cls_map=convert.CLS_ID):
    """小帮手：跑 process_image 并返回 (lines, dropped 计数)。默认 bdd10 口径。"""
    stats = {"dropped": Counter()}
    _, lines = convert.process_image(record, stats, cls_map)
    return lines, stats


def test_in_frame_car_full_line():
    """① 画面内 car 框 → 整行逐字符等于手算值。

    手算：box=(0,0,640,360)，cx=(0+640)/2/1280=0.25，cy=(0+360)/2/720=0.25，
    w=640/1280=0.5，h=360/720=0.5；bdd10 口径下 car 的类别 id=2。
    """
    lines, stats = _run({"name": "a.jpg", "labels": [
        {"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 640, "y2": 360}},
    ]})
    assert lines == ["2 0.250000 0.250000 0.500000 0.500000"]
    assert stats["dropped"]["car"] == 0


def test_poly2d_lane_excluded():
    """② 只有 poly2d 的 lane → 不产生任何 YOLO 行（两种口径都不收 lane）。"""
    lines, stats = _run({"name": "b.jpg", "labels": [
        {"category": "lane", "poly2d": [{"vertices": [[0, 0], [1, 1]]}]},
    ]})
    assert lines == []
    assert stats["dropped"]["lane"] == 1


def test_motor_id_is_6():
    """③ bdd10 口径：motor → 类别 id 必须是 6。"""
    lines, stats = _run({"name": "c.jpg", "labels": [
        {"category": "motor", "box2d": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
    ]})
    assert len(lines) == 1
    assert lines[0].startswith("6 ")


def test_clipped_out_of_frame_box():
    """④ x1=-50 越界框 → 先裁剪再归一化。

    手算：裁剪到画面内后为 (0, 0, 1280, 720)，
    cx=(0+1280)/2/1280=0.5，cy=(0+720)/2/720=0.5，w=1.0，h=1.0。
    """
    lines, stats = _run({"name": "d.jpg", "labels": [
        {"category": "car", "box2d": {"x1": -50, "y1": 0, "x2": 1300, "y2": 720}},
    ]})
    assert lines == ["2 0.500000 0.500000 1.000000 1.000000"]


def test_zero_width_box_dropped():
    """⑤ 宽为 0 的退化框 → 丢弃（无面积）。"""
    lines, stats = _run({"name": "e.jpg", "labels": [
        {"category": "car", "box2d": {"x1": 100, "y1": 100, "x2": 100, "y2": 200}},
    ]})
    assert lines == []
    assert stats["dropped"]["car"] == 1


def test_bdd8coco_mapping():
    """⑥ bdd8coco 口径：类别编号必须对齐 COCO 官方编号。

    手算（COCO 编号）：person=0、bicycle(bike)=1、car=2、motorcycle(motor)=3、
    traffic light=9。rider 并入 person（也是 0）；traffic sign 不在映射表 → 丢弃。
    拿 COCO 现成模型 val 时就是靠这套编号对号入座的，错一个号全表作废。
    """
    lines, stats = _run({"name": "f.jpg", "labels": [
        {"category": "rider",         "box2d": {"x1": 0,  "y1": 0, "x2": 10, "y2": 20}},  # → person=0
        {"category": "car",           "box2d": {"x1": 10, "y1": 0, "x2": 20, "y2": 20}},  # → 2
        {"category": "motor",         "box2d": {"x1": 20, "y1": 0, "x2": 30, "y2": 20}},  # → 3
        {"category": "traffic light", "box2d": {"x1": 30, "y1": 0, "x2": 40, "y2": 20}},  # → 9
        {"category": "traffic sign",  "box2d": {"x1": 40, "y1": 0, "x2": 50, "y2": 20}},  # 丢弃
    ]}, cls_map=convert.BDD_TO_COCO_ID)
    assert [line.split()[0] for line in lines] == ["0", "2", "3", "9"]
    assert stats["dropped"]["traffic sign"] == 1


def test_lane_only_image_writes_empty_txt(tmp_path):
    """⑦ 整图只有 lane → 转换后写出空 txt（端到端走 convert_split）。

    convert_split 的 out_dir / cls_map 都是显式传参 → 指到临时目录即可，不碰真实数据。
    """
    fake_ann = tmp_path / "ann.json"
    fake_ann.write_text(json.dumps([
        {"name": "only_lane.jpg", "labels": [
            {"category": "lane", "poly2d": [{"vertices": [[0, 0], [1, 1]]}]},
        ]},
    ]), encoding="utf-8")

    out_dir = tmp_path / "labels_out"
    stats = {"images": 0, "kept": Counter(), "dropped": Counter()}
    convert.convert_split("smoke", fake_ann, stats, out_dir, convert.CLS_ID)

    out = out_dir / "only_lane.txt"           # out_dir 即最终标签目录（split 名由调用方拼）
    assert out.exists()                       # 空图也要产出 txt
    assert out.read_text(encoding="utf-8") == ""   # 且内容为空


def test_datasets_class_tables_consistent():
    """⑧ 类别登记表必须一字不差：convert.py 与 harness/datasets.py 各存了一份。

    data 脚本保持可独立运行所以自带一份；工具包（datasets.py）自己一份。
    这条测试保证有人只改了其中一份时立刻报警。
    """
    from rvkit.harness import datasets

    assert datasets.CLASSES10 == convert.CLASSES          # 10 类口径：名称与顺序全同
    for name, cid in convert.BDD_TO_COCO_ID.items():      # 8 类口径：COCO 编号一一对应
        if name != "rider":                               # rider 并入 person，无独立编号
            assert datasets.CLASSES8_YAML[cid] == name
