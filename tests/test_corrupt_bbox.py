"""test_corrupt_bbox.py - 「做坏不会弄错标签」的自动测试（D6 计划书第 3 步）。

核心保证：做坏只改像素、不动框 → 标签文件必须一个字节都不差。
三层检查，从便宜到贵：
    ① 算子级：7 种坏都不改变图像的形状/类型/数值范围（合成小图，不依赖任何数据）
    ② 端到端：corrupt_one() 在临时目录的小数据上，两套标签与源文件逐字节相等
    ③ 真数据：data/corrupt 已生成时，seed 42 抽 50 张 × 7 种坏，逐一比对字节
运行：python -m pytest tests/ -v
"""

import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from rvkit.harness import generate_corrupt
from rvkit.harness.corruptions import CORRUPTION_NAMES, apply_corruption

# 真数据检查的前提：generate_corrupt 至少跑过一次（low_light_s2 是第一个产物）
DATA_ROOT = generate_corrupt.DATA_ROOT
CORRUPT_READY = (DATA_ROOT / "corrupt" / "low_light_s2" / "images").is_dir()


# ---- ① 算子级：做坏只能改像素，不能改几何 -------------------------------------

@pytest.mark.parametrize("cname", CORRUPTION_NAMES)
def test_corruption_preserves_shape_and_range(cname):
    """7 种坏各自跑一遍：输出仍是同形状的 BGR uint8、数值仍在 0~255。

    形状变了 = 框的坐标系就错了；类型/范围错了 = Ultralytics 读图会炸。
    """
    # 合成一张有内容的图：渐变背景 + 一个亮方块 + 一条亮线（避免全 0 图掩盖 bug）
    img = np.tile(np.linspace(30, 200, 64, dtype=np.uint8), (64, 1))
    img = np.stack([img] * 3, axis=-1)             # H×W → H×W×3 灰度当 BGR
    img[10:30, 10:30] = (40, 200, 40)              # 绿色方块
    img[50:52, 5:60] = (255, 255, 255)             # 白色横线

    out = apply_corruption(img, cname, severity=2, rng=np.random.default_rng(42))

    assert out.shape == img.shape                  # 几何不变：框的坐标才有意义
    assert out.dtype == np.uint8                   # 类型不变：YOLO 读图约定
    assert int(out.min()) >= 0 and int(out.max()) <= 255   # 数值范围合法
    assert not np.array_equal(out, img)            # 且确实"做坏"了，不是原样返回


# ---- ② 端到端：corrupt_one 产出的标签与源文件逐字节相等 -----------------------

@pytest.mark.parametrize("cname", CORRUPTION_NAMES)
def test_corrupt_one_copies_labels_byte_identical(cname, tmp_path):
    """在临时目录搭一个迷你数据树，跑 corrupt_one，比对两套标签的每个字节。

    这是"做坏不移动物体"这条物理事实的直接断言：标签是复制来的，
    复制的含义就是一个字节都不能变。
    """
    root = tmp_path / "data"
    img_name, txt_name = "abc-123.jpg", "abc-123.txt"
    label10, label8 = "2 0.500000 0.500000 0.250000 0.125000\n", "2 0.5 0.5 0.25 0.125\n"

    # 造源数据：一张真 jpg（cv2 写的全灰小图）+ 两套内容不同的标签
    # （write_text 在 Windows 会把 \n 写成 \r\n —— 所以断言一律对"源文件磁盘字节"比）
    (root / "val" / "images").mkdir(parents=True)
    (root / "val" / "labels").mkdir(parents=True)
    (root / "val8" / "labels").mkdir(parents=True)
    cv2.imwrite(str(root / "val" / "images" / img_name), np.full((16, 16, 3), 128, np.uint8))
    (root / "val" / "labels" / txt_name).write_text(label10, encoding="utf-8")
    (root / "val8" / "labels" / txt_name).write_text(label8, encoding="utf-8")
    src10 = (root / "val" / "labels" / txt_name).read_bytes()
    src8 = (root / "val8" / "labels" / txt_name).read_bytes()

    generate_corrupt.corrupt_one(img_name, cname, severity=2, data_root=root)

    got10 = (root / "corrupt" / f"{cname}_s2" / "labels" / txt_name).read_bytes()
    got8 = (root / "corrupt8" / f"{cname}_s2" / "labels" / txt_name).read_bytes()
    assert got10 == src10                          # 10 类标签逐字节相等
    assert got8 == src8                            # 8 类标签逐字节相等
    assert (root / "corrupt" / f"{cname}_s2" / "images" / img_name).exists()
    assert (root / "corrupt8" / f"{cname}_s2" / "images" / img_name).exists()


@pytest.mark.parametrize("cname", CORRUPTION_NAMES)
def test_corruption_is_deterministic(cname):
    """同一张图 + 同一种坏 + 同种子 → 两次结果逐像素相等（可复现铁律）。

    线程池里谁先谁后不该改变结果，所以随机性必须只来自 (图名, 坏名) 播种。
    rain 走 albumentations，这条测试顺便验证 Compose(seed=…) 真的钉死了它。
    """
    img = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    seed = generate_corrupt.name_seed("abc-123.jpg", cname)
    a = apply_corruption(img, cname, 2, np.random.default_rng(seed))
    b = apply_corruption(img, cname, 2, np.random.default_rng(seed))
    assert np.array_equal(a, b)


# ---- ③ 真数据：抽 50 张 × 7 种坏，逐一比对字节（没生成过数据就跳过）----------

@pytest.mark.skipif(not CORRUPT_READY,
                    reason="尚未生成坏图：先跑 python -m rvkit.harness.generate_corrupt")
def test_generated_labels_byte_identical_real_data():
    """计划书 D6 的验收动作：随机挑 50 张，断言做坏前后标签一个字节都不差。"""
    images_dir = DATA_ROOT / "corrupt" / "low_light_s2" / "images"
    all_names = sorted(p.name for p in images_dir.glob("*.jpg"))
    sample = random.Random(42).sample(all_names, min(50, len(all_names)))  # seed 42 可复现

    for name in sample:
        txt = Path(name).with_suffix(".txt").name
        src = (DATA_ROOT / "val" / "labels" / txt).read_bytes()
        src8 = (DATA_ROOT / "val8" / "labels" / txt).read_bytes()
        for cname in CORRUPTION_NAMES:
            got = (DATA_ROOT / "corrupt" / f"{cname}_s2" / "labels" / txt).read_bytes()
            got8 = (DATA_ROOT / "corrupt8" / f"{cname}_s2" / "labels" / txt).read_bytes()
            assert got == src, f"{cname}/{txt} 的 10 类标签被改动了！"
            assert got8 == src8, f"{cname}/{txt} 的 8 类标签被改动了！"
