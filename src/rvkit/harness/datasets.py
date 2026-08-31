#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""datasets.py - 把「条件名单」翻译成 Ultralytics 看得懂的说明书（D4）。

Ultralytics 考试（val）需要两样东西：
    1. 一个 txt：每行一张图的路径；
    2. 一个 yaml：path=数据根目录、val=上面的 txt、names=类别编号对照表。
它找标签的规则是死的：把图片路径里最后一段 "images" 替换成 "labels" 去找同名 txt。
所以图片必须放在某个 images/ 目录里，标签放在平级的 labels/（或 labels8 对应的
val8/labels/）目录里。

核心函数（计划书 D4 指定的三个 + 一个辅助）：
    make_paths_txt(...) 名单里的 "xxx.jpg" → 带目录的路径 txt
    make_yaml(...)      生成 yaml 说明书
    make_mini(...)      从名单取前 N 张做「迷你版」（本机没大显卡，跑 300 张才等得起）
    ensure_image_mirror(...)  bdd8coco 口径专用：把图片硬链接进 val8/images

另有一个云端入口（D9 第 3 步用）：
    python -m rvkit.harness.datasets --data-root /root/autodl-tmp/data
    → 生成 splits_train/train_clear_all_paths.txt 和 trainval_1k_paths.txt
      （带路径的训练名单，给 configs/m0_local.yaml 的 train/val 指向用）
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]     # 仓库根（本文件在 src/rvkit/harness/ 下）
REPO_SPLITS = REPO_ROOT / "data" / "splits"         # 随仓库走的名单（git clone 自带）

# ---- 类别登记表（与 data/convert.py 各存一份，tests/test_convert.py 保证一致）--

# 10 类口径：BDD100K 原始类别与编号（我们自己的 M0/M1/M2 用这套）
CLASSES10 = [
    "person", "rider", "car", "truck", "bus",
    "train", "motor", "bike", "traffic light", "traffic sign",
]

# 8 类 COCO 对齐口径：给 COCO 现成模型（yolo11s.pt）考试用。
# 编号必须与 COCO 官方一致：0 person / 1 bicycle(=bike) / 2 car / 3 motorcycle(=motor)
# / 5 bus / 6 train / 7 truck / 9 traffic light。
# 4、8 两号在 COCO 里是 airplane / boat，BDD 没有 → 用原名占位
# （标签文件里永远不会出现这两个号，评测时也不会计入 mAP）。
CLASSES8_YAML = [
    "person", "bike", "car", "motor", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
]

# 每种口径：图片目录（相对数据根目录）+ yaml 里的类别表
MODE_IMAGE_DIR = {"bdd10": "val/images", "bdd8coco": "val8/images"}
MODE_CLASS_NAMES = {"bdd10": CLASSES10, "bdd8coco": CLASSES8_YAML}


# ---- 名单处理 ---------------------------------------------------------------

def read_names(list_path):
    """读名单 txt → 图名列表（自动去空行和首尾空白，容忍文件末尾多一个换行）。"""
    lines = Path(list_path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def make_mini(names, n=300):
    """迷你版 = 名单的前 n 张。

    固定取前 n、不用随机：谁在什么时候跑都拿到同一批图，结果可复现、好排错。
    （每个条件名单本身已经打乱过顺序，所以前 n 张就是随机样本。）
    """
    return names[:n]


def write_names(path, names):
    """把图名列表写成 txt，每行一个文件名（与 data/ 下各脚本的写法一致）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{n}\n" for n in names), encoding="utf-8")


# ---- Ultralytics 说明书 -----------------------------------------------------

def make_paths_txt(names, images_subdir, out_path, data_root):
    """把名单变成「带目录的图片清单」：每行 xxx.jpg → <数据根>/val/images/xxx.jpg。

    关键坑（实测踩过）：Ultralytics 读这种 txt 时是按原样使用每行路径的——
    相对路径会被拿去相对「进程当前目录」解析，而不是拼上 yaml 里的 path 根目录。
    所以这里直接写绝对路径（正斜杠格式），运行目录怎么变都不会找错文件。
    这些 txt/yaml 是每次运行都重新生成的临时产物，绝对路径不影响可移植性。
    images_subdir 例如 "val/images"（10 类）或 "val8/images"（8 类镜像）。
    返回写好的 txt 路径。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [(Path(data_root) / images_subdir / name).resolve().as_posix()
             for name in names]
    out_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return out_path


def make_yaml(name, paths_txt, class_names, data_root, out_path):
    """生成 Ultralytics 的 yaml 说明书，返回 yaml 路径。

    长这样（三行核心）：
        path: <数据根目录绝对路径>     ← 图片/标签都相对它找
        val:  gen/bdd8coco/night.txt  ← 上一步的图片清单（相对 path）
        names: {0: person, ...}       ← 类别编号对照表
    """
    paths_txt = Path(paths_txt)
    # txt 路径转成「相对数据根目录」的形式（val: 字段的规范写法）
    val_rel = paths_txt.resolve().relative_to(Path(data_root).resolve()).as_posix()
    # path 用绝对路径 + 正斜杠：Ultralytics 内部拼接路径时对反斜杠容易出问题。
    # train 必须写：Ultralytics 规定 yaml 里 train/val 两个键缺一不可，
    # 评测（val）根本不会读 train，这里指向同一份清单纯当占位。
    body = [f"path: {Path(data_root).resolve().as_posix()}",
            f"# train 为占位（Ultralytics 强制要求该键存在，评测只读 val）",
            f"train: {val_rel}",
            f"val: {val_rel}",
            "names:",
            *(f"  {i}: {cls_name}" for i, cls_name in enumerate(class_names))]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return out_path


# ---- bdd8coco 专用：图片镜像 ------------------------------------------------

def ensure_image_mirror(names, src_dir, mirror_dir):
    """把需要的图片「硬链接」进镜像目录，返回新建的链接数（已有的跳过）。

    为什么需要：Ultralytics 按图片路径把 images→labels 找标签。8 类标签在
    val8/labels/，所以图片也必须出现在 val8/images/ 里它才找得到。
    硬链接 = 给同一个文件起第二个名字，不占额外磁盘空间（10k 张也就几秒钟）；
    万一文件系统不支持硬链接（跨盘 / U 盘 / 网盘目录），退回复制一份。
    """
    mirror_dir = Path(mirror_dir)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    n_linked = 0
    for name in names:
        dst = mirror_dir / name
        if dst.exists():               # 可重复运行：已链接过的图直接跳过
            continue
        src = Path(src_dir) / name
        try:
            os.link(src, dst)          # 硬链接：同一份数据、两个目录都能看到
        except OSError:
            shutil.copy2(src, dst)     # 兜底：复制（慢一点但一定成功）
        n_linked += 1
    return n_linked


# ---- 一站式组装 ---------------------------------------------------------------

def resolve_images_subdir(condition, mode, data_root):
    """返回一个条件该用的图片目录（相对数据根目录的 posix 路径）。

    自然条件（clean_day / night / …）→ val/images 或 val8/images；
    坏图条件（data/corrupt/<条件名>/ 已生成，如 low_light_s2）
                     → corrupt/<条件名>/images（10 类）或 corrupt8/<条件名>/images（8 类）。
    判断方式很朴素：数据目录里存在同名坏图文件夹就是坏图条件，不用改名单格式。
    """
    data_root = Path(data_root)
    if (data_root / "corrupt" / condition).is_dir():
        root8 = "corrupt8" if mode == "bdd8coco" else "corrupt"
        return f"{root8}/{condition}/images"
    return MODE_IMAGE_DIR[mode]


def build_eval_yaml(condition, names, mode, data_root):
    """把一个条件的图名列表变成可直接喂给 adapter.val() 的 yaml，返回 yaml 路径。

    中间产物统一放 data/gen/<mode>/ 下（路径 txt + yaml），出问题时可以人工
    打开检查——比如「yaml 里第一行图的路径，把 images 换成 labels 看文件在不在」。
    """
    data_root = Path(data_root)
    images_subdir = resolve_images_subdir(condition, mode, data_root)

    # 8 类口径的自然条件：先把这批图硬链接进 val8/images/，再照常用它拼路径。
    # 坏图条件不用这一步——generate_corrupt 已经把坏图硬链接进 corrupt8/images 了。
    if mode == "bdd8coco" and images_subdir == MODE_IMAGE_DIR[mode]:
        ensure_image_mirror(names, data_root / "val" / "images",
                            data_root / "val8" / "images")

    gen_dir = data_root / "gen" / mode
    txt = make_paths_txt(names, images_subdir, gen_dir / f"{condition}.txt",
                         data_root)
    return make_yaml(condition, txt, MODE_CLASS_NAMES[mode], data_root,
                     gen_dir / f"{condition}.yaml")


# ---- 云端训练名单（D9） -------------------------------------------------------

def build_train_lists(data_root, out_dir=None):
    """把两份训练名单变成带路径的 txt，供 configs/m0_local.yaml 的 train/val 指向。

    名单本体（train_clear_all.txt / trainval_1k.txt）随仓库走（git clone 自带），
    路径前缀按云端数据根目录拼 → 输出 <out>/train_clear_all_paths.txt 等。
    """
    data_root = Path(data_root)
    out_dir = Path(out_dir or data_root / "splits_train")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train_clear_all.txt", "trainval_1k.txt"):
        names = read_names(REPO_SPLITS / name)
        make_paths_txt(names, "train/images", out_dir / name.replace(".txt", "_paths.txt"),
                       data_root)
        print(f"  {out_dir / name.replace('.txt', '_paths.txt')}（{len(names)} 张）")
    return out_dir


if __name__ == "__main__":                 # 云电脑上：python -m rvkit.harness.datasets --data-root ...
    import argparse

    ap = argparse.ArgumentParser(description="生成云端训练用的带路径名单（splits_train/*_paths.txt）")
    ap.add_argument("--data-root", required=True,
                    help="云端数据根目录（与 configs/m0_local.yaml 的 path 一致）")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认 <data-root>/splits_train）")
    a = ap.parse_args()
    build_train_lists(a.data_root, a.out_dir)
