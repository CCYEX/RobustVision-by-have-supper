#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""adapter.py - 「翻译官」：我们和 Ultralytics YOLO 之间的传话人（D4）。

为什么要有它：Ultralytics 的 val() 返回一个大对象，里面数字藏得很深；
以后我们换模型（yolo11s.pt → 自己的 m0.pt → ONNX）时也不想到处改调用代码。
所以把「加载模型 + 考一次试 + 取分数」封装成一个类，外面永远只拿到一个小字典：

    {"map": mAP50-95, "map50": mAP50, "p": 精确率, "r": 召回率,
     "per_class": [每个类各自的 mAP50-95]}

map 是总分（我们主要看它）；per_class 是「哪类掉得狠」图的原料（D10 用）。
本模块 import 时不加载任何模型——ultralytics 在 __init__ 里才导入，
`import rvkit` 保持轻快（D8 的「假装外人」测试会感谢这一点）。
"""

from __future__ import annotations


class UltralyticsAdapter:
    """包住一个 YOLO 模型，对外只暴露 val() 一个动作。"""

    def __init__(self, weights, device="cpu"):
        """weights = 权重文件路径（如 "yolo11s.pt"）；device = "cpu" 或 "0"（第一块显卡）。"""
        # 延迟导入：直到真正要加载模型才 import ultralytics（顺带下载缺失的权重）
        from ultralytics import YOLO

        self.model, self.device = YOLO(weights), device

    def val(self, yaml_path, imgsz=640, batch=8):
        """对着一份 datasets.make_yaml() 生成的说明书考一次试，返回小字典分数。

        imgsz=640：把图缩到 640 边长再考（训练评测统一，分数才可比）；
        batch=8：一次喂 8 张；CPU 也就吃 2~3GB 内存，显卡则远小于 8GB 显存上限。
        verbose=False：别把 Ultralytics 自己的长篇报告刷屏，我们只要数字。
        """
        r = self.model.val(data=str(yaml_path), imgsz=imgsz, device=self.device,
                           batch=batch, verbose=False)
        # maps 数组按"模型类别数"取长，没有标准答案的类会被填成总分均值——
        # 所以每类分数必须按 ap_class_index（真正出现过的类）过滤后才干净。
        idx = [int(i) for i in r.box.ap_class_index]
        all_maps = [float(m) for m in r.box.maps]
        return {
            "map": float(r.box.map),          # 总分 mAP50-95（主要看的那个）
            "map50": float(r.box.map50),      # 宽松版：IoU≥0.5 就算对
            "p": float(r.box.mp),             # 精确率：报出来的框有多少是对的
            "r": float(r.box.mr),             # 召回率：该找的找到了多少
            # 每个类的 mAP50-95（只含有标注的类；顺序与 per_class_index 一一对应）
            "per_class": [all_maps[i] for i in idx],
            "per_class_index": idx,           # 类别 id → 用数据集类别表查名字
        }
