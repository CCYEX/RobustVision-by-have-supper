#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_m2_live.py - 一键训练 M2 旗舰 + 实时仪表盘。

M2 = 练习盘 data/train_m2/（19,463 张 = 11,454 原图 + 4,009 坏副本 + 4,000 真实夜雨）
——最终发布的模型。数据多 70%，轮数降到 30（早停 patience 仍为 15）。
用法：
    D:/Coding/DL_Env/PuTong_P3.11.15/python.exe experiments/train_m2_live.py
    断点续训：python experiments/train_m2_live.py --resume
"""

import argparse

from train_live import run_live

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="训练 M2 旗舰（实时仪表盘）")
    p.add_argument("--resume", action="store_true", help="从 runs/m2/weights/last.pt 续训")
    a = p.parse_args()
    run_live(name="m2", data_cfg="configs/m2_local.yaml", epochs=30, resume=a.resume)
