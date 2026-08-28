#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_train_lists.py — 从 train 标注总表生成两份「晴天白天」训练名单（计划书 D2 任务）。

背景：
    train 里 weather==clear 且 timeofday==daytime 的图实测共 12,454 张，决定全部用上：
    - 前 1,000 张（seed 42 打乱后）→ data/splits/trainval_1k.txt      随堂测验
    - 其余 11,454 张               → data/splits/train_clear_all.txt  正式训练

运行：python data/make_train_lists.py
（train 的 JSON 约 1.45GB，加载需一两分钟属正常现象）

约定：
    - 清单每行只写文件名、不带路径（与 calib.txt 等一致 → 本机 / 云电脑路径可移植）
    - 所有随机操作统一使用 random.Random(42)，同种子 → 每次打乱结果完全一样、可复现
    - 末尾自查：两文件行数 1,000 / 11,454，合计 12,454，互相无重复
"""

from __future__ import annotations   # 类型注解延迟求值，与项目其他脚本保持统一风格

import json                          # json.load()：把整个标注总表解析成 Python 列表
import random                        # random.Random(42)：seed 固定的伪随机数生成器
from pathlib import Path             # 路径对象：用 "/" 拼接路径，Windows/Linux 通用

# ---- 常量定义 ---------------------------------------------------------------
# 大写 = Python 惯例的"常量"；所有路径和魔法数字集中在这里，想改只改这一处。

DATA_ROOT = Path(__file__).resolve().parent          # 本脚本所在的 data/ 目录
TRAIN_ANN = DATA_ROOT / "train" / "annotations" / "bdd100k_labels_images_train.json"
SPLITS_DIR = DATA_ROOT / "splits"                    # 输出目录：data/splits/

QUIZ_N = 1_000                                       # 随堂测验规模：打乱后的前 1,000 张

# 计划书 D2 实测数字：clear ∩ daytime 应为 12,454 张。
# 若跑出来不是这个数，说明数据版本 / 过滤条件有问题 → 立即报错，别带病继续。
EXPECTED_TOTAL = 12_454

rng = random.Random(42)                              # 统一随机源：seed 固定 = 可复现


# ---- 核心逻辑 ---------------------------------------------------------------

def load_clear_day_names(path):
    """读取 train 标注总表，返回 weather==clear 且 timeofday==daytime 的图名列表。

    每条记录形如 {"name": "xxx.jpg", "attributes": {"weather": ..., "timeofday": ...}, ...}，
    只需要 name 和 attributes 两个字段，标注框（labels）虽大但不读进结果，不影响速度。
    """
    # with ... as f：上下文管理器，保证读完后文件必定关闭；"r" 只读 + utf-8 解码。
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)      # 1.45GB 的 JSON 整体载入，需一两分钟属正常

    names = []                      # 收集满足条件的图名
    for rec in records:
        # .get(键, 默认值)：字段缺失时返回默认值而不抛 KeyError —— 个别记录
        # 可能没有 attributes，这里兜底成空字典，让下一行的 .get 安全返回 None。
        attr = rec.get("attributes", {})
        # 两个条件必须同时成立（and）：晴天 且 白天 才入选。
        if attr.get("weather") == "clear" and attr.get("timeofday") == "daytime":
            names.append(rec["name"])
    return names


def write_list(path, names):
    """把图名列表写成 txt，每行一个文件名（不带路径）。"""
    # 先确保输出目录存在：mkdir(parents=True) 递归建目录，exist_ok=True 已存在也不报错。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 生成器表达式为每个图名产出 "xxx.jpg\n"，"".join 拼成整篇文本一次写入。
    path.write_text("".join(f"{n}\n" for n in names), encoding="utf-8")


def check_outputs(quiz_path, all_path):
    """自查：从磁盘把两个清单读回来核对（不依赖内存里的变量，防止写入环节出错漏检）。

    检查三件事，全部通过打印 ✓，任何一条不过立即抛异常终止：
        ① trainval_1k.txt 恰好 1,000 行
        ② train_clear_all.txt 恰好 11,454 行
        ③ 两文件合计 12,454 个图名且集合去重后仍是 12,454 → 无重复
           （行数合计 = 集合大小，说明两文件内部和互相之间都没有重复图名）
    """
    # read_text 读回整份清单；splitlines() 按行切开成列表，末尾换行不会产生多余空行。
    quiz = quiz_path.read_text(encoding="utf-8").splitlines()
    rest = all_path.read_text(encoding="utf-8").splitlines()

    # assert 断言：条件为真则静默通过；为假则抛 AssertionError 并显示消息，程序终止。
    assert len(quiz) == QUIZ_N, \
        f"trainval_1k.txt 行数 {len(quiz)} ≠ {QUIZ_N}"
    assert len(rest) == EXPECTED_TOTAL - QUIZ_N, \
        f"train_clear_all.txt 行数 {len(rest)} ≠ {EXPECTED_TOTAL - QUIZ_N}"

    # set() 去重；| 是集合并集。行数合计 == 去重后总数 ⇔ 全部图名互不重复。
    n_lines, n_unique = len(quiz) + len(rest), len(set(quiz) | set(rest))
    assert n_lines == EXPECTED_TOTAL, f"两文件合计 {n_lines} 行 ≠ {EXPECTED_TOTAL}"
    assert n_unique == EXPECTED_TOTAL, \
        f"发现重复：合计 {n_lines} 行，去重后仅 {n_unique} 个不同图名"

    print(f"✓ 自查通过：{quiz_path.name} {len(quiz)} 行 + "
          f"{all_path.name} {len(rest)} 行 = {n_lines} 行，无重复")


# ---- 入口 -------------------------------------------------------------------

def main():
    """流程：过滤 → 打乱 → 切两份 → 写文件 → 自查。"""
    print(f"加载 {TRAIN_ANN.name} …（约 1.45GB，需一两分钟）")
    names = load_clear_day_names(TRAIN_ANN)
    print(f"clear ∩ daytime 共 {len(names)} 张（计划书实测应为 {EXPECTED_TOTAL}）")

    # 数量对不上计划书实测值 → 大概率数据版本不对或过滤写错，宁可停下也别继续。
    assert len(names) == EXPECTED_TOTAL, \
        f"过滤结果 {len(names)} 张 ≠ 计划书实测 {EXPECTED_TOTAL} 张，请检查数据/条件"

    rng.shuffle(names)              # 就地打乱顺序；seed 42 固定 → 每次结果一致

    # 切片：[:1000] 前 1,000 张做随堂测验；[1000:] 剩下的全部做正式训练。
    # 两段首尾相接、不重不漏，天然互不重复（自查仍会再验证一遍）。
    quiz, rest = names[:QUIZ_N], names[QUIZ_N:]

    quiz_path = SPLITS_DIR / "trainval_1k.txt"
    all_path = SPLITS_DIR / "train_clear_all.txt"
    write_list(quiz_path, quiz)
    write_list(all_path, rest)
    print(f"已写出：{quiz_path.name}（{len(quiz)}）、{all_path.name}（{len(rest)}）")

    check_outputs(quiz_path, all_path)
    print(f"完成：清单已写入 {SPLITS_DIR}")


if __name__ == "__main__":
    # 只有直接运行本文件时才执行 main()；被别的脚本 import 时只暴露函数、不自动跑。
    main()
