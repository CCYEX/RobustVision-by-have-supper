#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_splits.py — 从 val 的 10000 张图生成校准/测试切分与条件子集清单。

输入:
    data/val/annotations/bdd100k_labels_images_val.json   (10000 张图的属性表)

输出（每行只写文件名，不含路径 —— 保证本机 / 云电脑路径可移植）:
    data/splits/calib.txt                校准集（约 3K）
    data/splits/test.txt                 测试集（约 7K）
    data/splits/conditions/clean_day.txt test 中 clear ∩ daytime
    data/splits/conditions/night.txt     test 中 timeofday == night
    data/splits/conditions/rain.txt      test 中 weather == rainy
    data/splits/conditions/snow.txt      test 中 weather == snowy
    data/splits/conditions/dawn_dusk.txt test 中 timeofday == dawn/dusk
    data/splits/conditions/fog.txt       test 中 weather == foggy
    data/splits/corrupt_base.txt         clean_day 中抽 2000 张（损坏增强底图）

约定:
    - 每个 weather × timeofday 组合内部打乱后按 3:7 切 → 两边各天气比例一致
    - weather / timeofday 为 undefined 的图不进任何命名条件文件
    - 所有随机操作统一使用 random.Random(42)
"""

# ---- 导入与说明 --------------------------------------------------------------
# 阅读顺序建议：先看文件最上面的 docstring（三引号字符串），它说明输入/输出/约定；
# 再从上往下顺着 main() -> 各函数读。下面 import 的都是 Python 标准库（无需 pip 安装）。

from __future__ import annotations   # 让类型注解延迟求值（本脚本未用注解，写上是统一风格）
import json                          # 处理 JSON：json.load() 把文件内容解析成 Python 的 dict/list
import random                        # 伪随机数：random.Random(42) 生成"seed 固定、结果可复现"的随机源
from collections import defaultdict  # 增强版字典：访问不存在的键时自动填默认值（list/int），省去 if 判断
from pathlib import Path             # 路径对象：用 "/" 运算符拼接路径，Windows/Linux 通用

# ---- 常量定义 ---------------------------------------------------------------
# 大写变量名是 Python 约定：表示"常量"，整个程序里不应被改动。
# 好处：所有"魔法数字"（0.3、2000、路径…）集中在这里，想改只需改这一处。

# __file__ 是"当前这个 .py 文件本身的绝对路径"。
# Path(...) 包成路径对象；.resolve() 解析掉 .. 和符号链接；.parent 取上一级目录。
# 于是 DATA_ROOT 恒等于存放本脚本的 data/ 目录 —— 从任何目录运行脚本都能正确定位。
DATA_ROOT = Path(__file__).resolve().parent

# Path 的 "/" 不是除法，而是"拼接路径"：DATA_ROOT / "val" / "annotations" / "xx.json"
# 等价于 Windows 的 data\val\annotations\xx.json，但写法跨平台通用。
VAL_ANNOTATION = DATA_ROOT / "val" / "annotations" / "bdd100k_labels_images_val.json"
SPLITS_DIR = DATA_ROOT / "splits"            # 输出目录：data/splits/
COND_DIR = SPLITS_DIR / "conditions"         # 条件子目录：data/splits/conditions/

CALIB_RATIO = 0.3                            # 3:7 切分中，calib 占 30%、test 占 70%
CORRUPT_N = 2000                             # corrupt_base 想抽 2000 张（受数据上限影响）

# CONDITIONS 是"命名条件清单"，列表顺序 = 后续打印、写文件的一致顺序
CONDITIONS = ["clean_day", "night", "rain", "snow", "dawn_dusk", "fog"]

# random.Random(42)：创建一个独立的随机数生成器，seed=42。
# 同一种子 → 每次运行 shuffle/sample 的结果完全一样 → 整个流程可复现。
rng = random.Random(42)


# ---- 核心逻辑 ---------------------------------------------------------------

def load_attrs(path):
    """读取 val 标注，返回一个字典：{图名: (weather, timeofday)}。

    例如 {"a.jpg": ("clear", "daytime"), "b.jpg": ("rainy", "night"), ...}
    图名是键，天气/时段元组是值。后续所有函数都靠这个字典查图属性。
    """
    # with ... as f：上下文管理器，保证文件读完自动关闭（即使中途报错也关）。
    # "r" 表示只读；encoding="utf-8" 指定按 UTF-8 解码，中文/特殊符号不会乱码。
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)   # 把整个 JSON 数组解析成 Python 列表，每个元素是一张图的 dict

    # 下面这段是"字典推导式"：{键表达式: 值表达式 for 变量 in 可迭代对象}
    # 等价于手写 for 循环往空字典里赋值，只是更简洁。逐行拆解：
    return {
        rec["name"]: (                         # 键：图名字符串，如 "a.jpg"
            rec["attributes"].get("weather", "undefined"),      # 值第 1 项：天气
            rec["attributes"].get("timeofday", "undefined"),    # 值第 2 项：时段
        )
        for rec in records                     # 遍历 JSON 列表里的每一张图 rec
    }
    # 语法说明：.get(键, 默认值) 是 dict 的方法 —— 有该键返回它的值，
    # 没有就返回默认值。这里用 "undefined" 兜底，即使某张图缺属性也不会 KeyError 崩溃。


def split_3_7(attrs):
    """按 weather × timeofday 组合做"分层"3:7 切分。

    为什么分层？若直接对 1 万张整体随机切，可能某个组合（如 foggy+night）全落在一侧，
    两边天气比例就失衡。所以先按 (weather, timeofday) 分组，每组内部再 3:7 切，
    这样每个组合在 calib/test 两侧占比相同，整体天气分布两边一致。
    返回 4 个值：calib 列表、test 列表、各组合进 calib/test 的计数字典。
    """
    # defaultdict(list)：访问不存在的键时自动创建空列表，不会 KeyError。
    # 也就是 groups[(w, t)].append(name) 首次遇到某组合时会自动初始化为 []。
    groups = defaultdict(list)
    for name, (w, t) in attrs.items():      # .items() 同时取出"键、值"；(w, t) 是元组解包
        groups[(w, t)].append(name)         # 键 = 元组 (weather, timeofday)；值 = 该组合的图名列表

    calib, test = [], []                     # 两个空列表，分别装分好的图名
    calib_counts, test_counts = defaultdict(int), defaultdict(int)   # 记录每组合各分几张

    for combo in sorted(groups):             # 遍历所有组合；sorted 固定顺序 → 结果可复现
        names = groups[combo]                # 取该组合的全部图名
        rng.shuffle(names)                   # 在"原列表上就地"打乱顺序（seed 固定 → 可复现）

        k = int(round(len(names) * CALIB_RATIO))   # 前 30% 的个数：round 四舍五入、int 转整数
        calib.extend(names[:k])              # 切片 [ :k] = 取前 k 个 → 进 calib
        test.extend(names[k:])               # 切片 [k: ] = 取第 k 个及之后 → 进 test
        calib_counts[combo] = k              # 记录该组合进 calib 的数量
        test_counts[combo] = len(names) - k  # 剩余全部进 test

    return calib, test, calib_counts, test_counts   # 一个函数返回 4 个值（自动打包成元组）
    # 注意：names[:k] 与 names[k:] 首尾相接、互不重叠，恰好完整覆盖整个 names 列表。


def select_conditions(test, attrs):
    """从 test 里挑出各命名条件的图名；weather/timeofday 为 undefined 的图不进任何条件。

    返回字典：{条件名: [图名, ...]}。注意一张图可同时满足多个条件
    （例如 rainy+night 会同时进 rain 和 night），这是符合预期的。
    """
    # 字典推导式：为 CONDITIONS 里每个名字建一个空列表 → 结果形如
    # {"clean_day": [], "night": [], "rain": [], ...}
    sets = {key: [] for key in CONDITIONS}

    for name in test:                # 只遍历 test（条件全部从 test 里选，不含 calib）
        w, t = attrs[name]           # 按图名查属性字典，解包成 (weather, timeofday)

        if w == "undefined" or t == "undefined":
            continue                 # continue = 跳过本次循环剩余代码，直接处理下一张图
                                     # （undefined 属性的图不属于任何命名条件）

        # 下面每个 if 都独立判断、互不冲突 → 一张图可以进多个条件列表
        if w == "clear" and t == "daytime":   # and：两个条件必须同时满足
            sets["clean_day"].append(name)    # clean_day = 晴天 且 白天
        if t == "night":                      # night 只看时段，不管天气
            sets["night"].append(name)
        if w == "rainy":
            sets["rain"].append(name)
        if w == "snowy":
            sets["snow"].append(name)
        if t == "dawn/dusk":                  # dawn/dusk = 黎明/黄昏
            sets["dawn_dusk"].append(name)
        if w == "foggy":
            sets["fog"].append(name)

    return sets


def write_list(path, names):
    """把一个图名列表写成 txt 文件，每行一个文件名（不带路径）。

    path 是目标文件（如 data/splits/calib.txt），names 是要写入的图名列表。
    """
    # path.parent 是目标文件所在目录（如 data/splits/）。
    # mkdir(parents=True) 递归创建缺失的父目录；exist_ok=True 表示目录已存在也不报错。
    path.parent.mkdir(parents=True, exist_ok=True)

    # "".join(...)：把可迭代的字符串片段拼接成一个长字符串。
    # 生成器表达式 f"{n}\n" 为每个图名造出 "图名.jpg\n"（\n 是换行符）。
    # 拼接结果 = "a.jpg\nb.jpg\nc.jpg\n"，写进文件正好一行一个图名。
    # write_text(...) 一次性写入整个字符串；encoding="utf-8" 统一编码。
    path.write_text("".join(f"{n}\n" for n in names), encoding="utf-8")


# ---- 打印 -------------------------------------------------------------------

def print_combination_table(calib_counts, test_counts):
    """在终端打印"每个 weather × timeofday 组合分到 calib/test 各几张"的表格。"""
    print("\n===== weather × timeofday 切分数量表 =====")

    # f-string（f"..."）里可以用 {表达式} 直接嵌入变量或运算结果。
    # 格式说明符 :<14 表示"左对齐、占 14 格"，:>7 表示"右对齐、占 7 格"，
    # 这样各列对齐、像打印表格。这一行是表头。
    print(f"{'weather':<14}{'timeofday':<12}{'calib':>7}{'test':>7}{'total':>7}")

    # set(calib_counts) | set(test_counts)：两个键集合的并集（| 是集合"或"）
    # → 保证所有出现过的组合都被打印；sorted(...) 固定打印顺序。
    for combo in sorted(set(calib_counts) | set(test_counts)):
        w, t = combo            # combo 是元组 (weather, timeofday)，解包成 w、t
        c, te = calib_counts[combo], test_counts[combo]
        print(f"{w:<14}{t:<12}{c:>7}{te:>7}{c + te:>7}")   # 一行 = 一个组合

    # sum(...) 对可迭代对象求和；.values() 取出字典的所有值
    c_total = sum(calib_counts.values())
    t_total = sum(test_counts.values())
    print(f"{'合计':<26}{c_total:>7}{t_total:>7}{c_total + t_total:>7}")   # 末行：合计


def print_condition_sizes(sets, corrupt_base):
    """打印各命名条件文件里的图数量，以及 corrupt_base 的规模。"""
    print("\n===== 命名条件（全部来自 test）=====")
    for key in CONDITIONS:                        # 按 CONDITIONS 顺序逐项打印
        # len(sets[key])：该条件列表的长度，也就是图的数量
        print(f"  {key:<12}{len(sets[key]):>6} 张")
    print(f"  corrupt_base   {len(corrupt_base):>6} 张（自 clean_day 抽取）")


# ---- 入口 -------------------------------------------------------------------

def main():
    """程序入口：按顺序完成 读取 → 切分 → 选条件 → 抽 corrupt_base → 打印统计。"""
    # VAL_ANNOTATION.name：Path 对象的 .name 只取最后一段文件名（不含目录）
    print(f"加载 {VAL_ANNOTATION.name} …（约 208MB，需十几秒）")
    attrs = load_attrs(VAL_ANNOTATION)      # 第 1 步：读 JSON，得到 {图名: (天气, 时段)}
    print(f"共 {len(attrs)} 张图")

    # 第 2 步：3:7 分层切分，一次拿回 4 个结果（元组解包）
    calib, test, calib_counts, test_counts = split_3_7(attrs)
    write_list(SPLITS_DIR / "calib.txt", calib)    # 写校准集清单
    write_list(SPLITS_DIR / "test.txt", test)      # 写测试集清单

    # 第 3 步：从 test 里挑命名条件，并逐个写成 conditions/ 下的 txt
    sets = select_conditions(test, attrs)
    for key in CONDITIONS:                         # 遍历 6 个条件名
        write_list(COND_DIR / f"{key}.txt", sets[key])   # 如 conditions/night.txt

    # 第 4 步：corrupt_base = 从 clean_day 里抽 2000（rng.sample 是不重复抽样）
    clean_day = sets["clean_day"]
    n = min(CORRUPT_N, len(clean_day))   # min()：取"2000"和"实际数量"里更小的那个
    corrupt_base = rng.sample(clean_day, n)   # sample(列表, n)：不重复地随机抽 n 个
    write_list(SPLITS_DIR / "corrupt_base.txt", corrupt_base)
    if n < CORRUPT_N:                    # 数据不够 2000 时如实提示（已确认接受上限）
        print(f"[提示] test 中 clean_day 仅 {len(clean_day)} 张（< {CORRUPT_N}），"
              f"corrupt_base 取全部 {n} 张（数据上限，已确认接受）")

    # 第 5 步：打印两张统计表，方便人工核对
    print_combination_table(calib_counts, test_counts)
    print_condition_sizes(sets, corrupt_base)
    print(f"\n完成：清单已写入 {SPLITS_DIR}")


if __name__ == "__main__":
    # 这个 if 是 Python 惯用法：只有"直接运行本文件"时才执行 main()。
    # 若别的脚本 import make_splits 进来，则不会自动跑，只把上面的函数暴露出去供复用。
    main()
