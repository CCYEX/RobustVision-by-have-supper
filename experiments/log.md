# 执行日志

```
## D1 08-27
- 做了：BDD100K 官方注册完成；repo 骨架建成并推送 GitHub（RobustVision-by-have-supper，README 用远程版本）
- 数据第一版：镜像下载 val.zip + bdd100k_labels.zip（逐图格式），验证通过
- 格式坑记录：多任务标注混入（lane/drivable area）；类名 motor/bike；train 逐图 frames[0].objects 结构
- GPU 累计：0 h / 6 h
```

```
## D2 08-28
- 数据换源：重新下载官方格式数据到 repo 的 data/（train/val/test 三分包，train/val 各带总表 JSON）
  · train: 70,000 图 + bdd100k_labels_images_train.json (1.45GB)；val: 10,000 图 + 总表 (208MB)；test: 20,000 图无标注（项目不用）
  · 结论：**"train 逐图合并"任务取消**，convert.py 只剩一条解析路径；旧位置 D:\Coding\RobustVision\data 已冗余可删
- 全量普查（val 精确 / train 流式）：
  · val: name∩images 10000/10000 ✓；零标注图 0；类别含 10 检测类 + lane(75730) + drivable area(17981) → 过滤规则不变
  · val 条件: clear 5346 | overcast 1239 | rainy 738 | snowy 769 | partly cloudy 738 | foggy 13 | undefined 1157
              daytime 5258 | night 3929 | dawn/dusk 778 | undefined 35
  · train 条件: night 27972 | rainy 5070 | snowy 5549 | clear 37344 | daytime 36729 → M2 真实偏移池充足
  · train 有 137 图无标注记录（复核：原文件 0 次出现）→ 跳过，不参与条件抽样
- 条件映射冻结（写进 data_card）：clean_day = clear∩daytime；night = timeofday==night；rain = rainy；
  snow = snowy（val 769 张，可作正式报告条件）；fog = foggy（13 张，仅参考点）；undefined 一律排除在命名条件外
- housekeeping：.gitignore 增补 data/train|val|test、data/yolo、*.zip；data/README.md 更新布局说明
- convert.py 完成（作者：本人；review 通过）：
  · 实测 val 7.9s / train 57.8s（含 1.45GB json.load，内存无压力）
  · 产出 labels：val 10,000 + train 69,863（137 无记录图自动跳过）；空 txt 0——本数据集每图至少 1 个有效框
  · train 每类保留（精确值，以此为准）：person 91,349 / rider 4,517 / car 713,211 / truck 29,971 /
    bus 11,672 / train 136 / motor 3,002 / bike 7,210 / traffic light 186,117 / traffic sign 239,686；
    丢弃 lane 528,643 + drivable area 125,723（全部因无 box2d 或不在白名单）
  · 备注：D2 早上的流式普查在块边界存在 ≤30 的重复计数误差，统计以 convert.py 输出为准
- 决策（make_splits 发现）：val 的 clear∩daytime 仅 1,764（clear 多为夜景、白天多阴天），test 侧 clean_day=1,235 不足原定 2,000
  → **corrupt_base = clean_day 的 test 全集（同图配对评测）**，不从 calib/train 凑数（防泄漏/防训练污染），不放宽定义（训练端评测端口径必须一致）
  → 连锁检查：calib 侧 529 张拟温度足够；night test ~2,750 不受影响；train clear∩daytime ~3 万张不受影响
- D2 完成：make_splits.py + audit.py + 6 条 fixture 测试全绿；验收复核（独立重算）全部一致
  · calib 2999 + test 7001 = 10000，交集 0；六条件集合与独立重算完全一致（night 2704 = 排除 66 张 weather=undefined 夜景后，符合冻结规则）
  · corrupt_base 与 clean_day 集合相等（1235）；m2 4000 行无重复且 4000/4000 存在于 train 标注
  · 框尺寸实测：traffic light 88.3% / traffic sign 74.9% / person 48.4% 为小目标（退化分析的参照基线）
- ⚠ 重大发现（来自 audit 交叉表）：train 的 clear∩daytime 仅 **12,454 张**（clear 的 22,884 张是夜景），不足原定 2 万
  → 决策：训练池改为**全量 12,454 不抽样**；留 1K trainval → 实际训练 11,454
  → 连锁调整：M1 副本 30% ≈ 3.4K；M2 = 11.4K + 4K 合成 + 4K 真实 ≈ 19.5K/epoch；GPU 核心预算下调为 3.5–4.7 h
  → 核心设定不变（模型从未见过夜间/雨天）；「审计发现池子不足→改全量」本身是数据审计价值的案例
- 明日 D3：生成 trainval_1k.txt + train_clear_all.txt（11,454）两份训练清单 + 本机环境 + run_s1.sh
- GPU 累计：0 h / 6 h
```
