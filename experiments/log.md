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
- 待办：convert.py / make_splits.py / audit 收尾（交叉表 + M2 抽样 + 框尺寸）+ fixture 测试
- GPU 累计：0 h / 6 h
```
