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

```
## D4+D5 08-29（提前完成）
- 做了：评测引擎三件套 datasets/adapter/runner + convert.py --mode bdd8coco（编号对齐 COCO）+ val8/ 双目录
- 数字（yolo11s × 8类 × mini300）：clean_day mAP50-95=0.2277；night −22.5%；fog −39.6%；snow −17.8%；rain +6.8%（迷你集偏白天所致，全量以云端为准）
- 修坑：Ultralytics r.box.maps 对无标注类填均值 → adapter 按 ap_class_index 过滤并返回 per_class_index
- GPU 累计：0 h / 6 h
```

```
## D6+D7 08-29（提前完成 → 9/2 门禁达标）
- 做了：corruptions.py（6 种 albumentations + 手写物理散射雾）+ generate_corrupt.py（双口径 corrupt/+corrupt8/ 硬链接）+ report.py + rvkit checkup CLI
- 决策：库版 RandomFog 视觉否决（漂浮白斑，见 fog_shootout.jpg）→ 自写大气散射模型（远浓近清）；albumentations 钉 >=1.4.24,<1.5
- 验证：33 条测试全绿；确定性磁盘级复核（同图同坏法重新生成逐字节一致）；硬链接 inode 验证；门禁预演一条命令 4 产物 ✓（已清理）
- 数字：最狠类 traffic light 平均 −49.1%（呼应数据卡 88.3% 小目标）
- 明日 D8：假装外人测试 + S1 物料（configs/m0_clear_day.yaml、make_aug_copies.py、predict_cache.py）
- GPU 累计：0 h / 6 h
```

```
## D9 08-29深夜~08-30晨（提前，本机路线）
- 做了：页面文件扩容（WinError 1455 根因：配额耗尽，三个怪现象同源）→ 重启 → train_m0.py 本机训 M0
  · 4060 约束三件套：batch=8（三模型统一可比）/ workers=0 / cache=ram；GPU 5.6GB 稳定、利用率 99%
- 数字：**M0 训练完成，40 轮 4.934h**；trainval_1k 上 mAP50-95=**0.311**、mAP50=0.555、P=0.717、R=0.513（预测区间 0.28~0.36 中段 ✓）
  · 按类：car 0.503 / truck 0.469 / bus 0.465 / traffic sign 0.347 / traffic light 0.294 / person 0.292（最弱两类正是小目标占比最高的）
- 修坑：Ultralytics 相对 project="runs" 被拼进 runs_dir → 输出嵌套 runs/detect/runs/m0；权重已挪回 runs/m0/，train_m0.py 改绝对路径（M1/M2 免踩）
- 环境：rvkit 已 pip install -e . --no-deps 进 PuTong 环境
- 明日（白天）：全量评测 m0_full（13 条件）+ predict_cache 10K 张 → 校准原料就位 → 晚上开训 M1
- GPU 累计：本机 4.9 h（自购整机，云预算 0/6 h 未动）
```

```
## D10 评测 08-30晨（M0 全量退化矩阵出炉）
- 数字（results/m0_full.md，test 侧 13 条件，10 类口径）：
  clean_day 0.3022（基准）｜ night −26.2% ｜ rain −9.0% ｜ snow −15.9% ｜ dawn_dusk +4.2%（略升，如实记录）
  ｜ fog −15.2%（仅 9 张，仅供参考）｜ 低光 −11.3% ｜ 运动模糊 −26.6% ｜ 高斯噪声 −34.3%（最狠）
  ｜ 合成雾 −2.5% ｜ 合成雨 −27.7% ｜ 低清 −7.1% ｜ JPEG −2.6%
  最受伤的类：motor（平均 −36.0%）→ RQ1 的答案成型
- 坑二连（predict_cache）：① 1 万条列表输入 → Ultralytics 全量预解码进内存 ≈28GB → MemoryError
  ② 改 256 分块后仍是同 loader：整块当一个大 batch → CUDA OOM 3.75GB
  → 修复：CHUNK=256 + predict 显式 batch=16（已改，待回跑）
- 状态：评测 4 件套已出（md/csv/perclass/热力图）；cache/m0.parquet 待重跑（一条命令，~8 分钟）
- GPU 累计：本机 4.9h 训练 + 评测 ~0.5h；云预算 0/6h 未动
```

```
## D9(校准)+D10(案例) 08-30上午（全部非训练任务收工）
- 做了：predict_cache 修两个 Ultralytics 列表输入坑（RAM 预解码 28GB / batch 参数不生效）→ CHUNK=8 出 cache/m0.parquet（425,697 框/万张）
- 校准：matching.py（贪心 IoU 判卷）+ calibrate.py（温度拟合/分条件 ECE/可靠性图）
  · 最佳 bug：for l in read_text() 逐字符迭代 → calib 集变 21 个字符 → NLL=nan；修 = splitlines()
  · M0 数字：T=0.762（<1 略谦虚，反预期）；NLL 0.3754→0.3650；ECE 全条件 −12~29%（雪 −28.5/雨 −24.7/夜 −20.0）；mAP 逐位不变 ✓
- 案例：plot_cases.py → 12 张夜间漏检行人（最多 6 人/张，最高把握 0.157）→ results/qualitative/
- 文档：case_study.md RQ1/RQ3 写实（含"晨昏 +4.2%"与"夜反而更诚实"两个诚实注记）
- 数据：train_m1 14,890 / train_m2 19,463 就绪；M1/M2 校准只需换缓存文件重跑
- 剩余：今晚 M1 → 明晚 M2 → 阶梯表+冻结 → 写作文 → 发布（窗口 9/6–9/11，硬底线 9/14）
- GPU 累计：本机 ~5.5h（训练 4.93 + 评测/缓存 ~0.6）；云预算 0/6h 未动
```

```
## D11(M1) 08-31凌晨（训练+评测+校准完成，分析待续）
- M1 训练：40 轮全程，trainval_1k mAP50-95=0.316（M0=0.311，clean 不降反升 ✓）
- 全量评测（results/m1_full.md）RQ2 关键发现：
  · 合成损坏大幅恢复：gauss_noise −34.3%→−15.1%（+0.058）、rain_s2 −27.7%→−13.0%（+0.044）、motion_blur −26.6%→−21.4%
  · 但真实夜间几乎没动：−26.2%→−27.0% —— 合成增强是"对称解药"，迁移不到真实夜间 → M2 的科学理由成立
  · 自然雨小幅迁移（−9.0%→−7.1%）；clean_day 持平（0.3022→0.3024，零牺牲）
  · 最伤类 motor 平均掉分 −36.0%→−25.9%（增强对最脆弱类帮助最大）
- 校准 M1：T=0.770，NLL 0.3769→0.3673，全条件 ECE −10.7~30.4%，mAP 不变 ✓
- 待续：case_study RQ2 成文 + 战绩 + ladder 表（M2 后）
- GPU 累计：本机 ~11h（M0 4.93 + M1 ~6.1）；云预算 0/6h 未动
```

```
## D13+D14(M2) 08-31晚（三模型阶梯表出炉，成绩冻结）
- M2 训练：30 轮 6.467h（13:30-20:03），trainval_1k mAP50-95=0.313（M0 0.311 / M1 0.316，稀释代价极小）
- 全量评测+缓存+校准一条龙完成（ladder.csv/md 由 experiments/make_ladder.py 生成）
- ★ 主结果（results/ladder.md）：
  · 夜间：M0 0.2231 → M1 0.2206（合成增强不迁移，复证）→ M2 0.2546（+3.40 分）
    真实夜间退化从 −26.2% 收窄到 −14.0% —— RQ2b 答案：真实与合成互补，各治一类偏移
  · 自然偏移平均：M2 +1.44 分（四条件全正）；雨几乎不再退化（vs clean 仅 −0.6%）
  · 代价明码：clean −0.61 分（≤1.0 达标）、合成损坏 −0.50 分（M1 仍是合成之王）
  · M2 校准：T=0.808；雪/雾 ECE −19~20%，但夜间 ECE 微升 −4.6%
    → 解释：见过真夜景后夜间置信度本就诚实，全局温度轻微失配（"越熟悉越可信"的反证）
- 成绩冻结：M0/M1/M2 全部数字定稿（P0/P1 验收线全过：夜间提升≥1.0 ✓ clean≤1.0 ✓）
- 剩余：写作文日（README/model card/demo）→ 终检 → 发布（窗口 9/6-9/11）
- GPU 累计：本机 ~18h（M0 4.93 + M1 6.1 + M2 6.47 + 评测缓存~1）；云预算 0/6h 未动
```

```
## D17(写作文日) 08-31晚（提前执行）
- README 全面改版：三模型阶梯头版 + 三发现 + 演示 + 权重占位 + rvkit 真实命令复现链
- docs/model_card_m2.md：训练构成/各条件成绩/置信度特性/适用边界/局限/许可
- demo/demo.gif：4 帧（晴/夜/雨/雪）M0|M2 分屏，839KB（make_demo_gif.py，可复现）
- case_study 状态行更新为"全部锁定"
- 剩余：终检（干净环境安装测试+自查四条）→ 简历/面试稿 → Release
```

```
## D18(终检自查) 08-31深夜
- 修复：可靠性图空占位面板的残留虚线（对角参考线只画进真实条件面板；三图像素级验证全白）
- 新增 docs/metrics_glossary.md：全库指标「大白话+术语」双解释词典（含电梯演讲版）
- 终检自查四条全过：
  · pytest 33 passed；TODO/FIXME 零残留
  · 数字一致性审计 122/122：以 ladder.csv 为真值源核对 README/模型卡/case_study/三份体检/
    三份校准（ECE 相对下降逐行重算、T 与 temperature.json 对账）
  · M2 权重重跑复现：clean_day 0.2961 / night 0.2546 与阶梯表逐位一致（3,939 张现推现算）
  · 仓库卫生：全部 md 相对链接/引用路径存在（README 3 个权重占位为发布预留）
- release_v0.1.0/ 本地暂存（不打 tag）：3×best.pt + M0/M1 补模型卡 + ladder.csv/md
  + demo.gif + 发布说明草稿
- docs/interview_prep.md：9 问口语稿 + 数字速查卡 + 英文追问一句话版
- 简历中英文 + PS 段落草稿：本地文档层（隐私考虑不入公开库）
- 按用户指示取消：干净 venv 重装复现测试（原 GATE-1 项）
- 剩余：用户过目调整 → 卡片⑥：tag v0.1.0 + GitHub Release 附件 + README 填下载链接
```
