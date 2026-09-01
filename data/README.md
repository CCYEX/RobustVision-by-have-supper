# data/

这个目录存放**脚本、切分索引与（已 gitignore 的）原始数据/派生数据**。脚本和清单入库，大文件不入库。

## 当前布局（2026-08-31 更新）

| 目录/文件 | 内容 | 是否入库 |
| --- | --- | --- |
| `convert.py / make_splits.py / audit.py / make_train_lists.py` | 数据四件套：标注转换 / 分层切分 / 审计 / 训练名单 | ✅ |
| `train/ val/ test/` | BDD100K 图像 + 官方标注总表（train 1.45GB / val 208MB）+ `<split>/labels/`（10 类 YOLO 标签）；test 为官方无标注图（项目不用） | ❌ gitignored |
| `train8/ val8/` | 8 类 COCO 对齐口径的标签（演示现成模型用；图像硬链接复用） | ❌ |
| `corrupt/ corrupt8/` | 7 种合成损坏 × 1,235 张 × 双口径（损坏底图 = clean_day test 全集，与干净参照同图配对） | ❌ |
| `train_m1/ train_m2/` | M1 / M2 训练盘（原图 + aug_ 前缀损坏副本 + 真实夜雨），由 make_aug_copies.py 生成 | ❌ |
| `gen/` | 评测引擎自动生成的条件 yaml + 路径清单工作区（双口径） | ❌ |
| `splits/` | 全部名单（calib/test/val_all/corrupt_base/m2_real_adverse/trainval_1k/train_clear_all + conditions/ 六自然条件 + cloud_eval/ 十三条件全量 + mini/ 300 张开发版），**只存文件名不存路径** | ✅ |
| `splits_train/` | 带绝对路径的训练/监控名单（喂训练器，换机器需重生成） | ❌ |

## 为什么部分目录不入库（2026-08-31 补）

两类原因，完全不同：

**第一类：原始数据（`train/ val/ test/`）——不是能不能生成的问题，是不能转分发**
- BDD100K 是伯克利的注册制许可数据集，放进公开仓库等于替官方转分发，违反其许可（根 README 与模型卡的"不再分发任何训练原图"即此承诺）；
- 体积也过不了关：GitHub 单文件上限 100MB，train 一套就 1.45GB；
- 使用者需自行去官方注册下载后放入本目录（见根 README 复现链第 1 步）。

**第二类：派生数据（labels / train8·val8 / corrupt·corrupt8 / train_m1·m2 / gen / splits_train）——因为能由「已入库的脚本 + 名单」确定性重建**
- `<split>/labels/`、`train8/ val8/`：`convert.py` 确定性转换（图像为硬链接，零额外磁盘）；
- `corrupt/ corrupt8/`：`generate_corrupt.py`，随机性全部 md5(图名+坏法) 播种 → 重建**逐字节一致**（已做磁盘级复核）；
- `train_m1/ train_m2/`：`make_aug_copies.py` 确定性组盘；
- `gen/`：评测引擎运行时自动生成的临时工作区；
- `splits_train/`：绝对路径名单，换机器本来就必须重生成（`splits/` 里存的是不含路径的纯文件名，跨机器通用）。

一句话：**原始数据不入库是版权与体积所迫（只能重新下载）；派生数据不入库是因为重建的确定性已验证（一条命令重建，不入库不损失任何可复现性）**。

## 已知数据问题（详见 experiments/log.md 与 docs/data_card.md）

- 标注混入多任务类别（lane / drivable area，poly2d 无 box2d）→ 转换按「类别白名单 + 含 box2d」过滤；
- 官方类名是 motor / bike（不是 motorcycle / bicycle）；
- train 有 137 张图像无标注记录（0.2%）→ 转换时跳过；
- 属性词表以实测为准（weather 含 undefined / overcast 等），条件定义冻结在 data_card；
- **clear∩daytime 远比直觉少**：train 12,454 / val 1,764（clear 大部分是夜景）→ 训练池全量使用、损坏底图同图配对。
