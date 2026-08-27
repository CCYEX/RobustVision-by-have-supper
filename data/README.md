# data/

这个目录存放**脚本、切分索引与（已 gitignore 的）原始数据**。

## 当前布局（2026-08-28 起）

- `train/  val/  test/`：BDD100K 图像 + `annotations/` 官方格式总表 JSON（train 1.45 GB / val 208 MB）。
  **已在 .gitignore 排除，绝不入库**——体积 7 GB+，且 BDD100K 许可要求不再分发原图。
- `splits/`（待生成）：条件子集与切分清单，**只存文件名不存路径**（保证云会话可移植）。
- `yolo/`（待生成）：convert.py 产出的 YOLO 格式标签，同样 gitignored。

## 已知数据问题（详见 experiments/log.md）

- 标注混入多任务类别（lane / drivable area，poly2d 无 box2d）→ 转换时按「类别白名单 + 含 box2d」过滤；
- 官方类名是 motor / bike；
- train 有 137 张图像无标注记录（0.2%）→ 无属性、不参与条件抽样，转换时跳过；
- 属性词表以实测为准：weather 含 undefined / overcast / partly cloudy 等，条件定义见 data_card。
