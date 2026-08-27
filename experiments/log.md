# 执行日志

```
## D1 08-27
- 做了：BDD100K 官方注册完成；PyPI 名称待查；repo 骨架已建（rvkit/，待 push GitHub）
- 数据：官方直连慢 → 用户自行从镜像下载两个包到 D:\Coding\RobustVision\data\
  · val.zip (556MB)：val/images 10,000 张 jpg + val/annotations/bdd100k_labels_images_val.json（官方格式总表）
  · bdd100k_labels.zip (147MB)：bdd100k/labels/100k/{train,val}/ 逐图 JSON（新版 2020 格式，train 70K + val 10K）
- 验证结果（全过）：
  · val 图像 10,000 张 ✓；总 JSON 10,000 条 ✓；JSON name 与图像文件名交集 10,000/10,000 ✓
  · 图像级 attributes（weather/scene/timeofday）两种格式都在 ✓
- 格式坑（convert.py 必须处理）：
  ① 两个包都是"多任务合并"标注：混有 lane / drivable area / lane/xxx / area/xxx 等
     非检测类别 → 过滤规则：category ∈ 10 类白名单 且 标注含 box2d
  ② 官方类名是 motor / bike（不是 motorcycle / bicycle）→ 类别表改为：
     ["person","rider","car","truck","bus","train","motor","bike","traffic light","traffic sign"]，丢弃 other
  ③ train 逐图 JSON 结构：{name, attributes, frames:[{objects:[{category,box2d,...}], timestamp}]}
     → 标注在 frames[0].objects；val 总 JSON 结构：{name, attributes, labels:[...]}
     → convert.py 需两条解析路径（val 总表 / train 逐图），或先把 train 逐图合并成同构总表
- 明日第一件事：make_splits.py（val 总表分层切 3K/7K）+ 条件分布审计
- GPU 累计：0 h / 6 h
```
