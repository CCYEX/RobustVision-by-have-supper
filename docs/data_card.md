# BDD100K Data Card

<!-- D3（08-29）audit.py 跑完后填写 -->

## 来源

- BDD100K detection subset（CVPR 2020），注册获取：http://bdd-data.berkeley.edu/
- 使用范围：10 类检测 + weather/timeofday/scene 属性；仅研究用途，本 repo 不分发原图

## 切分

| 用途 | 来源 | 规模 | 说明 |
| --- | --- | --- | --- |
| train | 70K train 中 clear∩daytime | 20K（seed 42 抽样） | 模型从未见过夜间/雨天 |
| train-val（监控） | train 内留出 | 1K | 降低每 epoch 开销 |
| calibration | 10K val 分层切出 | 3K | 仅用于拟合温度参数 |
| test | 10K val 剩余 | 7K | 按 weather × timeofday 分层 |

## 条件分布（待 audit 实测）

TODO：weather × timeofday 交叉表、各类别框数、框尺寸分布（小目标占比）。
