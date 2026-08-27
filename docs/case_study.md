# Case Study: Object Detection Robustness under Real-World Distribution Shifts

<!-- D10 骨架、D17 成文。协议与严谨性检查清单见执行计划书 4/P2 节 -->

## 研究问题

- RQ1：各自然/合成偏移条件让检测器损失多少 mAP？失败集中在哪些类别？
- RQ2：仅用 clear+daytime 训练时，合成增强能恢复多少偏移条件性能、clean 损失多少？
- RQ3：偏移条件下置信度还剩多少可信度？全局温度缩放能修复多少（分条件 ECE）？

## 设定

训练 YOLO11m 仅用 20K clear-day 图像；评测 = 自然条件子集（night / dawn-dusk / rain / …）
+ 7 种合成损坏 × 3 档强度。分层切分、calibration/test 分离、seed 42、mAP50-95 为主指标。

## 结果

TODO（数字锁定于 GATE-2，09-09）

## 局限

单一数据集；单一检测器家族；自然雾/雪样本少（相关结论主要依赖合成损坏）；v0.1.0 仅全局温度缩放。
