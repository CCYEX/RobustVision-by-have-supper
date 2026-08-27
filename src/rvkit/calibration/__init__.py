"""Module ②: detection calibration — global temperature scaling, per-condition ECE.

D14–D16 开发：IoU 匹配二值化 → logit 空间拟合全局温度 T → ECE/MCE/Brier/可靠性图。
输入为会话落盘的原始检测缓存（parquet），开发不需要 GPU。
"""
