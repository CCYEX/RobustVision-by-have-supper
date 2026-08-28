#!/usr/bin/env bash
# ============================================================
# run_s1.sh —— 9 月 4 日云电脑当天的作战清单（S1 会话）
#
# 这是什么：一个"终端命令清单"（bash 脚本，不是 Python）。
#   在云电脑（Linux）上打开终端，进入仓库根目录后敲：
#       bash experiments/run_s1.sh
#   终端就会从上到下自动逐条执行。
#
# 看懂本文件的 3 条规则：
#   1. 以 # 开头的行是注释，给人看的，机器不执行；
#   2. 命令从上到下逐条执行，一条卡住（比如训练要 1 小时）就停在那等；
#   3. set -e 的作用：任何一步失败立刻停止，防止错上加错、白烧租卡的钱。
# ============================================================
set -e

# ---------- 第0步：拿代码、装环境（约15分钟） ----------
# 把 GitHub 上的项目下载到云电脑（清单/配置/lock 文件都在里面）
git clone https://github.com/CCYEX/RobustVision-by-have-supper.git
cd RobustVision-by-have-supper
# 按 requirements-lock.txt 记录的版本清单装依赖（和本机版本一模一样，避免"我这能跑你那不能跑"）
pip install -r requirements-lock.txt

# ---------- 第1步：拿数据（约20-40分钟） ----------
# TODO(9/4 当天填)：用你本机下载时的同一来源下载 train/val 两个包，解压成：
#   data/train/images/  data/train/annotations/
#   data/val/images/    data/val/annotations/
# 数据到位后，重建 YOLO 标签（约 1 分钟，8 万个 txt）：
python data/convert.py

# ---------- 第2步：试跑 3 轮测速度（约15分钟） ----------
# 目的：正式训练前先知道这台机器多快，据此决定正式跑多少轮（省钱）
# 前提：configs/m0_clear_day.yaml 需在 D8 前于本机写好并 push（手册 D8 物料清单）
yolo train model=yolo11m.pt data=configs/m0_clear_day.yaml \
  epochs=3 imgsz=640 batch=32 amp=True device=0
# ↑ 训练画面会滚出 "x.xx it/s"。换算：每秒处理张数 = it/s × 32
#   ≥250 → 第3步 epochs=40；200~250 → 改 30；<200 → 训练名单减到 8000 张再测

# ---------- 第3步：正式训练 M0（约60-80分钟，会停在这里很久） ----------
# M0 = 只见过"晴天白天"的基线模型（本项目三模型阶梯的第一级）
yolo train model=yolo11m.pt data=configs/m0_clear_day.yaml \
  epochs=40 patience=15 imgsz=640 batch=32 amp=True seed=42 \
  workers=12 device=0 project=runs name=m0

# ---------- 第4步：等待期的 CPU 活（重要：训练开始后，另开一个终端窗口执行！） ----------
# 训练占着显卡，CPU 空着——正好拿来生成 M1 的"做坏图"、备好 M2 的真实图
# TODO(D8 前)：make_aug_copies.py / predict_cache.py 写好并 push 后，取消下面两行注释
# python -m rvkit.harness.make_aug_copies --list data/splits/train_clear_all.txt --ratio 0.3 --out train_m1/
# python <按 m2_real_adverse.txt 把 4000 张真实夜景雨天图+标签拷到 train_m2_seed/ 的命令>

# ---------- 第5步：训完立刻考试 + 存答案（约30分钟） ----------
# 用我们自己写的评测引擎给 M0 打分，并把它的全部答案存成表格（9/9 校准免再租卡）
# TODO(D8 前)：评测与缓存入口写好后，取消下面两行注释
# python -m rvkit.harness.runner --model runs/m0/weights/best.pt --full --out results/m0_full.csv
# python predict_cache.py runs/m0/weights/best.pt data/splits/test.txt cache/m0.parquet

# ---------- 第6步：回传 4 件套，然后关机 ----------
# 用云平台的"文件下载"功能拿回这 4 样小文件（共约 100MB，不传任何图片）：
#   runs/m0/weights/best.pt   ← 模型本体
#   results/m0_full.csv       ← 考试成绩
#   cache/m0.parquet          ← 模型的全部答案（9/9 校准用）
#   runs/m0/results.csv       ← 每轮成绩曲线
echo "S1 主干完成！记得：回传 4 件套 → git push → 关机。"
