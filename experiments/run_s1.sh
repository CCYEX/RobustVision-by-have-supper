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
# 装依赖。注意：本机 lock 里的 torch 是 Windows 专用包（+cu130），Linux 云端装不上；
# 云平台镜像自带 Linux 版 torch，这里跳过 torch/torchvision、其余照 lock 安装
grep -vE "^(torch|torchvision)==" requirements-lock.txt > /tmp/req_cloud.txt
pip install -r /tmp/req_cloud.txt

# ---------- 第1步：拿数据（约20-40分钟） ----------
# TODO(9/4 当天填)：用你本机下载时的同一来源下载 train/val 两个包，解压成：
#   data/train/images/  data/train/annotations/
#   data/val/images/    data/val/annotations/
# 数据到位后做两件事：
python data/convert.py                                     # ①重建 YOLO 标签（1 分钟，8 万个 txt）
python -m rvkit.harness.datasets --data-root /root/autodl-tmp/data
# ↑ ②生成带路径的训练名单 splits_train/*_paths.txt（路径要与 configs/m0_local.yaml（云端使用时把 path 改成云端数据根）
#   的 path 一致——云电脑数据不在 /root/autodl-tmp/data 就两边一起改）

# ---------- 第2步：试跑 3 轮测速度（约15分钟） ----------
# 目的：正式训练前先知道这台机器多快，据此决定正式跑多少轮（省钱）
# 前提：configs/m0_local.yaml 已入库（云端使用时把 path 改成云端数据根）
yolo train model=yolo11m.pt data=configs/m0_local.yaml（云端使用时把 path 改成云端数据根） \
  epochs=3 imgsz=640 batch=32 amp=True device=0
# ↑ 训练画面会滚出 "x.xx it/s"。换算：每秒处理张数 = it/s × 32
#   ≥250 → 第3步 epochs=40；200~250 → 改 30；<200 → 训练名单减到 8000 张再测

# ---------- 第3步：正式训练 M0（约60-80分钟，会停在这里很久） ----------
# M0 = 只见过"晴天白天"的基线模型（本项目三模型阶梯的第一级）
yolo train model=yolo11m.pt data=configs/m0_local.yaml（云端使用时把 path 改成云端数据根） \
  epochs=40 patience=15 imgsz=640 batch=32 amp=True seed=42 \
  workers=12 device=0 project=runs name=m0

# ---------- 第4步：等待期的 CPU 活（重要：训练开始后，另开一个终端窗口执行！） ----------
# 训练占着显卡，CPU 空着——正好拿来生成 M1 的"做坏图"、备好 M2 的真实图
python -m rvkit.harness.make_aug_copies --list data/splits/train_clear_all.txt --ratio 0.3 --out train_m1/
# ↑ 抽 30%（约 3.4K 张），每张随机做一种坏 → train_m1/{images,labels}，9/6 训 M1 用
python -m rvkit.harness.make_aug_copies --list data/splits/m2_real_adverse.txt --ratio 1.0 --mode copy --out train_m2_seed/
# ↑ 4000 张真夜景/雨天原图+标签原样拷到 train_m2_seed/，9/8 拼 M2 练习盘用
python -m rvkit.harness.generate_corrupt
# ↑ 评测用的坏图库（val 前 300 张 × 7 种，10 类+8 类双口径），第 5 步考试要用

# ---------- 第5步：训完立刻考试 + 存答案（约30分钟） ----------
# 用我们自己写的评测引擎给 M0 打分（全部 13 个条件、10 类口径），并把它的
# 全部答案存成 parquet 表格（9/9 校准免再租卡）
# 先组"全量考试"的名单目录：6 个自然条件（test 全量）+ 7 种坏（corrupt_base 全量
# 1,235 张，与干净参照同图配对）；GPU 上全量也就几分钟，直接出正式数字
mkdir -p data/splits/cloud_eval
for f in clean_day night rain snow dawn_dusk fog; do
  cp data/splits/conditions/$f.txt data/splits/cloud_eval/
done
for c in low_light motion_blur gauss_noise fog rain downscale jpeg; do
  cp data/splits/corrupt_base.txt data/splits/cloud_eval/${c}_s2.txt
done
rvkit checkup --model runs/m0/weights/best.pt --splits data/splits/cloud_eval/ \
  --mode labels10 --out results/m0_full.md
# ↑ 出 results/m0_full.{md,csv} + m0_full_perclass.csv + 热力图（10 类口径）
python -m rvkit.harness.predict_cache runs/m0/weights/best.pt data/splits/val_all.txt cache/m0.parquet
# ↑ calib+test 共 1 万张的答案表（9/9 校准的原料）

# ---------- 第6步：回传 4 件套，然后关机 ----------
# 用云平台的"文件下载"功能拿回这 4 样小文件（共约 100MB，不传任何图片）：
#   runs/m0/weights/best.pt   ← 模型本体
#   results/m0_full.csv       ← 考试成绩
#   cache/m0.parquet          ← 模型的全部答案（9/9 校准用）
#   runs/m0/results.csv       ← 每轮成绩曲线
echo "S1 主干完成！记得：回传 4 件套 → git push → 关机。"
