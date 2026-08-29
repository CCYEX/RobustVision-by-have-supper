"""rvkit CLI. D7 接入 checkup 子命令：给模型 + 名单，一条命令出完整体检报告。

用法（计划书 D7 定稿的四参数，别加）：
    rvkit checkup --model yolo11s.pt --splits data/splits/mini/ --mode labels8 \
        --out results/mini_report.md
生成四样东西：报告 md（含热力图）、主表 csv、每类分数 csv、热力图 png。

`rvkit --version` 保留：D8 的「假装外人」测试要靠它验证安装成功。
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rvkit",
        description="Robustness checkup toolkit for YOLO-family object detectors",
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    sub = parser.add_subparsers(dest="command")

    ck = sub.add_parser("checkup", help="自动算分 + 生成成绩报告（md + 热力图）")
    ck.add_argument("--model", required=True, help="模型权重，如 yolo11s.pt")
    ck.add_argument("--splits", required=True, help="条件名单目录，如 data/splits/mini/")
    ck.add_argument("--mode", default="labels8",
                    help="labels8=COCO 对齐 8 类（现成模型）；labels10=原始 10 类（自家模型）")
    ck.add_argument("--out", required=True, help="报告输出路径，如 results/mini_report.md")
    args = parser.parse_args()

    if args.version:
        from rvkit import __version__

        print(__version__)
    elif args.command == "checkup":
        _checkup(args)
    else:
        parser.print_help()


def _checkup(args) -> None:
    """checkup 全流程：runner 算分 → report 出报告。导入放在函数内，--version 保持轻快。"""
    from rvkit.harness import report, runner

    out = Path(args.out)
    df, perclass_df = runner.run_checkup(
        args.model, args.splits,
        mode=args.mode,
        out_stem=out.with_suffix(""),   # 主表 csv 与报告同名：mini_report.csv
        n_mini=0,                       # --splits 给的就是现成名单（mini），不再抽迷你
    )
    report.render(df, perclass_df, model=args.model, mode=args.mode, out=out)
    # 注：runner 会先写一版纯表格 md，report.render 随后用完整报告覆盖同名文件
    # ——最终 mini_report.md 里既有主表又有三行总结和热力图。
    print(f"报告已生成：{out}")


if __name__ == "__main__":
    main()
