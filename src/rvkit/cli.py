"""rvkit CLI. D7（09-02）在此实现 checkup 子命令，当前为可安装的占位入口。"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rvkit",
        description="Robustness checkup toolkit for YOLO-family object detectors",
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args()

    if args.version:
        from rvkit import __version__

        print(__version__)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
