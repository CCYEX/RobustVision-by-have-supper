# data/

这个目录只存放**脚本与切分索引**（convert.py、make_splits.py、audit.py、splits/*.txt）。

原始与转换后的数据集（图像、YOLO 标签、损坏副本）一律放在 repo 外（如 `D:\data\bdd100k\`），
由 .gitignore 排除。理由：体积（数 GB～数十 GB）+ BDD100K 许可要求不再分发原图。
