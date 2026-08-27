# RobustVision

**A more reliable driving object detector under adverse conditions** — a three-model ladder (M0 clear-day baseline → M1 synthetic-augmentation robust → M2 flagship trained with mixed real night/rain data), quantified on BDD100K across 5 natural conditions + 7 synthetic corruptions, with released weights and model cards. Includes **rvkit**, the detector-agnostic evaluation engine used to produce every number in this repo.

## Quickstart (evaluation engine)

<!-- GATE-1（09-03）用当天真实跑通的输出替换本节 -->

```python
import rvkit

harness = rvkit.RobustnessHarness(model="best.pt")     # any ultralytics .pt
report = harness.evaluate(data="bdd_val.yaml")         # natural conditions + synthetic corruptions
report.save("robustness_report.md")
```

CLI equivalent:

```bash
rvkit checkup --model best.pt --data bdd_val.yaml --out report.md
```

## Status

Work in progress (target: v0.1.0, 2026-09-15). Modules:

- [ ] ① Robustness Harness — condition × corruption degradation report
- [ ] ② Calibration Wrapper — temperature scaling + per-condition ECE
- [ ] ③ Robust Recipes — validated augmentation configs

## Non-Goals

No image-restoration frontends, no test-time online adaptation, no TensorRT export chain, no new architectures. See docs/case_study.md for the evidence behind what *is* included.

## License

MIT
