"""Command-line interface for headless use (e.g. training on a GPU server).

    python -m aunw.cli generate --out data/synthetic --n-train 1500
    python -m aunw.cli train    --data data/synthetic --out models/my_model --epochs 60
    python -m aunw.cli evaluate --model models/my_model --data data/synthetic
    python -m aunw.cli analyze  --model models/my_model --nm-per-px 2.5 images/*.tif --out results/run1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _progress(name: str, frac: float) -> None:
    sys.stdout.write(f"\r  {name:5s} {frac * 100:5.1f} %")
    sys.stdout.flush()
    if frac >= 1:
        print()


def cmd_generate(a) -> None:
    from .config import SyntheticConfig
    from .dataset import generate_dataset
    t = time.time()
    generate_dataset(Path(a.out), SyntheticConfig(), a.n_train, a.n_val, a.n_test,
                     a.test_size, a.seed, a.workers, _progress)
    print(f"done in {time.time() - t:.0f} s -> {a.out}")


def cmd_train(a) -> None:
    from .config import ModelConfig, TrainConfig
    from .training import TrainingMonitor, train
    mcfg = ModelConfig(base_filters=a.base_filters, depth=a.depth, dropout=a.dropout)
    tcfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, learning_rate=a.lr,
                       weight_decay=a.weight_decay, mixed_precision=a.mixed_precision)
    mon = TrainingMonitor()

    import threading
    def report():
        last = 0
        while mon.status in ("idle", "running"):
            time.sleep(5)
            s = mon.snapshot()
            if len(s["history"]) > last:
                last = len(s["history"])
                h = s["history"][-1]
                print(f"epoch {h['epoch']:3d}  loss {h['loss']:.4f}  val_loss {h['val_loss']:.4f}  "
                      f"mask dice {h.get('val_mask_dice_coef', 0):.3f}  "
                      f"cl dice {h.get('val_centerline_dice_coef', 0):.3f}  "
                      f"angle err {h.get('val_orientation_angle_error_deg', 0):.1f} deg  "
                      f"lr {h['lr']:.1e}  {h['seconds']:.0f}s", flush=True)
    threading.Thread(target=report, daemon=True).start()
    train(Path(a.out), Path(a.data), mcfg, tcfg, monitor=mon,
          init_from=Path(a.init) if a.init else None)
    time.sleep(6)
    print(mon.message)


def cmd_evaluate(a) -> None:
    from .evaluation import evaluate_synthetic
    from .model import load_model
    from .training import model_file
    model = load_model(model_file(Path(a.model)))
    res = evaluate_synthetic(model, Path(a.data), n_images=a.n_images)
    print(json.dumps(res["summary"], indent=2))


def cmd_analyze(a) -> None:
    from .config import AnalysisConfig
    from .export import export_bundle
    from .inference import analyze_image
    from .io import crop_databar, detect_databar, load_image
    from .model import load_model
    from .training import model_file
    model = load_model(model_file(Path(a.model)))
    results = []
    for f in a.images:
        sem = load_image(f)
        sem = crop_databar(sem, detect_databar(sem.pixels))
        if a.nm_per_px:
            sem.nm_per_px, sem.calibration_source = a.nm_per_px, "manual"
        r = analyze_image(model, sem, AnalysisConfig())
        print(f"{sem.name}: {len(r.wires)} wires (scale {r.scale:.2f})")
        results.append(r)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.zip").write_bytes(export_bundle(results, AnalysisConfig()))
    print(f"exported -> {out / 'results.zip'}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="aunw", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="create a synthetic dataset")
    g.add_argument("--out", default="data/synthetic")
    g.add_argument("--n-train", type=int, default=1500)
    g.add_argument("--n-val", type=int, default=250)
    g.add_argument("--n-test", type=int, default=12)
    g.add_argument("--test-size", type=int, default=768)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--workers", type=int, default=4)
    g.set_defaults(fn=cmd_generate)

    t = sub.add_parser("train", help="train a network on a synthetic dataset")
    t.add_argument("--data", default="data/synthetic")
    t.add_argument("--out", default="models/aunw_unet")
    t.add_argument("--init", help="continue from an existing .keras model")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--base-filters", type=int, default=24)
    t.add_argument("--depth", type=int, default=4)
    t.add_argument("--dropout", type=float, default=0.15)
    t.add_argument("--mixed-precision", action="store_true")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate", help="measurement accuracy on synthetic test images")
    e.add_argument("--model", required=True)
    e.add_argument("--data", default="data/synthetic")
    e.add_argument("--n-images", type=int, default=12)
    e.set_defaults(fn=cmd_evaluate)

    an = sub.add_parser("analyze", help="analyse SEM images")
    an.add_argument("--model", required=True)
    an.add_argument("--nm-per-px", type=float)
    an.add_argument("--out", default="results/cli")
    an.add_argument("images", nargs="+")
    an.set_defaults(fn=cmd_analyze)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
