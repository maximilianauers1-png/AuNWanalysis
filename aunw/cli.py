"""Command-line interface for headless use (e.g. training on a GPU server).

    python -m aunw.cli generate --out data/synthetic --n-train 1500
    python -m aunw.cli train    --data data/synthetic --out models/my_model --epochs 60
    python -m aunw.cli finetune --base models/my_model/model.keras --real tiles.npz --out models/ft
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


def _run_monitored(out: Path, fn, **kwargs) -> None:
    """Run train()/finetune() with a progress file; record failures there too."""
    import traceback
    from .training import TrainingMonitor
    mon = TrainingMonitor(out)
    try:
        fn(monitor=mon, **kwargs)
        print(mon.message)
    except Exception as exc:
        mon.update(force=True, status="error", message=f"{exc}\n{traceback.format_exc()}")
        raise


def cmd_train(a) -> None:
    from .config import ModelConfig, TrainConfig
    from .training import train
    mcfg = ModelConfig(base_filters=a.base_filters, depth=a.depth, dropout=a.dropout)
    tcfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, learning_rate=a.lr,
                       weight_decay=a.weight_decay, early_stop_patience=a.patience,
                       augment=not a.no_augment, mixed_precision=a.mixed_precision)
    _run_monitored(Path(a.out), train, model_dir=Path(a.out), data_dir=Path(a.data),
                   model_cfg=mcfg, train_cfg=tcfg, init_from=Path(a.init) if a.init else None)


def cmd_finetune(a) -> None:
    import numpy as np
    from .config import TrainConfig
    from .dataset import KEYS, load_split
    from .training import finetune
    with np.load(a.real) as f:
        real_train = {k: f[f"train_{k}"] for k in KEYS}
        real_val = {k: f[f"val_{k}"] for k in KEYS}
    tcfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, learning_rate=a.lr,
                       early_stop_patience=a.patience)
    _run_monitored(Path(a.out), finetune, base_model=Path(a.base), model_dir=Path(a.out),
                   real_train=real_train, real_val=real_val,
                   synthetic=load_split(Path(a.synthetic)), train_cfg=tcfg,
                   freeze_encoder=not a.no_freeze, real_fraction=a.real_fraction)


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
    t.add_argument("--patience", type=int, default=10, help="early-stopping patience")
    t.add_argument("--no-augment", action="store_true")
    t.add_argument("--mixed-precision", action="store_true")
    t.set_defaults(fn=cmd_train)

    ft = sub.add_parser("finetune", help="adapt a model to annotated real images")
    ft.add_argument("--base", required=True, help="model file (.keras) to start from")
    ft.add_argument("--out", required=True)
    ft.add_argument("--real", required=True,
                    help="npz with train_/val_ image, mask, centerline, angle tiles")
    ft.add_argument("--synthetic", default="data/synthetic/train.npz")
    ft.add_argument("--epochs", type=int, default=20)
    ft.add_argument("--batch-size", type=int, default=8)
    ft.add_argument("--lr", type=float, default=1e-4)
    ft.add_argument("--patience", type=int, default=6)
    ft.add_argument("--real-fraction", type=float, default=0.5)
    ft.add_argument("--no-freeze", action="store_true", help="also train the encoder")
    ft.set_defaults(fn=cmd_finetune)

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
