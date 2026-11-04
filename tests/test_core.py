"""Unit tests for the scientific core.  Run with:  pytest -q"""
import io

import keras
import numpy as np
import pytest
import tensorflow as tf
import tifffile

from aunw.config import AnalysisConfig, ModelConfig, SyntheticConfig
from aunw.dataset import _augment, _to_targets
from aunw.inference import predict_tiled
from aunw.io import detect_databar, load_image, normalize
from aunw.measure import analyze_maps, crossing_number, polyline_length
from aunw.model import build_unet, dice_bce_loss, masked_orientation_loss, receptive_field
from aunw.stats import compare_samples, describe, miles_lantuejoul_weights
from aunw.synthetic import Wire, render_sample


# --------------------------------------------------------------------------- helpers
def render_maps(wires: list[Wire], size: int = 256) -> dict:
    """Perfect network output for given wires (used to test the measurement alone)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    mask = np.zeros((size, size), np.float32)
    cl = np.zeros((size, size), np.float32)
    c2 = np.zeros((size, size), np.float32)
    s2 = np.zeros((size, size), np.float32)
    count = np.zeros((size, size))
    for w in wires:
        d, th = w.distance_field(xx, yy)
        inside = d <= w.width / 2
        mask = np.maximum(mask, np.clip(w.width / 2 - d + 0.5, 0, 1))
        cl[d <= 1.0] = 1
        count += inside
        c2 += inside * np.cos(2 * th)
        s2 += inside * np.sin(2 * th)
    single = count == 1
    return {"mask": mask, "centerline": cl * (mask > 0.5),
            "orientation": np.stack([c2 * single, s2 * single], -1)}


def straight(x0, y0, angle_deg, length, width) -> Wire:
    lc = length - width
    return Wire(np.array([x0, y0], float), np.radians(angle_deg), 0.0, lc, width)


# --------------------------------------------------------------------------- synthetic
def test_render_sample_shapes_and_labels():
    s = render_sample(np.random.default_rng(0), SyntheticConfig())
    assert s["image"].shape == (256, 256) and s["image"].dtype == np.uint8
    assert set(np.unique(s["mask"])) <= {0, 1}
    assert (s["centerline"] <= s["mask"]).all(), "centre-line must lie inside the mask"
    assert ((s["angle"] < 180) | (s["angle"] == 255)).all()


def test_arc_distance_field_is_exact():
    w = Wire(np.array([50.0, 50.0]), 0.3, 1 / 80, 100.0, 8.0)       # curved wire
    s = np.linspace(0, w.lc, 50)
    pts = w.point(s)
    d, _ = w.distance_field(pts[:, 0], pts[:, 1])
    assert np.allclose(d, 0, atol=1e-9), "points on the arc have distance 0"
    # arc length by fine polyline equals the analytic centre-line length
    fine = w.point(np.linspace(0, w.lc, 5000))
    assert np.isclose(np.linalg.norm(np.diff(fine, axis=0), axis=1).sum(), w.lc, rtol=1e-6)


# --------------------------------------------------------------------------- augmentation
@pytest.mark.parametrize("seed", range(6))
def test_augmentation_keeps_orientation_consistent(seed):
    """After random flips/rotations the label angle must match the wire geometry."""
    tf.random.set_seed(seed)
    w = straight(40, 60, 30, 180, 8)
    maps = render_maps([w], 256)
    ang = np.where(maps["mask"] > 0.5, 30, 255).astype(np.uint8)
    x, y = _to_targets(np.zeros((256, 256), np.uint8), (maps["mask"] > 0.5).astype(np.uint8),
                       maps["centerline"].astype(np.uint8), ang)
    x, y = _augment(x, y)
    m = y["mask"].numpy()[..., 0] > 0.5
    yy, xx = np.nonzero(m)
    cov = np.cov(np.stack([xx, yy]))
    evec = np.linalg.eigh(cov)[1][:, -1]
    theta_geom = np.arctan2(evec[1], evec[0])
    o = y["orientation"].numpy()[m]
    theta_lbl = 0.5 * np.arctan2(o[:, 1].mean(), o[:, 0].mean())
    diff = abs(theta_geom - theta_lbl) % np.pi
    assert min(diff, np.pi - diff) < np.radians(2)


# --------------------------------------------------------------------------- measurement
def test_crossing_number_classifies_x_junction():
    sk = np.zeros((11, 11), bool)
    sk[5, :] = True
    sk[:, 5] = True
    cn, _ = crossing_number(sk)
    assert cn[5, 5] == 4 and cn[5, 0] == 1 and cn[5, 2] == 2


def test_polyline_length_removes_staircase_bias():
    t = np.linspace(0, 1, 2000)
    pts = np.unique(np.round(np.stack([t * 60 + 3, t * 200 + 3], 1)).astype(int), axis=0)
    true = np.hypot(60, 200)
    assert abs(polyline_length(pts) - true) / true < 0.01


@pytest.mark.parametrize("angle", [0, 17, 45, 73])
def test_single_wire_length_and_width(angle):
    w = straight(60, 60, angle, 120, 9)
    wires, _ = analyze_maps(render_maps([w]), AnalysisConfig())
    assert len(wires) == 1
    assert wires[0]["complete"]
    assert abs(wires[0]["length_px"] - 120) / 120 < 0.03
    assert abs(wires[0]["width_px"] - 9) / 9 < 0.05


def test_crossing_wires_are_separated():
    a = straight(30, 128, 0, 200, 8)                 # horizontal
    b = straight(128, 30, 80, 200, 8)                # nearly vertical, crossing a
    wires, info = analyze_maps(render_maps([a, b]), AnalysisConfig())
    assert len(wires) == 2
    assert all(abs(x["length_px"] - 200) / 200 < 0.05 for x in wires)
    assert info["crossing_resolution"] == 1.0


def test_wire_cut_by_border_is_flagged():
    w = straight(-50, 100, 0, 200, 8)
    wires, _ = analyze_maps(render_maps([w]), AnalysisConfig())
    assert wires and wires[0]["touches_border"] and not wires[0]["complete"]


def test_particle_is_rejected():
    yy, xx = np.mgrid[0:128, 0:128]
    blob = (np.hypot(xx - 64, yy - 64) < 6).astype(np.float32)
    maps = {"mask": blob, "centerline": blob}
    wires, _ = analyze_maps(maps, AnalysisConfig())
    assert wires == []


# --------------------------------------------------------------------------- model / inference
def test_model_shapes_losses_and_serialisation(tmp_path):
    cfg = ModelConfig(base_filters=8, depth=3)
    m = build_unet(cfg)
    out = m(np.zeros((2, 64, 96, 1), np.float32))
    assert out["mask"].shape == (2, 64, 96, 1) and out["orientation"].shape == (2, 64, 96, 2)
    y = tf.ones((2, 64, 96, 1))
    assert float(dice_bce_loss(y, out["mask"])) > 0
    yo = tf.concat([tf.ones((2, 64, 96, 2)), tf.zeros((2, 64, 96, 1))], -1)
    assert float(masked_orientation_loss(yo, out["orientation"])) == 0.0   # weight 0 -> no loss
    m.save(tmp_path / "m.keras")
    m2 = keras.models.load_model(tmp_path / "m.keras", compile=False)
    assert np.allclose(m2(np.ones((1, 64, 64, 1)))["mask"], m(np.ones((1, 64, 64, 1)))["mask"], atol=1e-6)
    assert receptive_field(ModelConfig(depth=4)) == 200


def _identity_model():
    inp = keras.Input((None, None, 1))
    zeros = keras.layers.Conv2D(2, 1, kernel_initializer="zeros", bias_initializer="zeros")(inp)
    return keras.Model(inp, {"mask": keras.layers.Identity()(inp),
                             "centerline": keras.layers.Identity()(inp), "orientation": zeros})


@pytest.mark.parametrize("shape", [(300, 517), (100, 90), (256, 256)])
def test_tiled_prediction_is_seamless(shape):
    img = np.random.default_rng(0).random(shape).astype(np.float32)
    maps = predict_tiled(_identity_model(), img, tile=128, overlap=32, tta=True)
    assert maps["mask"].shape == shape
    assert np.abs(maps["mask"] - img).max() < 1e-5


# --------------------------------------------------------------------------- statistics
def test_describe_recovers_lognormal_parameters():
    x = np.random.default_rng(1).lognormal(np.log(500), 0.4, 5000)
    d = describe(x)
    assert abs(d["lognorm_mu"] - np.log(500)) < 0.02 and abs(d["lognorm_sigma"] - 0.4) < 0.02
    assert d["best_model"] == "log-normal"
    assert d["ci95_low"] < np.exp(np.log(500) + 0.4**2 / 2) < d["ci95_high"]


def test_miles_lantuejoul_weight():
    import pandas as pd
    df = pd.DataFrame({"extent_x_px": [0, 500], "extent_y_px": [0, 0]})
    w = miles_lantuejoul_weights(df, 1000, 1000)
    assert np.allclose(w, [1.0, 2.0])


def test_compare_samples_same_population():
    rng = np.random.default_rng(2)
    r = compare_samples(rng.normal(10, 1, 400), rng.normal(10, 1, 400))
    assert r["KS_p"] > 0.01


# --------------------------------------------------------------------------- io
def test_fei_metadata_and_16bit():
    buf = io.BytesIO()
    img = (np.random.default_rng(0).random((64, 80)) * 4000).astype(np.uint16)
    tifffile.imwrite(buf, img, extratags=[(34682, "s", 0, "[Scan]\nPixelWidth=3.2e-09\n", True)])
    sem = load_image(buf.getvalue(), "x.tif")
    assert sem.pixels.dtype == np.float32 and sem.pixels.max() <= 1.0
    assert sem.nm_per_px == pytest.approx(3.2)


def test_databar_detection():
    img = normalize(np.random.default_rng(0).random((400, 300)))
    img = np.vstack([img, np.zeros((50, 300), np.float32)])
    assert detect_databar(img) == 50


# --------------------------------------------------------------------------- training (end to end)
def test_training_run_saves_model(tmp_path):
    from aunw.config import TrainConfig
    from aunw.dataset import generate_dataset
    from aunw.training import TrainingMonitor, train
    generate_dataset(tmp_path / "data", SyntheticConfig(), n_train=16, n_val=8, n_test=0, workers=1)
    mon = TrainingMonitor()
    train(tmp_path / "model", tmp_path / "data", ModelConfig(base_filters=4, depth=2),
          TrainConfig(epochs=1, batch_size=4), monitor=mon)
    assert (tmp_path / "model" / "model.keras").exists()
    assert (tmp_path / "model" / "config.json").exists()
    assert mon.status == "finished" and len(mon.history) == 1 and mon.preview
