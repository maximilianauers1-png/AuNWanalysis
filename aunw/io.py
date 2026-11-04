"""Reading SEM images (TIFF / JPEG / PNG) and their calibration.

Three problems are solved here:

1. *Decoding*: SEM TIFFs are often 16-bit or RGB; JPEGs are 8-bit RGB. Everything
   is converted to a single-channel float32 image.
2. *Normalisation*: intensities are mapped to [0, 1] with robust percentiles so
   that the network sees the same dynamic range for synthetic and real images.
3. *Calibration*: FEI/Thermo Fisher and Zeiss microscopes write the pixel size
   into private TIFF tags; we parse it to get nm/pixel automatically.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

SUPPORTED = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp")

# Private TIFF tags used by the two most common SEM vendors.
_FEI_TAG = 34682     # INI-style text, "[Scan] PixelWidth=1.2e-09" (metres)
_ZEISS_TAG = 34118   # text block, "Image Pixel Size = 2.345 nm"
_UNIT_TO_NM = {"pm": 1e-3, "nm": 1.0, "µm": 1e3, "um": 1e3, "mm": 1e6, "m": 1e9}


@dataclass
class SEMImage:
    """A loaded micrograph plus everything needed to measure in physical units."""

    name: str
    pixels: np.ndarray            # float32, 2-D, normalised to [0, 1]
    nm_per_px: float | None       # None = uncalibrated (results stay in px)
    calibration_source: str       # "metadata", "manual", or "none"
    databar_rows: int = 0         # rows cropped from the bottom (info bar)

    @property
    def shape(self) -> tuple[int, int]:
        return self.pixels.shape


# --------------------------------------------------------------------------- decoding
def _to_gray(arr: np.ndarray) -> np.ndarray:
    """Collapse RGB(A) to luminance; keep 2-D arrays unchanged."""
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):  # channels-first
            arr = np.moveaxis(arr, 0, -1)
        rgb = arr[..., :3].astype(np.float32)
        arr = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Unsupported image shape {arr.shape}")
    return arr.astype(np.float32)


def normalize(img: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Robust min-max scaling: the `low`/`high` percentiles are mapped to 0 and 1.

    Percentiles instead of min/max make the result insensitive to a few hot or
    dead pixels, which are common in SEM detectors.
    """
    lo, hi = np.percentile(img, [low, high])
    if hi - lo < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def read_pixel_size_nm(tif: tifffile.TiffFile) -> float | None:
    """Return nm/pixel from metadata, or None if not found.

    Sources, in order: FEI/Thermo Fisher (parsed by tifffile), Zeiss SmartSEM
    (parsed by tifffile), raw vendor tag text, ImageJ calibration (XResolution
    = pixels per unit).
    """
    fei = tif.fei_metadata
    if fei and "Scan" in fei and "PixelWidth" in fei["Scan"]:
        return float(fei["Scan"]["PixelWidth"]) * 1e9           # metres -> nm
    sem = tif.sem_metadata
    if sem and "ap_image_pixel_size" in sem:
        _, value, unit = sem["ap_image_pixel_size"]
        return float(value) * _UNIT_TO_NM.get(unit, 1.0)
    tags = tif.pages[0].tags
    ij = tif.imagej_metadata
    if ij and "XResolution" in tags:
        unit = {"micron": "µm", "\\u00B5m": "µm"}.get(ij.get("unit", ""), ij.get("unit", ""))
        num, den = tags["XResolution"].value
        if unit in _UNIT_TO_NM and num:
            return den / num * _UNIT_TO_NM[unit]
    if _FEI_TAG in tags:
        m = re.search(r"PixelWidth\s*=\s*([0-9.eE+-]+)", str(tags[_FEI_TAG].value))
        if m:
            return float(m.group(1)) * 1e9          # metres -> nm
    if _ZEISS_TAG in tags:
        m = re.search(r"Image Pixel Size\s*=\s*([0-9.eE+-]+)\s*(pm|nm|µm|um|mm)",
                      str(tags[_ZEISS_TAG].value))
        if m:
            return float(m.group(1)) * _UNIT_TO_NM[m.group(2)]
    return None


def load_image(source: str | Path | bytes, name: str | None = None) -> SEMImage:
    """Load an SEM image from a path or raw bytes (e.g. a Streamlit upload)."""
    if isinstance(source, (str, Path)):
        name = name or Path(source).name
        raw = Path(source).read_bytes()
    else:
        raw = source
        name = name or "image"

    nm_per_px = None
    if name.lower().endswith((".tif", ".tiff")):
        with tifffile.TiffFile(io.BytesIO(raw)) as tif:
            arr = tif.pages[0].asarray()
            nm_per_px = read_pixel_size_nm(tif)
    else:
        arr = np.asarray(Image.open(io.BytesIO(raw)))

    pixels = normalize(_to_gray(arr))
    return SEMImage(name=name, pixels=pixels, nm_per_px=nm_per_px,
                    calibration_source="metadata" if nm_per_px else "none")


def detect_databar(img: np.ndarray, search_fraction: float = 0.3) -> int:
    """Estimate the height of the instrument info bar at the bottom of the image.

    Heuristic: info bars consist of rows with large saturated (pure black/white)
    areas. We scan upwards from the bottom and return the number of rows that
    belong to the uppermost contiguous run of such rows. Returns 0 if none.
    The user can always correct this value in the GUI.
    """
    h = img.shape[0]
    start = int(h * (1 - search_fraction))
    saturated = ((img[start:] <= 0.01) | (img[start:] >= 0.99)).mean(axis=1)
    bar = saturated > 0.6
    if not bar.any():
        return 0
    first = int(np.argmax(bar))          # first bar-like row inside the search window
    return h - (start + first)


def crop_databar(img: SEMImage, rows: int) -> SEMImage:
    if rows <= 0:
        return img
    return SEMImage(img.name, normalize(img.pixels[:-rows]), img.nm_per_px,
                    img.calibration_source, rows)


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED)
