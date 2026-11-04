"""Draws the application icon: a lab beaker filled with gold nanowires.

Run:  python tools/make_icon.py   ->  assets/aunw.png (1024 px) and assets/aunw.ico
Drawn at 4x resolution and down-sampled for smooth (anti-aliased) edges.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 1024 * 4                       # super-sampled canvas
OUT = Path(__file__).resolve().parent.parent / "assets"


def lerp(c0, c1, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))


def rounded_tile() -> Image.Image:
    """Blue rounded square with a vertical gradient (the colours of the app)."""
    grad = Image.new("RGBA", (S, S))
    top, bottom = (26, 86, 196), (13, 44, 107)
    px = np.zeros((S, S, 4), np.uint8)
    t = np.linspace(0, 1, S)[:, None]
    for i in range(3):
        px[..., i] = (top[i] + (bottom[i] - top[i]) * t).astype(np.uint8)
    px[..., 3] = 255
    grad = Image.fromarray(px, "RGBA")
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, S - 1, S - 1), radius=int(0.22 * S), fill=255)
    tile = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    tile.paste(grad, (0, 0), mask)
    return tile


def beaker_outline():
    """Polygon of the beaker body (slightly tapered, with a pouring spout top-left)."""
    u = S / 1024
    x0, x1, y_top, y_bot = 300 * u, 724 * u, 250 * u, 820 * u
    return [(x0 - 40 * u, y_top - 20 * u), (x0, y_top + 30 * u), (x0 + 10 * u, y_bot - 40 * u),
            (x0 + 50 * u, y_bot), (x1 - 50 * u, y_bot), (x1 - 10 * u, y_bot - 40 * u),
            (x1, y_top), (x1 + 25 * u, y_top - 15 * u)]


def draw_icon() -> Image.Image:
    """Tile + beaker; the beaker is drawn on its own layer and enlarged for small-size legibility."""
    tile = rounded_tile()
    beaker = draw_beaker()
    k = 1.2                                          # enlarge the beaker relative to the tile
    big = beaker.resize((int(S * k), int(S * k)), Image.LANCZOS)
    m = (big.width - S) // 2
    layer = big.crop((m, m + int(0.04 * S), m + S, m + S + int(0.04 * S)))   # centre crop, slightly lower
    return Image.alpha_composite(tile, layer).resize((1024, 1024), Image.LANCZOS)


def draw_beaker() -> Image.Image:
    u = S / 1024
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    body = beaker_outline()

    # --- liquid (clipped to the beaker) with nanowires inside
    liquid = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ld = ImageDraw.Draw(liquid)
    level = 470 * u
    ld.rectangle((0, level, S, S), fill=(140, 205, 255, 150))
    rng = np.random.default_rng(4)
    for _ in range(11):                              # gold nanowires: long thin rods
        cx, cy = rng.uniform(330, 690) * u, rng.uniform(510, 790) * u
        ang = rng.uniform(0, np.pi)
        half = rng.uniform(110, 210) * u
        dx, dy = np.cos(ang) * half, np.sin(ang) * half
        w = rng.uniform(15, 22) * u
        ld.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=(150, 98, 10, 255), width=int(w + 8 * u))
        ld.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=(245, 190, 40, 255), width=int(w))
        ld.line((cx - dx, cy - dy - w * 0.22, cx + dx, cy + dy - w * 0.22),
                fill=(255, 236, 150, 255), width=max(1, int(w * 0.28)))   # specular highlight
    clip = Image.new("L", (S, S), 0)
    ImageDraw.Draw(clip).polygon(body, fill=255)
    clip = clip.filter(ImageFilter.GaussianBlur(2 * u))
    img.paste(liquid, (0, 0), Image.composite(liquid.getchannel("A"), Image.new("L", (S, S), 0), clip))

    # --- glass: faint fill, white outline, graduation marks, highlight
    glass = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glass)
    gd.polygon(body, fill=(255, 255, 255, 38))
    gd.line(body + [body[0]][:0], fill=(255, 255, 255, 235), width=int(26 * u), joint="curve")
    for i, y in enumerate(range(330, 760, 70)):      # graduation marks on the right
        L = 70 if i % 2 == 0 else 40
        gd.line((640 * u - L * u, y * u, 640 * u, y * u), fill=(255, 255, 255, 190), width=int(12 * u))
    gd.line((360 * u, 330 * u, 368 * u, 720 * u), fill=(255, 255, 255, 110), width=int(22 * u))
    # liquid surface line
    gd.line((318 * u, level, 705 * u, level), fill=(210, 235, 255, 220), width=int(10 * u))
    return Image.alpha_composite(img, glass)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    icon = draw_icon()
    icon.save(OUT / "aunw.png")
    icon.resize((256, 256), Image.LANCZOS).save(
        OUT / "aunw.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("written", OUT / "aunw.png", OUT / "aunw.ico")


if __name__ == "__main__":
    main()
