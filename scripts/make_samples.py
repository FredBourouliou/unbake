"""Demo GIFs for the web editor: wide, readable motion (not part of the benchmark)."""
import math
from pathlib import Path

import numpy as np
from PIL import Image

from unbake.scene import Shape, outline, render

OUT = Path("web/samples")
SIZE, FPS, N = (256, 256), 24, 48
t = np.linspace(0, 1, N)


def save(name, shapes, background):
    frames = render(shapes, N, SIZE, background)
    pal = [Image.fromarray(f).quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
           for f in frames]
    pal[0].save(OUT / f"{name}.gif", save_all=True, append_images=pal[1:], duration=round(1000 / FPS), loop=0)


def track(x, y, rot=None, scale=None):
    rot = np.zeros(N) if rot is None else rot
    scale = np.ones(N) if scale is None else scale
    return np.c_[x, y, rot, scale]


# 1. Bounce: a ball hops across, a star spins, a ring breathes.
ball_x = 30 + 196 * t
ball_y = 200 - 140 * np.abs(np.sin(t * math.pi * 2.5))
save("bounce", [
    Shape("ring", (46, 134, 222), *outline("ring", 26), track(np.full(N, 190.0), np.full(N, 70.0),
                                                              scale=1 + 0.25 * np.sin(t * math.pi * 4))),
    Shape("star", (243, 156, 18), *outline("star", 24), track(np.full(N, 70.0), np.full(N, 70.0), rot=360 * t)),
    Shape("circle", (231, 76, 60), *outline("circle", 16), track(ball_x, ball_y)),
], (250, 248, 244))

# 2. Orbit: a square circles the centre, a triangle crosses in front of it.
a = t * 2 * math.pi
save("orbit", [
    Shape("square", (142, 68, 173), *outline("square", 20), track(128 + 80 * np.cos(a), 128 + 80 * np.sin(a), rot=-180 * t)),
    Shape("triangle", (39, 174, 96), *outline("triangle", 26), track(20 + 216 * t, 128 + 10 * np.sin(a * 2), rot=90 * t)),
    Shape("circle", (236, 240, 241), *outline("circle", 12), track(np.full(N, 128.0), np.full(N, 128.0), scale=1 + 0.4 * np.sin(a))),
], (24, 26, 32))

# 3. Assemble: three shapes slide in and overlap like a logo.
ease = 0.5 - 0.5 * np.cos(np.clip(t * 1.6, 0, 1) * math.pi)
save("assemble", [
    Shape("rect", (26, 188, 156), *outline("rect", 40, 0.35), track(128 - 160 * (1 - ease), np.full(N, 150.0), rot=-8 * ease)),
    Shape("circle", (231, 76, 60), *outline("circle", 30), track(np.full(N, 110.0), 110 - 150 * (1 - ease))),
    Shape("triangle", (243, 156, 18), *outline("triangle", 30), track(150 + 140 * (1 - ease), np.full(N, 118.0), rot=180 * (1 - ease))),
], (250, 248, 244))
print("samples written to", OUT)

# 4. Strokes: a spinning loader arc, a wave drawing itself, a check mark appearing.
a = np.linspace(0, 2 * math.pi, 160, endpoint=False)
circle = np.c_[np.cos(a), np.sin(a)] * 34
spin = (t * 1.5) % 1.0
arc = 0.15 + 0.55 * (0.5 - 0.5 * np.cos(2 * math.pi * t))
xw = np.linspace(-90, 90, 120)
wave = np.c_[xw, 14 * np.sin(xw / 180 * 4 * math.pi)]
check = np.array([[-22, 0], [-6, 16], [24, -18]], float)
check = np.vstack([np.linspace(check[0], check[1], 20), np.linspace(check[1], check[2], 40)[1:]])
draw_on = np.clip((t - 0.35) / 0.4, 0, 1)
draw_on = 0.5 - 0.5 * np.cos(math.pi * draw_on)
save("loader", [
    Shape("stroke", (46, 134, 222), circle, None, track(np.full(N, 80.0), np.full(N, 80.0)), width=8, closed=True,
          cap="round", trim=np.c_[spin, (spin + arc) % 1.0]),
    Shape("stroke", (231, 76, 60), wave, None, track(np.full(N, 128.0), np.full(N, 196.0)), width=6, closed=False,
          cap="round", trim=np.c_[np.zeros(N), np.clip(t * 1.3, 0, 1)]),
    Shape("stroke", (39, 174, 96), check, None, track(np.full(N, 182.0), np.full(N, 84.0)), width=9, closed=False,
          cap="round", trim=np.c_[np.zeros(N), draw_on]),
], (250, 248, 244))
print("loader sample written")
