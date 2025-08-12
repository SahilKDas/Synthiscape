#!/usr/bin/env python3
# Synthiscape — single-file procedural terrain generator (NumPy 2.0+ & Pygame)
# Deps: numpy, pygame, (optional) numba | Run: python synthiscape.py

import math, random, time, json
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pygame

# =========================
# ----- Optional Numba ----
# =========================
NB = False
try:
    from numba import njit, prange
    NB = True
except Exception:
    NB = False

# =========================
# ----- Config / UI -------
# =========================

# World size (requested)
WORLD_W, WORLD_H = 2400, 2400

# Window defaults
WINDOW_W, WINDOW_H = 500, 500

DISPLAY_SCALE    = 1.0
SEA_LEVEL        = 0.46
RIVER_THRESHOLD  = 110
RIVER_WIDTH_SCALE= 0.012
THERMAL_EROSION_ITERS_PER_E = 24
THERMAL_TALUS_ANGLE = 0.015

# Base octaves
ELEVATION_OCTAVES = [(1/256, 0.55), (1/128, 0.35), (1/64, 0.22), (1/32, 0.10)]
MOISTURE_OCTAVES  = [(1/128, 0.7), (1/64, 0.3), (1/32, 0.2)]

# Climate
TEMP_LAPSE_RATE   = 0.75
LAT_GRADIENT      = 0.9
BORDER_FADE       = 0.25

# River-carving sim
CARVE_RATE        = 0.0022
CARVE_SMOOTHING   = 0.6

# Async regen & perf knobs
FRAME_BUDGET_MS       = 12        # per-frame compute budget
HYDRO_DOWNSCALE       = 2         # 1 = full res; 2 = half-res hydrology (≈4× faster)
QUICK_PREVIEW_SCALE   = 4         # instant preview resolution divider

# Misc UI
LEGEND_ON_START   = False
UI_FONT_NAME      = None
SHOW_FPS_DEFAULT  = True

# Minimap
MINIMAP_SIZE      = 180
MINIMAP_PADDING   = 12

PALETTES = {
    1: "soft biomes",
    2: "vivid biomes",
    3: "earth tones",
    4: "monochrome height",
    5: "satellite-ish",
}

# =========================
# ----- Utilities ---------
# =========================

def clamp01(x): return np.clip(x, 0.0, 1.0)
def lerp(a, b, t): return a*(1-t) + b*t
def normalize01(a): return clamp01((a - a.min()) / (np.ptp(a) + 1e-9))
def save_png(path, surf): pygame.image.save(surf, path)

def to_surface(img_np):
    # safe & fast: contiguous RGB bytes; convert only if a display exists
    img_np = np.ascontiguousarray(img_np, dtype=np.uint8)
    H, W, _ = img_np.shape
    surf = pygame.image.frombuffer(img_np.tobytes(), (W, H), "RGB")
    disp = pygame.display.get_surface()
    return surf.convert() if disp else surf

# =========================
# ----- Perlin Noise ------
# =========================

def _perlin_gradients(grid_w, grid_h, rng):
    angles = rng.random((grid_h+1, grid_w+1))*2*np.pi
    return np.cos(angles), np.sin(angles)

def _fade(t): return t*t*t*(t*(t*6 - 15) + 10)

def perlin2d(width, height, scale_x, scale_y, seed):
    rng = np.random.default_rng(seed)
    grid_w = max(1, int(math.ceil(width*scale_x)))
    grid_h = max(1, int(math.ceil(height*scale_y)))
    gx, gy = _perlin_gradients(grid_w, grid_h, rng)
    xs = np.linspace(0, grid_w, width, endpoint=False)
    ys = np.linspace(0, grid_h, height, endpoint=False)
    X, Y = np.meshgrid(xs, ys)
    x0 = np.floor(X).astype(int); y0 = np.floor(Y).astype(int)
    x1 = x0 + 1; y1 = y0 + 1
    tx = X - x0; ty = Y - y0
    u = _fade(tx); v = _fade(ty)
    x0 %= (grid_w+1); x1 %= (grid_w+1)
    y0 %= (grid_h+1); y1 %= (grid_h+1)
    def dot(ix, iy, dx, dy): return gx[iy, ix]*dx + gy[iy, ix]*dy
    n00 = dot(x0, y0, tx,   ty)
    n10 = dot(x1, y0, tx-1, ty)
    n01 = dot(x0, y1, tx,   ty-1)
    n11 = dot(x1, y1, tx-1, ty-1)
    nx0 = n00*(1-u) + n10*u
    nx1 = n01*(1-u) + n11*u
    nxy = nx0*(1-v) + nx1*v
    return clamp01((nxy - nxy.min()) / (np.ptp(nxy) + 1e-9))

def fbm(width, height, octaves, seed):
    total = np.zeros((height, width), dtype=np.float32); amp_sum = 0.0
    for i, (freq, amp) in enumerate(octaves):
        total += perlin2d(width, height, freq, freq, seed + 101*i) * amp
        amp_sum += amp
    total /= (amp_sum + 1e-9)
    return clamp01(total)

def ridged_fbm(width, height, octaves, seed, gain=2.0):
    total = np.zeros((height, width), dtype=np.float32); amp_sum = 0.0
    for i, (freq, amp) in enumerate(octaves):
        n = perlin2d(width, height, freq, freq, seed + 313*i)
        r = 1.0 - np.abs(2.0*n - 1.0)
        a = amp * (gain**(i*0.25))
        total += r * a
        amp_sum += a
    total /= (amp_sum + 1e-9)
    return clamp01(total)

# =========================
# --- Island Mask & Ops ---
# =========================

def radial_island_mask(w, h, strength=0.3):
    y, x = np.ogrid[0:h, 0:w]
    cx, cy = (w-1)/2.0, (h-1)/2.0
    dx = (x - cx)/(cx+1e-9)
    dy = (y - cy)/(cy+1e-9)
    r = np.sqrt(dx*dx + dy*dy)
    m = 1 - clamp01(r)
    if strength <= 0: return np.ones((h, w), dtype=np.float32)
    return lerp(np.ones_like(m), m, strength).astype(np.float32)

# =========================
# ------ Rivers/Flow ------
# =========================

# 8-neighborhood, (dx, dy) pairs
NEIGH = np.array([
    (-1, -1), ( 0, -1), ( 1, -1),
    (-1,  0),           ( 1,  0),
    (-1,  1), ( 0,  1), ( 1,  1),
], dtype=np.int32)

# ---- Numba-accelerated pieces ----
if NB:
    @njit(cache=True)
    def nb_compute_flowdir(h, sea_level):
        H, W = h.shape
        flowdir = np.full((H, W), -1, dtype=np.int8)
        for y in range(H):
            for x in range(W):
                if h[y, x] <= sea_level:
                    flowdir[y, x] = -1
                    continue
                best_drop = 0.0
                best_k = -1
                for k in range(8):
                    dx = NEIGH[k, 0]; dy = NEIGH[k, 1]
                    nx = x + dx; ny = y + dy
                    if nx < 0 or ny < 0 or nx >= W or ny >= H:
                        continue
                    drop = h[y, x] - h[ny, nx]
                    if drop > best_drop:
                        best_drop = drop
                        best_k = k
                flowdir[y, x] = best_k
        return flowdir

    @njit(cache=True)
    def nb_accumulate_flow(order_y, order_x, flowdir, sea_level, h):
        H, W = h.shape
        acc = np.ones((H, W), dtype=np.float32)
        n = order_y.shape[0]
        for i in range(n):
            y = order_y[i]
            x = order_x[i]
            k = flowdir[y, x]
            if k < 0:
                continue
            dx = NEIGH[k, 0]; dy = NEIGH[k, 1]
            nx = x + dx; ny = y + dy
            if nx < 0 or ny < 0 or nx >= W or ny >= H:
                continue
            if h[ny, nx] <= sea_level:
                continue
            acc[ny, nx] += acc[y, x]
        return acc

    @njit(cache=True)
    def nb_distance_transform(mask, passes):
        H, W = mask.shape
        dist = np.empty((H, W), dtype=np.float32)
        big = 1e9
        for y in range(H):
            for x in range(W):
                dist[y, x] = 0.0 if mask[y, x] else big
        for _ in range(passes):
            # forward
            for y in range(H):
                for x in range(W):
                    d = dist[y, x]
                    if y > 0:
                        d = min(d, dist[y-1, x] + 1.0)
                    if x > 0:
                        d = min(d, dist[y, x-1] + 1.0)
                    if y > 0 and x > 0:
                        d = min(d, dist[y-1, x-1] + 1.4142)
                    dist[y, x] = d
            # backward
            for y in range(H-1, -1, -1):
                for x in range(W-1, -1, -1):
                    d = dist[y, x]
                    if y < H-1:
                        d = min(d, dist[y+1, x] + 1.0)
                    if x < W-1:
                        d = min(d, dist[y, x+1] + 1.0)
                    if y < H-1 and x < W-1:
                        d = min(d, dist[y+1, x+1] + 1.4142)
                    dist[y, x] = d
        return dist

def flow_accumulation(height, sea_level):
    h = height
    # compute flow direction (Numba if available)
    if NB:
        flowdir = nb_compute_flowdir(h, sea_level)
        # sort order in Python (fast enough, avoids Numba argsort issues)
        order = np.argsort(-h.ravel(), kind="quicksort")
        oy, ox = np.unravel_index(order, h.shape)
        acc = nb_accumulate_flow(oy.astype(np.int32), ox.astype(np.int32), flowdir, sea_level, h)
        return acc, flowdir
    else:
        H, W = h.shape
        flowdir = -np.ones((H, W), dtype=np.int8)
        acc = np.ones((H, W), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                if h[y, x] <= sea_level:
                    flowdir[y, x] = -1; continue
                best_drop = 0.0; best_k = -1
                for k, (dx, dy) in enumerate(NEIGH):
                    nx, ny = x + dx, y + dy
                    if nx < 0 or ny < 0 or nx >= W or ny >= H: continue
                    drop = h[y, x] - h[ny, nx]
                    if drop > best_drop: best_drop, best_k = drop, k
                flowdir[y, x] = best_k
        order = np.dstack(np.unravel_index(np.argsort(-h.ravel()), h.shape))[0]
        for y, x in order:
            k = flowdir[y, x]
            if k < 0: continue
            dx, dy = NEIGH[k]; nx, ny = x + dx, y + dy
            if ny < 0 or nx < 0 or nx >= W or ny >= H: continue
            if h[ny, nx] <= sea_level: continue
            acc[ny, nx] += acc[y, x]
        return acc, flowdir

def fill_depressions(height, sea_level, iterations=2):
    h = height.copy(); H, W = h.shape
    for _ in range(iterations):
        for y in range(1, H-1):
            for x in range(1, W-1):
                if h[y, x] <= sea_level: continue
                nb = h[y-1:y+2, x-1:x+2]; m = nb.min()
                if h[y, x] < m: h[y, x] = m + 1e-4
    return normalize01(h)

# =========================
# ------- Erosion ---------
# =========================

def thermal_erosion(height, iters=24, talus=0.02):
    # Keep NumPy version (works well). If you want a Numba kernel here too, I can add it.
    H, W = height.shape; h = height.copy()
    for _ in range(iters):
        delta = np.zeros_like(h)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0: continue
                NX = np.clip(np.arange(W)+dx, 0, W-1)
                NY = np.clip(np.arange(H)+dy, 0, H-1)
                YY, XX = np.meshgrid(NY, NX, indexing='ij')
                s = h - h[YY, XX]
                mask = s > talus
                move = np.zeros_like(h)
                move[mask] = (s[mask] - talus) * 0.5 / 8.0
                delta[mask] -= move[mask]
                tmp = np.zeros_like(h); tmp[YY, XX][mask] += move[mask]
                delta += tmp
        h += delta; h = normalize01(h)
    return h

# =========================
# ------- Biomes ----------
# =========================

BIOME_TABLE = [
    ["ice",      "tundra",     "boreal",     "boreal",      "boreal"],
    ["grass",    "shrub",      "temperate",  "temperate",   "rainforest"],
    ["desert",   "savanna",    "woodland",   "seasonal",    "tropical"],
    ["desert",   "savanna",    "tropical",   "tropical",    "mangrove"],
]

BIOME_COLORS = {
    "ice":        (230, 240, 255),
    "tundra":     (198, 214, 214),
    "boreal":     (95, 130, 90),
    "grass":      (170, 205, 120),
    "shrub":      (190, 180, 120),
    "temperate":  (110, 165, 95),
    "rainforest": (60, 130, 75),
    "desert":     (220, 205, 140),
    "savanna":    (205, 190, 110),
    "woodland":   (140, 170, 90),
    "seasonal":   (85, 150, 85),
    "tropical":   (70, 140, 80),
    "mangrove":   (75, 115, 90),
    "ocean":      (25, 75, 130),
    "shore":      (235, 225, 170),
    "river":      (45, 110, 200),
}

def classify_biomes(height, temp, moisture, sea_level):
    H, W = height.shape
    biome = np.empty((H, W), dtype=object)
    ocean_mask = height <= sea_level
    shore_mask = (~ocean_mask) & (height <= sea_level + 0.01)
    t_idx = np.clip((temp * 3.999).astype(int), 0, 3)
    m_idx = np.clip((moisture * 4.999).astype(int), 0, 4)
    for r in range(H):
        for c in range(W):
            biome[r, c] = BIOME_TABLE[t_idx[r, c]][m_idx[r, c]]
    biome[ocean_mask] = "ocean"
    biome[shore_mask] = "shore"
    return biome

# =========================
# ----- Color Mapping -----
# =========================

def ramp_height(h, palette_id=5):
    a = clamp01(h)
    if palette_id == 4:
        g = (a*255).astype(np.uint8); return np.dstack([g, g, g])
    if palette_id == 5:
        c = np.zeros((*a.shape, 3), dtype=np.uint8)
        land = a > SEA_LEVEL; n = normalize01(a)
        low  = (n < 0.55) & land
        mid  = (n >= 0.55) & (n < 0.78) & land
        high = (n >= 0.78) & (n < 0.92) & land
        peak = (n >= 0.92) & land
        c[low]  = np.array([145, 120, 95], dtype=np.uint8)
        c[mid]  = np.array([90, 140, 85], dtype=np.uint8)
        c[high] = np.array([140, 140, 145], dtype=np.uint8)
        c[peak] = np.array([250, 250, 250], dtype=np.uint8)
        ocean = a <= SEA_LEVEL; depth = normalize01(SEA_LEVEL - a)
        c[ocean] = np.stack([
            (20 + 20*depth[ocean]).astype(np.uint8),
            (60 + 40*depth[ocean]).astype(np.uint8),
            (110 + 100*depth[ocean]).astype(np.uint8)
        ], axis=-1)
        return c
    if palette_id == 1:
        n = a; c = np.zeros((*n.shape, 3), dtype=np.uint8)
        c[...,0] = (80 + 120*n).astype(np.uint8)
        c[...,1] = (90 + 130*n).astype(np.uint8)
        c[...,2] = (100 + 110*n).astype(np.uint8); return c
    if palette_id == 2:
        n = a; c = np.zeros((*n.shape, 3), dtype=np.uint8)
        c[...,0] = (50 + 205*n).astype(np.uint8)
        c[...,1] = (40 + 215*n).astype(np.uint8)
        c[...,2] = (60 + 195*n).astype(np.uint8); return c
    if palette_id == 3:
        n = a; r = (120 + 100*n).astype(np.uint8)
        g = (100 + 90*n).astype(np.uint8)
        b = (80 + 70*n).astype(np.uint8)
        return np.dstack([r, g, b])
    return (a*255).astype(np.uint8)

# =========================
# ----- World Builder -----
# =========================

@dataclass
class World:
    W: int; H: int; seed: int
    height: np.ndarray
    moisture: np.ndarray
    temperature: np.ndarray
    flowacc: np.ndarray
    rivers: np.ndarray
    biome: np.ndarray
    filled: bool
    tectonics: bool

def distance_transform(mask, passes=2):
    if NB:
        return nb_distance_transform(mask.astype(np.bool_), passes)
    else:
        H, W = mask.shape
        dist = np.full((H, W), 1e9, dtype=np.float32); dist[mask] = 0.0
        for _ in range(passes):
            for y in range(H):
                for x in range(W):
                    d = dist[y, x]
                    if y > 0: d = min(d, dist[y-1, x] + 1)
                    if x > 0: d = min(d, dist[y, x-1] + 1)
                    if y > 0 and x > 0: d = min(d, dist[y-1, x-1] + 1.4142)
                    dist[y, x] = d
            for y in range(H-1, -1, -1):
                for x in range(W-1, -1, -1):
                    d = dist[y, x]
                    if y < H-1: d = min(d, dist[y+1, x] + 1)
                    if x < W-1: d = min(d, dist[y, x+1] + 1)
                    if y < H-1 and x < W-1: d = min(d, dist[y+1, x+1] + 1.4142)
                    dist[y, x] = d
        return dist

def build_world(W, H, seed=None, do_fill=False, tectonics=False):
    if seed is None: seed = random.randint(0, 1_000_000)
    elevation_base = ridged_fbm(W, H, ELEVATION_OCTAVES, seed) if tectonics else fbm(W, H, ELEVATION_OCTAVES, seed)
    ridges = ridged_fbm(W, H, [(1/512, 1.0)], seed+777) if tectonics else perlin2d(W, H, 1/512, 1/512, seed+777)
    elevation = normalize01(0.80*elevation_base + 0.20*ridges)
    mask = radial_island_mask(W, H, BORDER_FADE)
    elevation = normalize01(elevation * (0.6 + 0.4*mask))
    if do_fill: elevation = fill_depressions(elevation, SEA_LEVEL, iterations=2)

    moisture_noise = fbm(W, H, MOISTURE_OCTAVES, seed+11)
    lat = np.linspace(-1, 1, H).reshape(H, 1); lat = 1 - LAT_GRADIENT*np.abs(lat)
    elev_cool = 1 - TEMP_LAPSE_RATE*normalize01(elevation)
    temp = clamp01(0.55*lat + 0.45*elev_cool + 0.06*(np.random.default_rng(seed+5).standard_normal((H, W))))

    flowacc, _ = flow_accumulation(elevation, SEA_LEVEL)
    rivers = flowacc > RIVER_THRESHOLD
    carved = elevation.copy(); carved[rivers] = np.maximum(0.0, carved[rivers] - 0.02)

    barrier = np.maximum.accumulate(carved, axis=1); shadow = normalize01(barrier)
    ocean = carved <= SEA_LEVEL
    coast_dist = distance_transform(ocean, passes=2)
    coast_influence = np.exp(-coast_dist/12.0)
    river_influence = np.where(rivers, 1.0, 0.0)
    moisture = clamp01(0.58*moisture_noise + 0.30*coast_influence + 0.18*river_influence - 0.22*shadow)

    biome = classify_biomes(carved, temp, moisture, SEA_LEVEL)
    return World(W, H, seed, carved.astype(np.float32), moisture.astype(np.float32),
                 temp.astype(np.float32), flowacc.astype(np.float32), rivers, biome,
                 do_fill, tectonics)

# =========================
# ------ Rendering --------
# =========================

def river_mask_rgba(world) -> np.ndarray:
    h = world.height; H, W = h.shape
    r = np.zeros((H, W, 4), dtype=np.uint8); land = h > SEA_LEVEL
    f = world.flowacc
    width = normalize01(np.log1p(f)) * 255.0
    rmask = (f > RIVER_THRESHOLD) & land
    if np.any(rmask):
        rgba = np.zeros_like(r)
        rgba[...,0] = BIOME_COLORS["river"][0]
        rgba[...,1] = BIOME_COLORS["river"][1]
        rgba[...,2] = BIOME_COLORS["river"][2]
        alpha = np.zeros((H, W), dtype=np.uint8)
        alpha[rmask] = np.clip(width[rmask], 110, 255).astype(np.uint8)
        rgba[...,3] = alpha; return rgba
    return r

def add_contours(img, height, interval=0.025, strength=0.45):
    levels = (height / interval) % 1.0
    band = np.abs(levels - 0.5); lines = band < 0.03
    out = img.copy(); out[lines] = (out[lines] * (1.0 - strength)).astype(np.uint8)
    return out

def add_coastline(img, height, sea_level, color=(30,30,30), strength=0.7):
    ocean = height <= sea_level
    edges = np.zeros_like(ocean)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dx == 0 and dy == 0: continue
            nb = np.roll(np.roll(ocean, dx, axis=1), dy, axis=0)
            edges |= (nb != ocean)
    out = img.copy(); c = np.array(color, dtype=np.uint8)
    out[edges] = (out[edges]*(1-strength) + c*strength).astype(np.uint8)
    return out

def compose_view(world, mode="biome", palette=5, show_grid=False, show_rivers=True,
                 show_sea=True, show_borders=True, show_contours=False, show_coast=False):
    h = world.height; H, W = h.shape; ocean = h <= SEA_LEVEL
    if mode == "height":
        img = ramp_height(h, palette_id=palette)
    elif mode == "temperature":
        t = clamp01(world.temperature); img = np.zeros((H, W, 3), dtype=np.uint8)
        img[...,0] = (255*t).astype(np.uint8)
        img[...,1] = (120*(1 - np.abs(t-0.5)*2)).astype(np.uint8)
        img[...,2] = (255*(1-t)).astype(np.uint8)
    elif mode == "moisture":
        m = clamp01(world.moisture); img = np.zeros((H, W, 3), dtype=np.uint8)
        img[...,0] = (50*(1-m)).astype(np.uint8)
        img[...,1] = (180*m).astype(np.uint8)
        img[...,2] = (220*m).astype(np.uint8)
    else:
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for name, col in BIOME_COLORS.items():
            mask = (world.biome == name)
            if mask.any(): img[mask] = np.array(col, dtype=np.uint8)
    if show_sea and mode != "biome":
        depth = normalize01(SEA_LEVEL - h)
        img[ocean] = np.stack([
            (20 + 20*depth[ocean]).astype(np.uint8),
            (60 + 40*depth[ocean]).astype(np.uint8),
            (110 + 100*depth[ocean]).astype(np.uint8)
        ], axis=-1)
    if show_rivers:
        rgba = river_mask_rgba(world)
        if rgba[...,3].any():
            base = img.astype(np.uint16); alpha = rgba[...,3:4].astype(np.uint16); color = rgba[...,:3].astype(np.uint16)
            img = ((base*(255-alpha) + color*alpha) // 255).astype(np.uint8)
    if show_borders and mode == "biome":
        edges = biome_edges(world.biome); img[edges] = (img[edges]*0.6).astype(np.uint8)
    if show_contours: img = add_contours(img, h)
    if show_coast:    img = add_coastline(img, h, SEA_LEVEL)
    if show_grid:     img[::32, :] = 0; img[:, ::32] = 0
    return img

def biome_edges(biome):
    H, W = biome.shape
    edges = np.zeros((H, W), dtype=bool)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dx == 0 and dy == 0: continue
            nb = np.roll(np.roll(biome, dx, axis=1), dy, axis=0)
            edges |= (nb != biome)
    return edges

def draw_legend(screen, font, mode, palette, auto_flow, filled, iso, auto_carve, tectonics, world, fps):
    nb_status = "on" if NB else "off"
    lines = [
        f"Mode: {mode}  | Palette: {PALETTES.get(palette, palette)} | Iso={'on' if iso else 'off'} | Tectonics={'on' if tectonics else 'off'} | Numba:{nb_status}",
        f"World: {world.W}×{world.H}  Window: {screen.get_width()}×{screen.get_height()}  Sea={SEA_LEVEL:.2f}  Rivers>{RIVER_THRESHOLD}  FPS:{fps:4.1f}",
        f"AutoFlow={'on' if auto_flow else 'off'}  AutoCarve={'on' if auto_carve else 'off'}  FillDep={'on' if filled else 'off'}",
        "Keys: R random, N next-seed, J tectonics, D fill, E erosion, V carve, Shift+V auto-carve, F auto-flow",
        "      B biomes, Y height, M moisture, T temperature, I iso, C contours, L coast, P palette, G grid, H help, K legend",
        "      S save PNG, X export height.npy+world.json, +/- zoom, Arrow keys pan",
        f"Pygame {pygame.version.ver}  NumPy {np.__version__}",
    ]
    x, y = 10, 10
    for text in lines:
        surf = font.render(text, True, (240, 240, 240))
        bg = pygame.Surface((surf.get_width()+8, surf.get_height()+4), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160)); screen.blit(bg, (x-4, y-2)); screen.blit(surf, (x, y))
        y += surf.get_height() + 6

def draw_help(screen, font):
    help_lines = [
        "Synthiscape — Controls",
        "Generation: R random | N next-seed | J tectonics | D fill depressions",
        "Simulation: E thermal erosion | V carve tick | Shift+V auto-carve | F auto-flow",
        "Views: B biomes | Y height | M moisture | T temperature | I isometric",
        "Overlays: C contours | L coastline | P palette | G grid | K legend | H help",
        "Navigation: Arrow keys pan | Mouse wheel or +/- zoom",
        "Saving: S save PNG (iso or map) | X export height.npy + world.json",
        "Tip: Map regen is non-blocking — watch the progress bar; Esc cancels.",
    ]
    pad = 12
    w = max(font.size(s)[0] for s in help_lines) + pad*2
    h = sum(font.size(s)[1]+4 for s in help_lines) + pad*2
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 220))
    y = pad
    for i, s in enumerate(help_lines):
        color = (255, 255, 255) if i == 0 else (230, 230, 230)
        txt = font.render(s, True, color)
        panel.blit(txt, (pad, y))
        y += txt.get_height() + 4
    rect = panel.get_rect(center=(screen.get_width()//2, screen.get_height()//2))
    screen.blit(panel, rect.topleft)

def draw_progress(screen, font, p):
    p = max(0.0, min(1.0, p))
    w = int(screen.get_width() * 0.6)
    h = 24
    x = (screen.get_width()-w)//2
    y = screen.get_height() - h - 18
    bg = pygame.Surface((w, h), pygame.SRCALPHA)
    bg.fill((0,0,0,180))
    screen.blit(bg, (x, y))
    fill = int((w-4) * p)
    bar = pygame.Surface((fill, h-4))
    bar.fill((80, 200, 120))
    screen.blit(bar, (x+2, y+2))
    txt = font.render(f"Generating... {int(p*100)}%", True, (240,240,240))
    screen.blit(txt, (x + (w - txt.get_width())//2, y + (h - txt.get_height())//2))

# ---------- Isometric preview ----------

def render_isometric(world, scale=2, z_exaggeration=180):
    h = world.height; H, W = h.shape
    color = ramp_height(h, palette_id=5).astype(np.int16)
    sobel_x = np.zeros_like(h); sobel_y = np.zeros_like(h)
    sobel_x[:,1:-1] = (h[:,2:] - h[:,:-2])*0.5
    sobel_y[1:-1,:] = (h[2:,:] - h[:-2,:])*0.5
    slope = np.sqrt(sobel_x**2 + sobel_y**2)
    shade = (1.0 - clamp01(slope*8.0)).reshape(H, W, 1)
    shaded = np.clip(color*shade + 20, 0, 255).astype(np.uint8)

    iso_w = min(2200, int((W + H) * scale * 0.9))
    iso_h = min(1400, int((W + H) * scale * 0.5 + z_exaggeration))
    surf = pygame.Surface((iso_w, iso_h), pygame.SRCALPHA)

    step = 2 if max(W,H) >= 700 else 1
    for y in range(0, H, step):
        for x in range(0, W, step):
            z = (h[y, x] - SEA_LEVEL) * z_exaggeration
            ix = int((x - y) * 0.5 * scale + iso_w//2)
            iy = int((x + y) * 0.25 * scale + (iso_h//3) - z)
            col = shaded[y, x]
            if h[y, x] <= SEA_LEVEL: col = np.array([30, 90, 150], dtype=np.uint8)
            if 0 <= ix < iso_w and 0 <= iy < iso_h:
                surf.fill(col, (ix, iy, step, step))
    return surf

# ---------- River-carving tick ----------

def gaussian_blur1D(a, axis=0):
    if axis == 0:
        b = a.copy()
        b[1:-1,:] = (a[1:-1,:] + 0.5*(a[:-2,:] + a[2:,:]))
        return b
    else:
        b = a.copy()
        b[:,1:-1] = (a[:,1:-1] + 0.5*(a[:,:-1] + a[:,2:]))
        return b

def carve_rivers_tick(world):
    h = world.height
    f = normalize01(np.log1p(world.flowacc))
    carve = CARVE_RATE * f
    mask = (world.flowacc > RIVER_THRESHOLD) & (h > SEA_LEVEL)
    delta = np.zeros_like(h, dtype=np.float32)
    delta[mask] = carve[mask]
    if CARVE_SMOOTHING > 0:
        delta = gaussian_blur1D(gaussian_blur1D(delta, axis=0), axis=1) * CARVE_SMOOTHING + delta*(1-CARVE_SMOOTHING)
    h = np.maximum(0.0, h - delta)
    h = normalize01(h)
    world.height = h
    world.flowacc, _ = flow_accumulation(world.height, SEA_LEVEL)
    world.rivers = world.flowacc > RIVER_THRESHOLD
    ocean = world.height <= SEA_LEVEL
    coast_dist = distance_transform(ocean, passes=2)
    coast_influence = np.exp(-coast_dist/12.0)
    river_influence = np.where(world.rivers, 1.0, 0.0)
    barrier = np.maximum.accumulate(world.height, axis=1); shadow = normalize01(barrier)
    world.moisture = clamp01(0.56*world.moisture + 0.24*coast_influence + 0.18*river_influence - 0.18*shadow)
    world.biome = classify_biomes(world.height, world.temperature, world.moisture, SEA_LEVEL)

# =========================
# ------ Async Regen -------
# =========================

class RegenJob:
    def __init__(self, W, H, seed, do_fill, tectonics):
        self.W, self.H, self.seed = W, H, seed
        self.do_fill, self.tectonics = do_fill, tectonics
        self.gen = self._build_world_stepwise()
        self.done = False
        self.progress = 0.0
        self.preview_img: Optional[np.ndarray] = None

    def step(self, ms_budget=8):
        t0 = time.perf_counter()
        try:
            while (time.perf_counter() - t0)*1000.0 < ms_budget:
                next(self.gen)
        except StopIteration as w:
            self.world = w.value
            self.done = True

    def _build_world_stepwise(self):
        W, H = self.W, self.H
        seed = self.seed

        # 1) quick low-res preview (height)
        s = max(2, QUICK_PREVIEW_SCALE)
        w2, h2 = max(64, W//s), max(64, H//s)
        base = ridged_fbm(w2, h2, ELEVATION_OCTAVES, seed) if self.tectonics else fbm(w2, h2, ELEVATION_OCTAVES, seed)
        ridg = ridged_fbm(w2, h2, [(1/512,1.0)], seed+777) if self.tectonics else perlin2d(w2, h2, 1/512, 1/512, seed+777)
        elev2 = normalize01(0.80*base + 0.20*ridg)
        mask2 = radial_island_mask(w2, h2, BORDER_FADE)
        elev2 = normalize01(elev2 * (0.6 + 0.4*mask2))
        img2 = ramp_height(elev2, palette_id=5)
        self.preview_img = np.kron(img2, np.ones((s, s, 1), dtype=np.uint8))[:H, :W, :]
        self.progress = 0.12
        yield

        # 2) full-res elevation (octave-by-octave)
        elev = np.zeros((H, W), dtype=np.float32); amp_sum = 1e-9
        for i, (freq, amp) in enumerate(ELEVATION_OCTAVES):
            elev += perlin2d(W, H, freq, freq, seed + 101*i).astype(np.float32) * amp
            amp_sum += amp
            self.progress = 0.12 + 0.45*(i+1)/len(ELEVATION_OCTAVES)
            yield
        elev /= amp_sum
        if self.tectonics:
            elev = normalize01(0.8*elev + 0.2*ridged_fbm(W, H, [(1/512,1.0)], seed+777))
        mask = radial_island_mask(W, H, BORDER_FADE)
        elev = normalize01(elev * (0.6 + 0.4*mask))
        if self.do_fill:
            elev = fill_depressions(elev, SEA_LEVEL, iterations=2)
        self.progress = 0.62; yield

        # 3) temperature & base moisture
        rng = np.random.default_rng(seed+5)
        lat = np.linspace(-1, 1, H).reshape(H,1)
        lat = 1 - LAT_GRADIENT*np.abs(lat)
        elev_cool = 1 - TEMP_LAPSE_RATE*normalize01(elev)
        temp = clamp01(0.55*lat + 0.45*elev_cool + 0.06*rng.standard_normal((H,W)))
        moist_noise = fbm(W, H, MOISTURE_OCTAVES, seed+11)
        self.progress = 0.72; yield

        # 4) hydrology (downscaled for speed, with Numba accel inside)
        ds = max(1, int(HYDRO_DOWNSCALE))
        if ds > 1:
            eh = elev[::ds, ::ds]
            flowacc_ds, _ = flow_accumulation(eh, SEA_LEVEL)
            flowacc = np.kron(flowacc_ds, np.ones((ds, ds), dtype=flowacc_ds.dtype))[:H, :W]
        else:
            flowacc, _ = flow_accumulation(elev, SEA_LEVEL)
        rivers = flowacc > RIVER_THRESHOLD
        carved = elev.copy(); carved[rivers] = np.maximum(0.0, carved[rivers] - 0.02)
        self.progress = 0.88; yield

        # 5) moisture finalize & biomes
        ocean = carved <= SEA_LEVEL
        coast_dist = distance_transform(ocean, passes=2)
        coast_influence = np.exp(-coast_dist/12.0)
        barrier = np.maximum.accumulate(carved, axis=1)
        shadow = normalize01(barrier)
        moisture = clamp01(0.58*moist_noise + 0.30*coast_influence + 0.18*np.where(rivers,1.0,0.0) - 0.22*shadow)
        biome = classify_biomes(carved, temp, moisture, SEA_LEVEL)
        self.progress = 1.0; yield

        return World(W, H, seed, carved.astype(np.float32), moisture.astype(np.float32),
                     temp.astype(np.float32), flowacc.astype(np.float32), rivers, biome,
                     self.do_fill, self.tectonics)

# =========================
# -------- Minimap --------
# =========================

def render_minimap(world, palette=5):
    img = ramp_height(world.height, palette_id=palette)
    surf = to_surface(img)
    mini = pygame.transform.smoothscale(surf, (MINIMAP_SIZE, MINIMAP_SIZE))
    return mini

def draw_minimap(screen, minimap_surf, cam_x, cam_y, zoom, world_w, world_h):
    if minimap_surf is None: return None
    x = screen.get_width() - MINIMAP_SIZE - MINIMAP_PADDING
    y = screen.get_height() - MINIMAP_SIZE - MINIMAP_PADDING
    bg = pygame.Surface((MINIMAP_SIZE+8, MINIMAP_SIZE+8), pygame.SRCALPHA)
    bg.fill((0,0,0,160))
    screen.blit(bg, (x-4, y-4))
    screen.blit(minimap_surf, (x, y))
    vw = screen.get_width()/zoom
    vh = screen.get_height()/zoom
    rx = int(x + (cam_x / world_w) * MINIMAP_SIZE)
    ry = int(y + (cam_y / world_h) * MINIMAP_SIZE)
    rw = max(2, int((vw / world_w) * MINIMAP_SIZE))
    rh = max(2, int((vh / world_h) * MINIMAP_SIZE))
    pygame.draw.rect(screen, (255,255,255), (rx, ry, rw, rh), 1)
    return pygame.Rect(x, y, MINIMAP_SIZE, MINIMAP_SIZE)

# =========================
# --------- App ----------
# =========================

def main():
    pygame.init()
    font = pygame.font.SysFont(UI_FONT_NAME, 16)
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)

    W, H = WORLD_W, WORLD_H
    seed = random.randint(0, 1_000_000)
    world = build_world(W, H, seed=seed, do_fill=False, tectonics=False)

    palette = 5; mode = "biome"
    show_grid = False; show_rivers = True; show_sea = True
    show_borders = True; show_contours = False; show_coast = False
    show_legend = LEGEND_ON_START
    show_help = False; show_fps = SHOW_FPS_DEFAULT
    iso_view = False; auto_carve = False

    zoom = DISPLAY_SCALE; cam_x, cam_y = 0.0, 0.0

    def refresh_surface():
        nonlocal base_img, base_surf
        base_img = compose_view(world, mode, palette, show_grid, show_rivers,
                                show_sea, show_borders, show_contours, show_coast)
        base_surf = to_surface(base_img)

    base_img = compose_view(world, mode, palette, show_grid, show_rivers,
                            show_sea, show_borders, show_contours, show_coast)
    base_surf = to_surface(base_img)

    pygame.display.set_caption(f"Synthiscape — seed {world.seed}  [{world.W}×{world.H}]")

    running = True; pan_speed = 420
    regenerate_flow_each_frame = False
    seed_stepper = 1
    iso_cache = None; iso_cache_seed = None; iso_cache_height_id = None
    minimap_cache = render_minimap(world, palette=5)

    regen_job: Optional[RegenJob] = None
    show_progress = False

    while running:
        dt = clock.tick(60)/1000.0
        fps = clock.get_fps()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen_w, screen_h = max(300, event.w), max(240, event.h)
                screen = pygame.display.set_mode((screen_w, screen_h), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0: zoom = min(8.0, zoom * 1.1)
                else: zoom = max(0.2, zoom / 1.1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mini_rect = draw_minimap(screen, minimap_cache, cam_x, cam_y, zoom, W, H)
                if mini_rect and mini_rect.collidepoint(event.pos):
                    relx = (event.pos[0] - mini_rect.x) / mini_rect.w
                    rely = (event.pos[1] - mini_rect.y) / mini_rect.h
                    vw = screen.get_width()/zoom
                    vh = screen.get_height()/zoom
                    cam_x = max(0, min(W - vw, relx*W - vw/2))
                    cam_y = max(0, min(H - vh, rely*H - vh/2))
            elif event.type == pygame.KEYDOWN:
                mod = pygame.key.get_mods()
                if event.key == pygame.K_ESCAPE:
                    if regen_job is not None and not regen_job.done:
                        regen_job = None; show_progress = False
                    else:
                        running = False
                elif event.key == pygame.K_h: show_help = not show_help
                elif event.key == pygame.K_k: show_legend = not show_legend
                elif event.key == pygame.K_F3: show_fps = not show_fps
                elif event.key == pygame.K_r:
                    regen_job = RegenJob(W, H, seed=random.randint(0, 1_000_000),
                                         do_fill=world.filled, tectonics=world.tectonics)
                    show_progress = True
                elif event.key == pygame.K_n:
                    regen_job = RegenJob(W, H, seed=(world.seed + seed_stepper) % 1_000_003,
                                         do_fill=world.filled, tectonics=world.tectonics)
                    show_progress = True
                elif event.key == pygame.K_j:
                    regen_job = RegenJob(W, H, seed=world.seed,
                                         do_fill=world.filled, tectonics=not world.tectonics)
                    show_progress = True
                elif event.key == pygame.K_d:
                    regen_job = RegenJob(W, H, seed=world.seed,
                                         do_fill=not world.filled, tectonics=world.tectonics)
                    show_progress = True
                elif event.key == pygame.K_e:
                    world.height = thermal_erosion(world.height, iters=THERMAL_EROSION_ITERS_PER_E, talus=THERMAL_TALUS_ANGLE)
                    world.flowacc, _ = flow_accumulation(world.height, SEA_LEVEL)
                    world.rivers = world.flowacc > RIVER_THRESHOLD
                    ocean = world.height <= SEA_LEVEL
                    coast_dist = distance_transform(ocean, passes=2)
                    coast_influence = np.exp(-coast_dist/12.0)
                    river_influence = np.where(world.rivers, 1.0, 0.0)
                    barrier = np.maximum.accumulate(world.height, axis=1); shadow = normalize01(barrier)
                    world.moisture = clamp01(0.58*world.moisture + 0.20*coast_influence + 0.18*river_influence - 0.18*shadow)
                    world.biome = classify_biomes(world.height, world.temperature, world.moisture, SEA_LEVEL)
                    iso_cache = None; minimap_cache = render_minimap(world, palette=5); refresh_surface()
                elif event.key == pygame.K_v:
                    if mod & pygame.KMOD_SHIFT:
                        auto_carve = not auto_carve
                    else:
                        carve_rivers_tick(world); iso_cache = None; minimap_cache = render_minimap(world, palette=5); refresh_surface()
                elif event.key == pygame.K_f:
                    regenerate_flow_each_frame = not regenerate_flow_each_frame
                elif event.key == pygame.K_m:
                    mode = "moisture"; iso_view = False; refresh_surface()
                elif event.key == pygame.K_t:
                    mode = "temperature"; iso_view = False; refresh_surface()
                elif event.key == pygame.K_b:
                    mode = "biome"; iso_view = False; refresh_surface()
                elif event.key == pygame.K_y:
                    mode = "height"; iso_view = False; refresh_surface()
                elif event.key == pygame.K_g:
                    show_grid = not show_grid; refresh_surface()
                elif event.key == pygame.K_c:
                    show_contours = not show_contours; refresh_surface()
                elif event.key == pygame.K_l:
                    show_coast = not show_coast; refresh_surface()
                elif event.key == pygame.K_p:
                    palette = 1 + (palette % 5); refresh_surface()
                elif event.key == pygame.K_i:
                    iso_view = not iso_view
                elif event.key == pygame.K_s:
                    ts = int(time.time())
                    if iso_view:
                        iso = render_isometric(world, scale=2, z_exaggeration=180)
                        path = f"synthiscape_iso_{world.seed}_{ts}.png"
                        pygame.image.save(iso, path); print(f"Saved {path}")
                    else:
                        path = f"synthiscape_{world.seed}_{mode}_{ts}.png"
                        save_png(path, to_surface(base_img)); print(f"Saved {path}")
                elif event.key == pygame.K_x:
                    ts = int(time.time()); base = f"synthiscape_{world.seed}_{ts}"
                    np.save(f"{base}_height.npy", world.height.astype(np.float32))
                    with open(f"{base}_world.json", "w", encoding="utf-8") as f:
                        json.dump({
                            "seed": int(world.seed),
                            "size": [int(world.W), int(world.H)],
                            "sea_level": float(SEA_LEVEL),
                            "river_threshold": int(RIVER_THRESHOLD),
                            "filled_depressions": bool(world.filled),
                            "tectonics": bool(world.tectonics),
                            "palette": int(palette),
                            "mode": mode,
                        }, f, indent=2)
                    print(f"Exported {base}_height.npy and {base}_world.json")
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    zoom = min(8.0, zoom * 1.1)
                elif event.key == pygame.K_MINUS:
                    zoom = max(0.2, zoom / 1.1)

        # Pan (arrow keys only)
        keys = pygame.key.get_pressed()
        dx = dy = 0.0
        if keys[pygame.K_LEFT]:  dx -= 1
        if keys[pygame.K_RIGHT]: dx += 1
        if keys[pygame.K_UP]:    dy -= 1
        if keys[pygame.K_DOWN]:  dy += 1
        cam_x += dx * pan_speed * dt / zoom
        cam_y += dy * pan_speed * dt / zoom
        screen_w, screen_h = screen.get_width(), screen.get_height()
        cam_x = max(0, min(cam_x, W - screen_w/zoom))
        cam_y = max(0, min(cam_y, H - screen_h/zoom))

        # Optional auto updates
        if regenerate_flow_each_frame:
            world.flowacc, _ = flow_accumulation(world.height, SEA_LEVEL)
            world.rivers = world.flowacc > RIVER_THRESHOLD
            refresh_surface()
        if auto_carve:
            carve_rivers_tick(world); iso_cache = None; minimap_cache = render_minimap(world, palette=5); refresh_surface()

        # Non-blocking regen
        if regen_job and not regen_job.done:
            regen_job.step(FRAME_BUDGET_MS)
            if regen_job.preview_img is not None:
                base_surf = to_surface(regen_job.preview_img)
        if regen_job and regen_job.done:
            world = regen_job.world
            pygame.display.set_caption(f"Synthiscape — seed {world.seed}  [{world.W}×{world.H}]")
            iso_cache = None
            minimap_cache = render_minimap(world, palette=5)
            refresh_surface()
            regen_job = None
            show_progress = False

        # Draw
        screen.fill((12, 12, 18))
        if iso_view:
            height_id = id(world.height)
            if iso_cache is None or iso_cache_seed != world.seed or iso_cache_height_id != height_id:
                iso_cache = render_isometric(world, scale=2, z_exaggeration=180)
                iso_cache_seed = world.seed; iso_cache_height_id = height_id
            rect = iso_cache.get_rect(center=(screen.get_width()//2, screen.get_height()//2))
            screen.blit(iso_cache, rect.topleft)
        else:
            vw, vh = max(1, int(W*zoom)), max(1, int(H*zoom))
            view = pygame.transform.smoothscale(base_surf, (vw, vh)) if zoom <= 1.5 else pygame.transform.scale(base_surf, (vw, vh))
            screen.blit(view, (-int(cam_x*zoom), -int(cam_y*zoom)))

        mini_rect = draw_minimap(screen, minimap_cache, cam_x, cam_y, zoom, W, H)

        if show_legend:
            draw_legend(screen, font, mode, palette,
                        auto_flow=regenerate_flow_each_frame, filled=world.filled,
                        iso=iso_view, auto_carve=auto_carve, tectonics=world.tectonics, world=world,
                        fps=fps if show_fps else 0.0)

        if show_help: draw_help(screen, font)
        if show_progress and regen_job: draw_progress(screen, font, regen_job.progress)

        pygame.display.flip()

    pygame.quit()

# =========================
# --------- Entry ---------
# =========================

if __name__ == "__main__":
    main()
