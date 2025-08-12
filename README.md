# Synthiscape — Procedural Terrain Generator

Synthiscape is an advanced, single-file Python project that procedurally generates massive, fully explorable worlds in real time.
Built with Pygame, NumPy, and optional Numba acceleration, it’s designed for both tech enthusiasts and curious players who love watching a world take shape from pure math.
[Play Synthiscape](https://bleu-manatee.itch.io/synthiscape)

# 🌍 Features

Procedural World Generation — Elevation, moisture, temperature, rivers, and biomes generated from noise functions and simulation.

Massive Maps — Supports worlds up to 2400×2400 tiles (larger possible with enough memory).

Multiple Views — Switch between heightmaps, biome maps, moisture, temperature, and more with a key press.

Erosion & Rivers — Watch terrain evolve over time with simulated water flow and thermal erosion.

Help Overlay — Press H anytime for a list of commands.

Zoom & Pan — Arrow keys to move, mouse wheel or +/- to zoom.

Export — Save PNG screenshots or export raw heightmap & metadata for use in your own projects.

# 🎮 Controls

Press H in-game to see all available commands.

Key highlights:

R — Regenerate world

B / Y / M / T — Switch map views (biomes, height, moisture, temperature)

E / V / Shift+V — Apply erosion, carve rivers, or auto-carve

C / L / P / G / K — Toggle overlays (contours, coastline, palette, grid, legend)

Arrow Keys — Pan the camera

Mouse Wheel or +/- — Zoom in/out

S — Save a PNG in the same directory as the exe/python file

X — Export data files

# 💻 Tech Details

Language: Python

Libraries: Pygame, NumPy, optional Numba

Engine: Fully custom procedural generation system in one file

Performance: Optimized rendering — only the visible portion of the map is drawn

# 📦 Who’s it for?

Game developers looking for terrain inspiration

Procedural generation hobbyists

Anyone who enjoys watching a world appear from scratch
