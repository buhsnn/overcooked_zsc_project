import os
import json
import random
from PIL import Image

# ====================== CONFIG ======================
BASE_GRAPHICS = os.path.join(os.path.dirname(__file__), "..", "graphics")
OUTPUT_TXT    = os.path.join(os.path.dirname(__file__), "custom_layouts")
OUTPUT_PNG    = os.path.join(os.path.dirname(__file__), "custom_layouts_png")
MAP_COUNT     = 10
TILE_SIZE     = 40

# Mapping Overcooked chars → sprite names
TILE_SPRITES = {
    "X": "counter.png",
    " ": "floor.png",
    "O": "onions.png",
    "T": "tomatoes.png",
    "P": "pot.png",
    "D": "dishes.png",
    "S": "serve.png",
}

# ================================================================
# 1) SPRITESHEET LOADER (TexturePacker formats A et B)
# ================================================================
def load_any_spritesheet(json_path, base_folder):
    with open(json_path, "r") as f:
        data = json.load(f)

    sprites = {}

    # --- Format A: { "frames": { name: { frame: {x,y,w,h} } } } ---
    if "frames" in data and isinstance(data["frames"], dict):
        sheet = Image.open(json_path.replace(".json", ".png")).convert("RGBA")
        for name, info in data["frames"].items():
            frame = info["frame"]
            x, y, w, h = frame["x"], frame["y"], frame["w"], frame["h"]
            sprites[name] = sheet.crop((x, y, x + w, y + h))
        return sprites

    # --- Format B: { "textures": [ { "image":..., "frames":[...] } ] } ---
    if "textures" in data:
        for tex in data["textures"]:
            sheet = Image.open(os.path.join(base_folder, tex["image"])).convert("RGBA")
            for f in tex["frames"]:
                frame = f["frame"]
                x, y, w, h = frame["x"], frame["y"], frame["w"], frame["h"]
                sprites[f["filename"]] = sheet.crop((x, y, x + w, y + h))
        return sprites

    raise ValueError("Unsupported spritesheet format: " + json_path)


# ================================================================
# 2) PNG RENDERER
# ================================================================
def render_png_from_grid(grid, sprites, out_path):
    H, W = len(grid), len(grid[0])
    canvas = Image.new("RGBA", (W * TILE_SIZE, H * TILE_SIZE))

    for y in range(H):
        for x in range(W):
            char = grid[y][x]
            name = TILE_SPRITES.get(char)
            if not name or name not in sprites:
                
                tile = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (255, 0, 0, 255))
            else:
                tile = sprites[name].resize((TILE_SIZE, TILE_SIZE), Image.NEAREST)
            canvas.paste(tile, (x * TILE_SIZE, y * TILE_SIZE))

    canvas.save(out_path)


# ================================================================
# 3) Official Template + RANDOMISATION
# ================================================================


TEMPLATES = {
    "cramped_room": [
        "XXPXX",
        "O   O",
        "X   X",
        "XDXSX",
    ],
    "counter_circuit_o_1order": [
        "XXXPPXXX",
        "X      X",
        "D XXXX S",
        "X      X",
        "XXXOOXXX",
    ],
    "forced_coordination": [
        "XXXPX",
        "O X P",
        "O X X",
        "D X X",
        "XXXSX",
    ],
    "coordination_ring": [
        "XXXPX",
        "X   P",
        "D X X",
        "O   X",
        "XOSXX",
    ],
}


def _flip_h(grid):
    """Horizontal flip"""
    return [row[::-1] for row in grid]


def _flip_v(grid):
    """vertical flip"""
    return grid[::-1]


def _transpose(grid):
    """Transpose"""
    return ["".join(col) for col in zip(*grid)]


def generate_stylish_overcooked_map():
    """
    Generates a NEW original Overcooked-style layout with the SAME dimensions
    as one of the official templates.
    
    Steps:
    1) Pick a template size
    2) Fill interior with random walls (controlled)
    3) Ensure space is connected
    4) Place stations (S, P, O/T, D) on walls or corners
    """

    # Pick one template shape (just its dimensions)
    template_name, template = random.choice(list(TEMPLATES.items()))
    H = len(template)
    W = len(template[0])

    # --- Step 1: empty grid ---
    grid = [[" " for _ in range(W)] for _ in range(H)]

    # Border walls
    for x in range(W):
        grid[0][x] = grid[H-1][x] = "X"
    for y in range(H):
        grid[y][0] = grid[y][W-1] = "X"

    # --- Step 2: add interior walls probabilistically ---
    for y in range(1, H-1):
        for x in range(1, W-1):
            if random.random() < 0.18:   # small probability
                grid[y][x] = "X"

    # --- Step 3: ensure small patterns (corridor, island, bottleneck) ---
    pattern = random.choice(["corridor", "bottleneck", "island", "none"])

    if pattern == "corridor":
        mid = H//2
        for x in range(1, W-1):
            if random.random() < 0.5:
                grid[mid][x] = " "

    elif pattern == "bottleneck":
        bx = W//2
        for y in range(1, H-1):
            if y not in (1, H-2):
                grid[y][bx] = "X"

    elif pattern == "island":
        if H >= 5 and W >= 5:
            for y in range(2, H-2):
                for x in range(2, W-2):
                    if random.random() < 0.25:
                        grid[y][x] = "X"

    # --- Step 4: place stations along walls ---
    wall_slots = []
    for x in range(1, W-1):
        wall_slots += [(0, x), (H-1, x)]
    for y in range(1, H-1):
        wall_slots += [(y, 0), (y, W-1)]

    random.shuffle(wall_slots)

    # Serve station
    sy, sx = wall_slots.pop()
    grid[sy][sx] = "S"

    # Pot
    py, px = wall_slots.pop()
    grid[py][px] = "P"

    # Dish
    dy, dx = wall_slots.pop()
    grid[dy][dx] = "D"

    # 1–2 ingredients
    for _ in range(random.randint(1, 2)):
        y, x = wall_slots.pop()
        grid[y][x] = random.choice(["O", "T"])

    # Convert to string rows
    return ["".join(row) for row in grid]


# ================================================================
# 4) MAIN
# ================================================================
def main():
    print("=== LOADING SPRITES ===")
    sprites = {}

    for fname in ["objects.json", "terrain.json", "soups.json"]:
        try:
            sprites.update(
                load_any_spritesheet(
                    os.path.join(BASE_GRAPHICS, fname),
                    BASE_GRAPHICS,
                )
            )
            print(f"[OK] Loaded {fname}")
        except Exception as e:
            print(f"[WARN] Could not load {fname}: {e}")

    os.makedirs(OUTPUT_TXT, exist_ok=True)
    os.makedirs(OUTPUT_PNG, exist_ok=True)

    print("\n=== GENERATING RANDOM MAPS (remix of official template) ===\n")

    for i in range(MAP_COUNT):
        name = f"map_{i}"

        grid = generate_stylish_overcooked_map()

        # TXT
        with open(os.path.join(OUTPUT_TXT, f"{name}.layout"), "w") as f:
            for row in grid:
                f.write(row + "\n")

        # PNG
        render_png_from_grid(grid, sprites, os.path.join(OUTPUT_PNG, f"{name}.png"))

        print(f"[OK] Generated {name}  ({len(grid[0])}x{len(grid)})")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
