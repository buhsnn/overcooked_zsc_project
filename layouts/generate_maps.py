import os
import json
import random
from PIL import Image

# For validation
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


# ====================== CONFIG ======================
BASE_GRAPHICS = os.path.join(os.path.dirname(__file__), "..", "graphics")
OUTPUT_TXT    = os.path.join(os.path.dirname(__file__), "test_custom_layouts")
OUTPUT_PNG    = os.path.join(os.path.dirname(__file__), "test_custom_layouts_png")
MAP_COUNT     = 200
TILE_SIZE     = 40


# Mapping Overcooked chars → sprite names
TILE_SPRITES = {
    "X": "counter.png",
    " ": "floor.png",
    "O": "onions.png",
    # "T": "tomatoes.png",
    "P": "pot.png",
    "D": "dishes.png",
    "S": "serve.png",
}


# ================================================================
# 1) SPRITESHEET LOADER
# ================================================================
def load_any_spritesheet(json_path, base_folder):
    with open(json_path, "r") as f:
        data = json.load(f)

    sprites = {}

    # TexturePacker format A
    if "frames" in data and isinstance(data["frames"], dict):
        sheet = Image.open(json_path.replace(".json", ".png")).convert("RGBA")
        for name, info in data["frames"].items():
            f = info["frame"]
            x, y, w, h = f["x"], f["y"], f["w"], f["h"]
            sprites[name] = sheet.crop((x, y, x + w, y + h))
        return sprites

    # TexturePacker format B
    if "textures" in data:
        for tex in data["textures"]:
            sheet = Image.open(os.path.join(base_folder, tex["image"])).convert("RGBA")
            for frame_info in tex["frames"]:
                f = frame_info["frame"]
                x, y, w, h = f["x"], f["y"], f["w"], f["h"]
                sprites[frame_info["filename"]] = sheet.crop((x, y, x + w, y + h))
        return sprites

    raise ValueError("Unsupported spritesheet format")


# ================================================================
# 2) PNG RENDERER
# ================================================================
def render_png_from_grid(grid, sprites, out_path):
    H, W = len(grid), len(grid[0])
    canvas = Image.new("RGBA", (W*TILE_SIZE, H*TILE_SIZE))

    for y in range(H):
        for x in range(W):
            char = grid[y][x]
            sprite_name = TILE_SPRITES.get(char)

            if sprite_name not in sprites:
                tile = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (255,0,0,255))
            else:
                tile = sprites[sprite_name].resize((TILE_SIZE, TILE_SIZE), Image.NEAREST)

            canvas.paste(tile, (x*TILE_SIZE, y*TILE_SIZE))

    canvas.save(out_path)


# ================================================================
# 3) VALIDATION WITH OVERCOOKED-AI
# ================================================================
def validate_with_overcooked(grid):
    """
    Injects a player '1' automatically on the first free floor,
    then asks OvercookedGridworld to parse the layout.
    Returns True if valid.
    """
    H = len(grid)
    W = len(grid[0])

    # Convert to mutable grid
    g = [list(row) for row in grid]

    # Find a free spot to place player '1'
    # placed = False
    # for y in range(1, H-1):
    #     for x in range(1, W-1):
    #         if g[y][x] == " ":
    #             g[y][x] = "1"
    #             placed = True
    #             break
    #     if placed:
    #         break

    # if not placed:
    #     return False

    # Try validation
    try:
        OvercookedGridworld.from_grid(g)
        return True
    except:
        return False



# ================================================================
# 4) GENERATOR FOR ORIGINAL SMALL MAPS
# ================================================================
TEMPLATE_SIZES = [
    (4, 5),  # cramped_room 4 rows, width 5
    (5, 8),  # counter_circuit 5 rows, width 8
]

TEMPLATE_DICT = {
    "grid": [],
    "start_bonus_orders": [],
    "start_all_orders" : [
        { "ingredients" : ["onion", "onion", "onion"]}
    ],
    "rew_shaping_params": None
}


def generate_random_layout():
    """
    Creates a brand new layout with:
    - Same dimensions as one official template
    - Random interior walls
    - Stations placed on walls only
    - No players (Overcooked-AI places them automatically)
    """

    H, W = random.choice(TEMPLATE_SIZES)

    # Base empty grid
    grid = [[" " for _ in range(W)] for _ in range(H)]

    # Walls around border
    for x in range(W):
        grid[0][x] = "X"
        grid[H-1][x] = "X"
    for y in range(H):
        grid[y][0] = "X"
        grid[y][W-1] = "X"

    # Random interior walls
    for y in range(1, H-1):
        for x in range(1, W-1):
            if random.random() < 0.15:
                grid[y][x] = "X"

    # Choose pattern
    pattern = random.choice(["corridor", "bottleneck", "none"])

    if pattern == "corridor":
        mid = H//2
        for x in range(1, W-1):
            if random.random() < 0.6:
                grid[mid][x] = " "

    elif pattern == "bottleneck":
        bx = W//2
        for y in range(1, H-1):
            if y not in (1, H-2):
                grid[y][bx] = "X"

    # Collect wall positions for stations
    wall_positions = []
    for x in range(1, W-1):
        wall_positions += [(0, x), (H-1, x)]
    for y in range(1, H-1):
        wall_positions += [(y, 0), (y, W-1)]
    random.shuffle(wall_positions)

    # # Place stations
    # grid[wall_positions.pop()[0]][wall_positions.pop()[1]] = "S"  # serve
    # grid[wall_positions.pop()[0]][wall_positions.pop()[1]] = "P"  # pot
    # grid[wall_positions.pop()[0]][wall_positions.pop()[1]] = "D"  # dishes

    # Ingredients
    for _ in range(random.randint(1, 2)):
        y, x = wall_positions.pop()
        grid[y][x] = "O" #random.choice(["O", "T"])
        
    for _ in range(random.randint(1, 2)):
        y, x = wall_positions.pop()
        grid[y][x] = "S" #random.choice(["O", "T"])
        
    for _ in range(random.randint(1, 2)):
        y, x = wall_positions.pop()
        grid[y][x] = "P" #random.choice(["O", "T"])    
        
    for _ in range(random.randint(1, 2)):
        y, x = wall_positions.pop()
        grid[y][x] = "D" #random.choice(["O", "T"])    
    
    
    for i in range(1, 3) :
        placed = False
        for x in range(1, W-1) :
            for y in range(1, H-1) :
                if placed :
                    break
                if grid[y][x] == " " :
                    grid[y][x] = str(i)
                    placed = True
                    
            

    return ["".join(row) for row in grid]


def save_map_pretty(config: dict, path: str):
    grid = config["grid"]
    other = {k: v for k, v in config.items() if k != "grid"}

    other_pretty = json.dumps(other, indent=4, ensure_ascii=False)

    other_pretty = (
        other_pretty
        .replace("null", "None")
        .replace("true", "True")
        .replace("false", "False")
    )

    result = "{\n"
    result += f'    "grid":  """{grid}""",\n'

    result += other_pretty[1:]  

    with open(path, "w", encoding="utf-8") as f:
        f.write(result)



# ================================================================
# 5) MAIN
# ================================================================
def main():
    print("=== LOADING SPRITES ===")
    sprites = {}

    for fname in ["objects.json", "terrain.json", "soups.json"]:
        try:
            sprites.update(load_any_spritesheet(os.path.join(BASE_GRAPHICS, fname), BASE_GRAPHICS))
            print(f"[OK] Loaded {fname}")
        except Exception as e:
            print(f"[WARN] Could not load {fname}: {e}")

    os.makedirs(OUTPUT_TXT, exist_ok=True)
    os.makedirs(OUTPUT_PNG, exist_ok=True)

    print("\n=== GENERATING RANDOM OVERCOOKED MAPS ===\n")

    generated = 0
    attempts = 0

    while generated < MAP_COUNT:
        attempts += 1
        grid = generate_random_layout()

        # Validate with Overcooked-AI
        if not validate_with_overcooked(grid):
            print(f"[SKIP] Invalid layout (attempt {attempts})")
            continue

        name = f"map_{generated}"
        generated += 1

        # Write txt
        
        TEMPLATE_DICT["grid"] = "\n                ".join(grid)
        # with open(os.path.join(OUTPUT_TXT, f"{name}.layout"), "w") as f:
        #     for row in grid:
        #         f.write(row + "\n")

        save_map_pretty(TEMPLATE_DICT, os.path.join(OUTPUT_TXT, f"{name}.layout"))


        # Write png
        render_png_from_grid(grid, sprites, os.path.join(OUTPUT_PNG, f"{name}.png"))

        print(f"[OK] Saved {name}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
