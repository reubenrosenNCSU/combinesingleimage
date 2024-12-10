import os
import re
import numpy as np
from PIL import Image

def merge_png_tiles(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all tile files in the input directory
    tile_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    # Extract unique image base names
    # Updated regex to handle extra segments after <row>_<column>
    pattern = re.compile(r"(.+?)_tile_(\d+)_(\d+).*\.png")
    images_info = {}

    for tile_file in tile_files:
        match = pattern.match(tile_file)
        if match:
            base_name, row, col = match.groups()
            row, col = int(row), int(col)

            if base_name not in images_info:
                images_info[base_name] = []
            images_info[base_name].append((row, col, tile_file))

    merged_images = []

    # Merge tiles for each unique base name
    for base_name, tiles in images_info.items():
        # Determine the final image dimensions
        rows = sorted(set(row for row, _, _ in tiles))
        cols = sorted(set(col for _, col, _ in tiles))
        tile_height = rows[1] - rows[0] if len(rows) > 1 else None
        tile_width = cols[1] - cols[0] if len(cols) > 1 else None
        full_height = rows[-1] + tile_height if tile_height else rows[0]
        full_width = cols[-1] + tile_width if tile_width else cols[0]

        # Create a blank image to hold the full image
        first_tile_path = os.path.join(input_dir, tiles[0][2])
        first_tile = Image.open(first_tile_path)
        mode = first_tile.mode  # e.g., "RGB" or "RGBA"
        blank_color = (0, 0, 0, 0) if mode == "RGBA" else (0, 0, 0)
        full_image = Image.new(mode, (full_width, full_height), blank_color)

        # Place each tile in the correct position
        for row, col, tile_file in tiles:
            tile_path = os.path.join(input_dir, tile_file)
            tile = Image.open(tile_path)
            full_image.paste(tile, (col, row))

        # Save the merged image
        output_path = os.path.join(output_dir, f"{base_name}_merged.png")
        full_image.save(output_path)
        merged_images.append(output_path)

    return merged_images

# Example usage
input_dir = '/home/greenbaum-gpu/Reuben/keras-retinanet/output'  # Replace with your actual path
output_dir = '/home/greenbaum-gpu/Reuben/keras-retinanet/finaloutput'  # Replace with your actual path
merged_images = merge_png_tiles(input_dir, output_dir)
print(f"Merged {len(merged_images)} images.")
