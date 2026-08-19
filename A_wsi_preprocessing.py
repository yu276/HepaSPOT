import sys
import types

sys.modules["spams"] = types.ModuleType("spams")

import os
import openslide
from openslide import OpenSlideError
import numpy as np
import cv2
from PIL import Image
import torch
import staintools


TARGET_MEAN = np.array([59.19, 44.26, -34.21])
TARGET_STD = np.array([17.35, 14.38, 14.09])


def setup_gpu():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not detected. Using CPU.")


def reinhard_stain_normalization(tile_image):
    try:
        img = np.array(tile_image)
        img = staintools.LuminosityStandardizer.standardize(img)

        normalizer = staintools.ReinhardColorNormalizer()
        normalizer.target_means = TARGET_MEAN
        normalizer.target_stds = TARGET_STD

        normalized = normalizer.transform(img)

        return Image.fromarray(np.uint8(normalized))

    except Exception as e:
        print(f"Error during stain normalization: {e}. Returning original image.")
        return tile_image


def get_foreground_mask(slide, level=0):
    try:
        thumbnail = slide.read_region(
            (0, 0),
            level,
            slide.level_dimensions[level]
        )

        thumbnail = np.array(thumbnail.convert("RGB"))

        mask_white = cv2.inRange(
            thumbnail,
            np.array([200, 200, 200]),
            np.array([255, 255, 255])
        )

        mask_black = cv2.inRange(
            thumbnail,
            np.array([0, 0, 0]),
            np.array([50, 50, 50])
        )

        combined_mask = np.ones_like(mask_white, dtype=np.uint8) * 255
        combined_mask[mask_white > 0] = 0
        combined_mask[mask_black > 0] = 0

        return combined_mask

    except Exception as e:
        print(f"Error generating foreground mask: {e}")
        return None


def get_pixel_size_at_resolution(
    slide,
    target_resolution_um_per_pixel=0.5
):
    try:
        mpp_x = float(
            slide.properties.get("openslide.mpp-x", 0.25)
        )
        mpp_y = float(
            slide.properties.get("openslide.mpp-y", 0.25)
        )

        base_resolution = (mpp_x + mpp_y) / 2.0
        downsample_factor = (
            target_resolution_um_per_pixel / base_resolution
        )

        target_level = 0

        for level, downsample in enumerate(slide.level_downsamples):
            if downsample >= downsample_factor:
                target_level = level
                break

            target_level = level

        print(
            f"Base resolution: {base_resolution:.4f} um/pixel, "
            f"selected level: {target_level}"
        )

        return target_level

    except Exception as e:
        print(f"Error determining target resolution: {e}")
        return 0


def extract_tiles(
    slide,
    output_dir,
    tile_sizes,
    target_resolution_um_per_pixel=0.5
):
    try:
        target_level = get_pixel_size_at_resolution(
            slide,
            target_resolution_um_per_pixel
        )

        level_dimensions = slide.level_dimensions[target_level]

        mask = get_foreground_mask(
            slide,
            target_level
        )

        if mask is None:
            mask = np.ones(
                (level_dimensions[1], level_dimensions[0]),
                dtype=np.uint8
            ) * 255

        for tile_size in tile_sizes:

            tiles_x = int(
                np.ceil(level_dimensions[0] / tile_size[0])
            )

            tiles_y = int(
                np.ceil(level_dimensions[1] / tile_size[1])
            )

            tile_count = 0

            for y in range(tiles_y):
                for x in range(tiles_x):

                    x_pos = x * tile_size[0]
                    y_pos = y * tile_size[1]

                    actual_width = min(
                        tile_size[0],
                        level_dimensions[0] - x_pos
                    )

                    actual_height = min(
                        tile_size[1],
                        level_dimensions[1] - y_pos
                    )

                    if actual_width <= 0 or actual_height <= 0:
                        continue

                    tile_mask = mask[
                        y_pos:y_pos + actual_height,
                        x_pos:x_pos + actual_width
                    ]

                    foreground_ratio = (
                        np.sum(tile_mask > 0)
                        / (actual_width * actual_height)
                    )

                    if foreground_ratio >= 0.25:

                        try:
                            level_0_x = int(
                                x_pos
                                * slide.level_downsamples[target_level]
                            )

                            level_0_y = int(
                                y_pos
                                * slide.level_downsamples[target_level]
                            )

                            tile = slide.read_region(
                                (level_0_x, level_0_y),
                                target_level,
                                (actual_width, actual_height)
                            ).convert("RGB")

                            if (
                                actual_width < tile_size[0]
                                or actual_height < tile_size[1]
                            ):
                                new_tile = Image.new(
                                    "RGB",
                                    tile_size,
                                    (255, 255, 255)
                                )

                                new_tile.paste(tile, (0, 0))
                                tile = new_tile

                            tile = reinhard_stain_normalization(tile)

                            filename = (
                                f"tile_x{x_pos:06d}_"
                                f"y{y_pos:06d}_"
                                f"w{tile_size[0]:04d}_"
                                f"h{tile_size[1]:04d}_"
                                f"level{target_level}_norm.png"
                            )

                            filepath = os.path.join(
                                output_dir,
                                filename
                            )

                            tile.save(filepath, "PNG")

                            tile_count += 1

                        except Exception as e:
                            print(
                                f"Error processing tile "
                                f"({x}, {y}): {e}"
                            )

                            continue

            print(
                f"Saved {tile_count} "
                f"{tile_size[0]}x{tile_size[1]} tiles "
                f"to {output_dir}"
            )

    except Exception as e:
        print(f"Error during tile extraction: {e}")


def main():

    setup_gpu()

    input_dir = "/path/to/wsi_directory"
    output_base_dir = "/path/to/output_directory"

    tile_sizes = [(256, 256)]
    target_resolution = 0.5

    for root, dirs, files in os.walk(input_dir):

        for file in files:

            if file.lower().endswith(".mrxs"):

                print(f"Processing slide: {file}")

                file_path = os.path.join(
                    root,
                    file
                )

                try:
                    slide = openslide.OpenSlide(
                        file_path
                    )

                    file_basename = os.path.splitext(
                        file
                    )[0]

                    output_dir = os.path.join(
                        output_base_dir,
                        file_basename
                    )

                    os.makedirs(
                        output_dir,
                        exist_ok=True
                    )

                    extract_tiles(
                        slide,
                        output_dir,
                        tile_sizes,
                        target_resolution_um_per_pixel=target_resolution
                    )

                    slide.close()

                    print(
                        f"Finished processing: {file}"
                    )

                except OpenSlideError as e:
                    print(
                        f"OpenSlide error: {e}"
                    )

                except Exception as e:
                    print(
                        f"Error processing {file}: {e}"
                    )


if __name__ == "__main__":
    main()