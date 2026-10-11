"""Image preprocessing utilities."""

import numpy as np
from PIL import Image


def _get_quad(aspect_ratio: float) -> np.ndarray:
    """Return normalized source coordinates for centered padding.

    Expect the input aspect ratio divided by the output aspect ratio.
    """
    if aspect_ratio > 1.0:
        quad = np.array(
            [
                [0.0, 0.5 - 0.5 * aspect_ratio],
                [0.0, 0.5 + 0.5 * aspect_ratio],
                [1.0, 0.5 + 0.5 * aspect_ratio],
                [1.0, 0.5 - 0.5 * aspect_ratio],
            ]
        )
    elif aspect_ratio < 1.0:
        quad = np.array(
            [
                [0.5 - 0.5 / aspect_ratio, 0.0],
                [0.5 - 0.5 / aspect_ratio, 1.0],
                [0.5 + 0.5 / aspect_ratio, 1.0],
                [0.5 + 0.5 / aspect_ratio, 0.0],
            ]
        )
    else:
        quad = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
            ]
        )
    return quad


def _transform_image(
    image: Image.Image,
    size: tuple[int, int] | list[int],
    quad: np.ndarray,
) -> Image.Image:
    """Transform an image using normalized source corners.

    Expect size as (width, height) and quad as a (4, 2) array.
    """
    new_quad = np.zeros_like(quad)
    new_quad[:, 0] = quad[:, 0] * image.size[0]
    new_quad[:, 1] = quad[:, 1] * image.size[1]
    new_quad = (
        new_quad[0][0], new_quad[0][1],
        new_quad[1][0], new_quad[1][1],
        new_quad[2][0], new_quad[2][1],
        new_quad[3][0], new_quad[3][1],
    )
    return image.transform(size, Image.QUAD, new_quad)


def resize_image(
    image: np.ndarray,
    target_size: tuple[int, int] | list[int],
) -> np.ndarray:
    """Resize an image with centered padding, preserving its aspect ratio.

    Args:
        image: Input image array.
        target_size: Output size as (width, height).

    Returns:
        Resized image as a uint8 array.
    """
    pil_image = Image.fromarray(image)
    in_ar = pil_image.width / pil_image.height
    out_ar = target_size[0] / target_size[1]
    rel_ar = in_ar / out_ar

    quad = _get_quad(aspect_ratio=rel_ar)
    return np.array(
        _transform_image(pil_image, target_size, quad),
        dtype=np.uint8,
    )