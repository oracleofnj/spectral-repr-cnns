"""Create scaled and shifted images for exploration."""
from PIL import Image
import os
import numpy as np
__DEFAULT_PATH = '../Images'


def open_image(filename, path=__DEFAULT_PATH):
    """Open an image file with Pillow."""
    if filename is None:
        raise ValueError('Filename is required.')
    full_path = os.path.join(path, filename)
    im = Image.open(full_path).convert('RGBA')
    return im


def save_derived_image(im, filename=None, path=__DEFAULT_PATH):
    """Save a pillow image as a PNG."""
    if filename is None:
        filename = 'Derived/{0:08x}.png'.format(np.random.randint(2 ** 31))
    full_path = os.path.join(path, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    im.save(full_path, 'PNG')


def downscale_image(orig_image, max_width, max_height):
    """Rescale an image to a smaller image."""
    orig_width = orig_image.width
    orig_height = orig_image.height

    # Compute how much to multiply the existing dimensions by
    width_multo = max_width / orig_width
    height_multo = max_height / orig_height
    multo = min(height_multo, width_multo)

    # Create the new image
    new_width = int(orig_width * multo)
    new_height = int(orig_height * multo)
    new_image = orig_image.resize(
        (new_width, new_height),
        resample=Image.LANCZOS
    )

    return new_image


def add_to_background(
    foreground_image,
    destination_left,
    destination_top,
    destination_max_width,
    destination_max_height,
    background_image=None,
    background_width=128,
    background_height=128,
):
    """Add an image to a background image.

    If background_image is None, the function will create a solid
    grey background image of dimensions (background_width, background_height)
    and paste the image onto that.
    """
    if background_image is None:
        new_background_image = Image.new(
            'RGBA',
            (background_width, background_height),
            '#7f7f7f'
        )
    else:
        new_background_image = background_image.copy()

    rescaled_foreground_image = downscale_image(
        foreground_image,
        destination_max_width,
        destination_max_height,
    )
    new_background_image.paste(
        rescaled_foreground_image,
        box=(destination_left, destination_top),
        mask=rescaled_foreground_image
    )

    return new_background_image


def make_random_size(destination_width=128, destination_height=128):
    """Generate random coordinates where a scaled image can be placed."""
    scale = np.random.randint(
        16,
        1 + min(destination_width, destination_height)
        )

    left = np.random.randint(0, 1 + destination_width - scale)
    top = np.random.randint(0, 1 + destination_height - scale)
    width = scale
    height = scale

    return left, top, width, height
