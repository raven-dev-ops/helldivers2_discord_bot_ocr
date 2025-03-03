# boundary_drawing.py

import cv2
import logging
import numpy as np
from config import TARGET_WIDTH, TARGET_HEIGHT, PLAYER_OFFSET, NUM_PLAYERS

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

########################################
# "Close Enough" Tolerance in Pixels
########################################
TOLERANCE = 5

##################################################
# KNOWN RESOLUTIONS
##################################################
KNOWN_RESOLUTIONS = {
    (1280, 800): {
        'regions': {
            "Name":        (87, 133, 262, 152),
            "Kills":       (229, 225, 293, 247),
            "Accuracy":    (229, 259, 293, 278),
            "Shots Fired": (229, 291, 293, 311),
            "Shots Hit":   (229, 322, 293, 346),
            "Deaths":      (250, 352, 293, 376),
        },
        # For subsequent players, we shift horizontally by 305
        'offset': 305,
    },
    (1920, 1080): {
        'regions': {
            "Name":        (130, 200, 360, 230),
            "Kills":       (340, 338, 450, 375),
            "Accuracy":    (340, 386, 450, 420),
            "Shots Fired": (340, 435, 450, 470),
            "Shots Hit":   (340, 483, 449, 518),
            "Deaths":      (375, 528, 450, 566),
        },
        'offset': PLAYER_OFFSET,  # from config
    }
}

def is_close_enough(w, h, target_w, target_h, tolerance=TOLERANCE):
    """
    Returns True if (w, h) is within ±tolerance of (target_w, target_h).
    """
    return (abs(w - target_w) <= tolerance) and (abs(h - target_h) <= tolerance)

def adjust_region(region, offset, player_index, player_offset):
    """
    Adjusts region coordinates horizontally only:
      1) x_offset from 'offset' is applied to left/right,
      2) Horizontal offset for subsequent players = player_index * player_offset.
    No vertical shifting is performed.
    """
    left, top, right, bottom = region
    x_off, _ = offset  # Ignore any y_offset

    # Apply initial horizontal offset
    left  += x_off
    right += x_off

    # Apply horizontal offset for subsequent players
    left  += player_index * player_offset
    right += player_index * player_offset

    # Ensure no negative coords
    left   = max(0, left)
    top    = max(0, top)
    right  = max(0, right)
    bottom = max(0, bottom)

    return (left, top, right, bottom)

def define_regions(image_shape=None):
    """
    Picks a base region set if the image is "close enough" to 1280x800 or 1920x1080.
    Otherwise, fallback to 1920x1080 + scaling for the bounding boxes.

    Returns a dict of bounding boxes keyed by:
        "P1 Name", "P1 Kills", ..., "P2 Name", ...
    """
    if not image_shape:
        # Fallback if we don't know shape
        logger.warning("No image shape provided; defaulting to 1920x1080 base.")
        chosen_data = KNOWN_RESOLUTIONS[(1920, 1080)]
        base_width, base_height = (1920, 1080)
        scale_x = scale_y = 1.0
    else:
        h, w = image_shape[:2]
        logger.error(f"Detected actual shape (height={h}, width={w})")

        # Check 1280x800 "close enough"
        if is_close_enough(w, h, 1280, 800):
            chosen_data = KNOWN_RESOLUTIONS[(1280, 800)]
            base_width, base_height = (1280, 800)
            logger.error("USING 1280x800 BOUNDARIES (CLOSE ENOUGH)")
        # Check 1920x1080 "close enough"
        elif is_close_enough(w, h, 1920, 1080):
            chosen_data = KNOWN_RESOLUTIONS[(1920, 1080)]
            base_width, base_height = (1920, 1080)
            logger.error("USING 1920x1080 BOUNDARIES (CLOSE ENOUGH)")
        else:
            # fallback: scale from 1920x1080
            chosen_data = KNOWN_RESOLUTIONS[(1920, 1080)]
            base_width, base_height = (1920, 1080)
            logger.warning(
                f"Image is {w}x{h}, not close to 1280x800 or 1920x1080. "
                "Falling back to 1920x1080 with scaling."
            )

        scale_x = float(w) / base_width
        scale_y = float(h) / base_height

    base_regions = chosen_data['regions']
    chosen_player_offset = chosen_data['offset']

    regions = {}
    for player_index in range(NUM_PLAYERS):
        for key, (base_left, base_top, base_right, base_bottom) in base_regions.items():
            # Adjust horizontally only
            region_no_scale = adjust_region(
                (base_left, base_top, base_right, base_bottom),
                (0, 0),
                player_index,
                chosen_player_offset
            )
            # Then scale to actual image size
            left   = int(region_no_scale[0] * scale_x)
            top    = int(region_no_scale[1] * scale_y)
            right  = int(region_no_scale[2] * scale_x)
            bottom = int(region_no_scale[3] * scale_y)

            label = f"P{player_index + 1} {key}"
            regions[label] = (left, top, right, bottom)
            logger.debug(f"{label} -> {regions[label]}")

    logger.debug(f"Final regions dict: {regions}")
    return regions

def resize_image_with_padding(image, target_size):
    """
    [Optional Function]
    Resize an image while maintaining aspect ratio, adding padding if necessary.
    Use this *after* region detection/ OCR if you want a standard size for display.
    """
    if len(image.shape) < 3 or image.shape[2] not in [3, 4]:
        raise ValueError("Invalid image format; must have 3 or 4 channels.")

    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image while preserving aspect ratio
    resized = cv2.resize(image, (new_w, new_h))
    # Create a new "blank" (white) image with target size
    padded = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return padded

def draw_boundaries(image, regions):
    """
    Draw bounding boxes on 'image' for debugging/verification.
    """
    for label, (x1, y1, x2, y2) in regions.items():
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image
