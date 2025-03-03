# ocr_processing.py

import logging
import re
import cv2
import pytesseract
import numpy as np
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# OCR HELPER FUNCTIONS
# =============================================================================

def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """
    Adjust brightness and contrast of an image.
    alpha: contrast (1.0 means no change)
    beta: brightness (0 means no change)
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def perform_ocr(segment, label):
    """
    Performs OCR on the given segment with multiple preprocessing steps.
    Returns the first non-empty text recognized by Tesseract.
    """
    try:
        if label == "Name":
            # Whitelist includes underscores + space
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ<>#0123456789_ '
        else:
            # For numeric fields
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=.0123456789%'

        def preprocess_original(seg):
            return seg

        def preprocess_grayscale(seg):
            return cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)

        def preprocess_threshold(seg):
            gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh

        def preprocess_blur(seg):
            gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            return blurred

        def preprocess_adaptive_threshold(seg):
            gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 31, 2
            )
            return thresh

        def preprocess_brightness_contrast(seg):
            gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
            return adjust_brightness_contrast(gray, alpha=1.5, beta=30)

        preprocessing_methods = [
            preprocess_original,
            preprocess_grayscale,
            preprocess_threshold,
            preprocess_blur,
            preprocess_adaptive_threshold,
            preprocess_brightness_contrast
        ]

        for preprocess in preprocessing_methods:
            preprocessed_segment = preprocess(segment)
            text = pytesseract.image_to_string(preprocessed_segment, config=custom_config).strip()
            logger.debug(f"OCR raw text for label '{label}': '{text}'")
            if text:
                return text
        return None
    except Exception as e:
        logger.error(f"OCR error for label '{label}': {e}")
        return None

def clean_ocr_result(text, label):
    """
    Cleans the OCR result based on the label.
    For 'Name', tries to preserve digits and letters, removing odd punctuation.
    For numeric fields, removes non-digit or non-percentage characters.
    """
    if not text:
        return None

    logger.debug(f"Original text for label '{label}': '{text}'")

    if label == "Name":
        # Example misreads where you do NOT replace '0' or '1'
        misreads = {
            '2': 'Z',
            '3': 'E',
            '4': 'A',
            '5': 'S',
            '6': 'G',
            '7': 'T',
            '8': 'B',
            '9': 'G',
            '|': 'I',
            '@': 'A',
            '$': 'S',
            '&': 'E',
            '!': 'I',
            '£': 'E',
            '€': 'E',
        }
        for wrong, right in misreads.items():
            text = text.replace(wrong, right)

        # Replace underscores with spaces
        text = text.replace('_', ' ')

        # Optionally remove trailing single uppercase letters if suspicious:
        # E.g. "blacksnowA" -> "blacksnow"
        text = re.sub(r'([A-Za-z0-9])([A-Z])$', r'\1', text)

        # Convert multiple spaces to a single space
        text = re.sub(r'\s+', ' ', text).strip()

        # Keep letters, digits, spaces
        text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)

    else:
        # For numeric fields
        if label == "Accuracy":
            text = re.sub(r'[^\d\.%]', '', text)
        else:
            text = re.sub(r'[^\d]', '', text)

    if not text:
        return None

    return text

def process_for_ocr(image, regions, NUM_PLAYERS=4):
    """
    Extracts text from the image for each player's stats,
    then cleans it with clean_ocr_result().
    Returns a list of dictionaries containing standardized keys:
      ['player_name', 'Kills', 'Shots Fired', 'Shots Hit', 'Deaths', 'Accuracy']
    """
    player_data = []

    for player_index in range(NUM_PLAYERS):
        player_stats = {}
        shots_fired = 0
        shots_hit = 0
        ocr_accuracy = None

        for key in ['Name', 'Kills', 'Shots Fired', 'Shots Hit', 'Deaths', 'Accuracy']:
            label = f"P{player_index + 1} {key}"
            segment = regions.get(label)
            if segment is None:
                logger.error(f"Label {label} not found in regions.")
                player_stats[label] = "0"
                continue

            x1, y1, x2, y2 = segment
            image_segment = image[y1:y2, x1:x2]

            ocr_result = perform_ocr(image_segment, key)
            if not ocr_result:
                logger.error(f"OCR failed for {label}. No text extracted.")
                player_stats[label] = "0"
                continue

            cleaned_result = clean_ocr_result(ocr_result, key)
            if not cleaned_result:
                logger.error(f"Failed to clean OCR result for {label}. Raw: '{ocr_result}'")
                player_stats[label] = "0"
                continue

            # Store or parse data
            if key == "Name":
                player_stats[label] = cleaned_result
            elif key == "Shots Fired":
                try:
                    shots_fired = int(cleaned_result)
                except ValueError:
                    shots_fired = 0
            elif key == "Shots Hit":
                try:
                    shots_hit = int(cleaned_result)
                except ValueError:
                    shots_hit = 0
            elif key == "Accuracy":
                try:
                    ocr_accuracy = float(cleaned_result.rstrip('%'))
                except ValueError:
                    ocr_accuracy = None
            else:
                player_stats[label] = cleaned_result

        # Correct Shots Hit if bigger than Shots Fired
        if shots_hit > shots_fired:
            logger.warning(
                f"Shots Hit exceeds Shots Fired for Player {player_index + 1}. "
                f"Recalculating Shots Hit to {shots_fired}."
            )
            shots_hit = shots_fired

        # Recalc accuracy if needed
        if ocr_accuracy is not None and ocr_accuracy <= 100.0:
            accuracy = ocr_accuracy
        else:
            accuracy = (shots_hit / shots_fired) * 100 if shots_fired > 0 else 0
            accuracy = min(accuracy, 100.0)

        # Store final accuracy
        player_stats[f"P{player_index + 1} Accuracy"] = f"{accuracy:.1f}%"

        # Build final dict with standardized keys
        formatted_player = {}
        for k, v in player_stats.items():
            splitted = k.split(" ", 1)
            if len(splitted) == 2:
                field_key = splitted[1]
                if field_key == "Name":
                    field_key = "player_name"
            else:
                field_key = k
            formatted_player[field_key] = v

        # Shots Fired / Shots Hit / Accuracy numeric
        formatted_player["Shots Fired"] = shots_fired
        formatted_player["Shots Hit"] = shots_hit
        formatted_player["Accuracy"] = f"{accuracy:.1f}%"

        player_data.append(formatted_player)

    return player_data


# =============================================================================
# PARTIAL MATCHING LOGIC (OPTIONAL)
# =============================================================================

def find_best_partial_match(ocr_name: str, registered_names: list[str], threshold: float = 70.0):
    """
    Attempts to find the best partial match for 'ocr_name' within 'registered_names'.
    Uses difflib's SequenceMatcher ratio, plus a simple substring bonus.

    Returns (best_match, best_score) if above threshold, otherwise (None, 0).
    """
    best_match = None
    best_score = 0.0
    ocr_name_lower = ocr_name.lower()

    for db_name in registered_names:
        db_name_lower = db_name.lower()

        # Basic full-string ratio
        ratio_full = SequenceMatcher(None, ocr_name_lower, db_name_lower).ratio() * 100

        # Substring bonus if one is contained in the other
        substring_bonus = 0
        if ocr_name_lower in db_name_lower or db_name_lower in ocr_name_lower:
            substring_bonus = 20  # Tweak as needed

        score = ratio_full + substring_bonus
        if score > best_score:
            best_score = score
            best_match = db_name

    if best_score >= threshold:
        return best_match, best_score
    else:
        return None, 0.0

def match_player_names(ocr_players, registered_users, threshold=70.0):
    """
    Example helper that tries partial matching on each OCR'd 'player_name'
    against a list of 'registered_users'.

    :param ocr_players: list of dicts from process_for_ocr() with 'player_name'
    :param registered_users: list of known player names (e.g., from your DB)
    :param threshold: match score threshold
    :return: same list, but with a 'matched_user' key if found
    """
    for p in ocr_players:
        ocr_name = p.get('player_name')
        if not ocr_name:
            logger.info(f"No OCR name found for {p}. Skipping.")
            continue

        best_match, best_score = find_best_partial_match(ocr_name, registered_users, threshold=threshold)
        if best_match:
            logger.info(f"Best match for '{ocr_name}' is '{best_match}' with a score of {best_score:.1f}.")
            p['matched_user'] = best_match
        else:
            logger.info(f"No good match found for '{ocr_name}'.")
            p['matched_user'] = None

    return ocr_players
