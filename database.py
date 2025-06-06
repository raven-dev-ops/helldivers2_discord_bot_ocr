import logging
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Dict, Any, Tuple, Optional
from config import (
    MONGODB_URI, DATABASE_NAME,
    REGISTRATION_COLLECTION, STATS_COLLECTION,
    SERVER_LISTING_COLLECTION, MATCH_SCORE_THRESHOLD
)
from rapidfuzz import process, fuzz
from datetime import datetime
import re

logger = logging.getLogger(__name__)

# MongoDB Client and Database
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client[DATABASE_NAME]

registration_collection = db[REGISTRATION_COLLECTION]
stats_collection = db[STATS_COLLECTION]
server_listing_collection = db[SERVER_LISTING_COLLECTION]

from ocr_processing import clean_ocr_result

def normalize_name(name: str) -> str:
    """
    Normalize player names for comparison:
    - Lowercase
    - Remove spaces and underscores
    - Remove trailing numbers (for OCR artifacts like 'Ranjizoro 1')
    - Keep all letters and numbers (so 'pi3' stays 'pi3')
    """
    name = str(name).lower().replace(' ', '').replace('_', '')
    name = re.sub(r'\d+$', '', name)  # Remove only trailing numbers (NOT all numbers)
    return name

################################################
# SERVER LISTING LOOKUPS
################################################

async def get_server_listing_by_id(discord_server_id: int) -> Optional[Dict[str, Any]]:
    """
    Returns the Server_Listing doc for the given guild ID, or None if not found.
    """
    try:
        doc = await server_listing_collection.find_one({"discord_server_id": discord_server_id})
        if doc:
            logger.debug(f"Fetched Server_Listing for guild {discord_server_id}: {doc}")
        return doc
    except Exception as e:
        logger.error(f"Error fetching Server_Listing for ID {discord_server_id}: {e}")
        return None

################################################
# PLAYER REGISTRATION
################################################

async def get_registered_users() -> List[Dict[str, Any]]:
    """
    Fetch all users in the Alliance registration collection.
    """
    try:
        docs = await registration_collection.find(
            {},
            {"player_name": 1, "discord_id": 1, "discord_server_id": 1, "_id": 0}
        ).to_list(length=None)
        logger.info(f"Retrieved {len(docs)} registered users.")
        return docs
    except Exception as e:
        logger.error(f"Error fetching registered users: {e}")
        return []

async def get_registered_user_by_discord_id(discord_id: int) -> Optional[Dict[str, Any]]:
    """
    Find a user in the Alliance collection by their discord_id.
    """
    try:
        doc = await registration_collection.find_one(
            {"discord_id": discord_id},
            {"player_name": 1, "_id": 0}
        )
        if doc:
            logger.info(f"Found registered user for discord_id {discord_id}: {doc.get('player_name','Unknown')}")
        else:
            logger.warning(f"No registered user found for discord_id {discord_id}")
        return doc
    except Exception as e:
        logger.error(f"Error fetching user for discord_id {discord_id}: {e}")
        return None

################################################
# FUZZY MATCHING (IMPROVED)
################################################

def find_best_match(
    ocr_name: str,
    registered_names: List[str],
    threshold: int = 70,
    min_len: int = 3
) -> Tuple[Optional[str], Optional[float]]:
    """
    Fuzzy match `ocr_name` against the list of `registered_names` (normalized).
    Picks best candidate even if ambiguous, never skips.
    """
    if not ocr_name or not registered_names:
        return None, None

    ocr_name_norm = normalize_name(ocr_name.strip())
    logger.debug(f"Attempting to find best match for OCR name '{ocr_name}' (normalized '{ocr_name_norm}')")

    # Exact match pass (normalized)
    for db_name in registered_names:
        if ocr_name_norm == normalize_name(db_name):
            logger.info(f"Exact match: '{ocr_name}' == '{db_name}'")
            return db_name, 100.0

    # For very short names, only allow exact match
    if len(ocr_name_norm) < min_len:
        logger.info(f"Name '{ocr_name}' too short for fuzzy matching.")
        return None, None

    # Fuzzy matching (normalized)
    norm_name_map = {normalize_name(n): n for n in registered_names}
    norm_db_names = list(norm_name_map.keys())

    matches = process.extract(ocr_name_norm, norm_db_names, scorer=fuzz.partial_ratio, limit=3)
    matches = [(norm_name_map[m[0]], m[1]) for m in matches if m[1] >= threshold]
    if not matches:
        logger.info(f"No good fuzzy match found for '{ocr_name}'.")
        return None, None

    matches.sort(key=lambda x: -x[1])
    top_score = matches[0][1]
    top_matches = [m for m in matches if m[1] == top_score]

    if len(top_matches) == 1:
        logger.info(f"Fuzzy match for '{ocr_name}': '{top_matches[0][0]}' with score {top_matches[0][1]}")
        return top_matches[0][0], top_matches[0][1]
    else:
        # If ambiguous, pick the candidate with the closest length, then log and pick first
        best = min(top_matches, key=lambda m: abs(len(m[0]) - len(ocr_name)))
        logger.warning(f"Ambiguous matches for '{ocr_name}', picking '{best[0]}' from: {top_matches}")
        return best[0], best[1]

################################################
# STATS INSERTION
################################################

async def insert_player_data(players_data: List[Dict[str, Any]], submitted_by: str):
    """
    Insert each player's stats data into the stats_collection.
    """
    for player in players_data:
        doc = {
            "player_name": player.get("player_name", "Unknown"),
            "Kills": player.get("Kills", "N/A"),
            "Accuracy": player.get("Accuracy", "N/A"),
            "Shots Fired": player.get("Shots Fired", "N/A"),
            "Shots Hit": player.get("Shots Hit", "N/A"),
            "Deaths": player.get("Deaths", "N/A"),
            "discord_id": player.get("discord_id", None),
            "discord_server_id": player.get("discord_server_id", None),
            "clan_name": player.get("clan_name", "N/A"),
            "submitted_by": submitted_by,
            "submitted_at": datetime.utcnow()
        }
        try:
            await stats_collection.insert_one(doc)
            logger.info(f"Inserted player data for {doc['player_name']}, submitted by {submitted_by}.")
        except Exception as e:
            logger.error(f"Failed to insert player data for {doc['player_name']}: {e}")

################################################
# CLAN NAME LOOKUP
################################################

async def get_clan_name_by_discord_server_id(discord_server_id: Any) -> str:
    """
    Look up 'discord_server_name' from 'Server_Listing' by discord_server_id.
    """
    if not discord_server_id:
        return "N/A"
    try:
        int_server_id = int(discord_server_id)
        doc = await server_listing_collection.find_one({"discord_server_id": int_server_id})
        if doc and "discord_server_name" in doc:
            return doc["discord_server_name"]
        return "N/A"
    except Exception as e:
        logger.error(f"Error fetching clan name for server_id {discord_server_id}: {e}")
        return "N/A"
