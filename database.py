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

logger = logging.getLogger(__name__)

# MongoDB Client and Database
mongo_client = AsyncIOMotorClient(MONGODB_URI)
db = mongo_client[DATABASE_NAME]

registration_collection = db[REGISTRATION_COLLECTION]
stats_collection = db[STATS_COLLECTION]
server_listing_collection = db[SERVER_LISTING_COLLECTION]

# We'll import the same 'clean_ocr_result' used by OCR to unify name cleaning
from ocr_processing import clean_ocr_result

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
# FUZZY MATCHING
################################################

def find_best_match(
    ocr_name: str,
    registered_names: List[str],
    threshold: int = 50,
    min_len: int = 4
) -> Tuple[Optional[str], Optional[float]]:
    """
    Fuzzy match `ocr_name` against the list of `registered_names`.
    For names shorter than min_len, only allow case-insensitive exact match.
    Returns (best_match, match_score).
    """
    if not ocr_name or not registered_names:
        return None, None

    ocr_name = ocr_name.strip()
    logger.debug(f"Attempting to find best match for OCR name '{ocr_name}'")

    # If too short, require exact (case-insensitive) match only
    if len(ocr_name) < min_len:
        for db_name in registered_names:
            if ocr_name.lower() == db_name.lower():
                logger.info(f"Short name: exact match '{ocr_name}' == '{db_name}'")
                return db_name, 100.0
        logger.info(f"No exact match for short name '{ocr_name}'.")
        return None, None

    # Standard fuzzy matching for longer names
    match = process.extractOne(
        ocr_name,
        registered_names,
        scorer=fuzz.partial_ratio,
        score_cutoff=threshold
    )
    if match:
        best_match, match_score = match[0], match[1]
        logger.info(f"Best match for '{ocr_name}' is '{best_match}' with a score of {match_score}.")
        return best_match, match_score
    else:
        logger.info(f"No good match found for '{ocr_name}'.")
        return None, None

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
