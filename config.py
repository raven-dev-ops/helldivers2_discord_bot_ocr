# config.py

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Basic Bot/DB Configuration (unchanged)
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
MONGODB_URI = os.getenv('MONGODB_URI')
DATABASE_NAME = 'GPTHellbot'

# Mongo Collections
REGISTRATION_COLLECTION = 'Alliance'
STATS_COLLECTION = 'User_Stats'
SERVER_LISTING_COLLECTION = 'Server_Listing'

# OCR & Image-Processing Configs
TARGET_WIDTH = int(os.getenv('TARGET_WIDTH', '1920'))
TARGET_HEIGHT = int(os.getenv('TARGET_HEIGHT', '1080'))
PLAYER_OFFSET = int(os.getenv('PLAYER_OFFSET', '460'))
NUM_PLAYERS = int(os.getenv('NUM_PLAYERS', '4'))
ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg')
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# OCR Matching
MATCH_SCORE_THRESHOLD = int(os.getenv('MATCH_SCORE_THRESHOLD', '50'))  # Lower = more tolerant

# ------------------------------------------
# REMOVED server-specific environment usage:
# REQUIRED_ROLE_ID     -> replaced by DB
# MONITOR_CHANNEL_ID   -> replaced by DB
# GPT_NETWORK_ID       -> replaced by DB
# ------------------------------------------
