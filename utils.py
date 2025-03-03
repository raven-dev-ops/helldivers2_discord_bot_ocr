# utils.py

import logging

# Configure logging
logger = logging.getLogger(__name__)

async def send_ephemeral(interaction, content):
    """Send an ephemeral message to the user."""
    try:
        await interaction.followup.send(content=content, ephemeral=True)
    except Exception as e:
        logger.error(f"Error sending ephemeral message: {e}")
