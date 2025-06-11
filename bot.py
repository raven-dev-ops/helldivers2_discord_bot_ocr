import discord
from discord import app_commands
from discord.ext import commands
import logging
import asyncio
from PIL import Image
from io import BytesIO
import numpy as np
import re
import traceback

# Database helpers
from database import (
    get_registered_users,
    insert_player_data,
    find_best_match,
    get_registered_user_by_discord_id,
    get_clan_name_by_discord_server_id,
    get_server_listing_by_id
)
from config import (
    DISCORD_TOKEN, ALLOWED_EXTENSIONS, MATCH_SCORE_THRESHOLD,
    TARGET_WIDTH, TARGET_HEIGHT
)
from ocr_processing import process_for_ocr, clean_ocr_result
from boundary_drawing import (
    define_regions,
    resize_image_with_padding
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=intents)
tree = bot.tree

########################################
# HELPER FUNCTIONS
########################################

def prevent_discord_formatting(name: str) -> str:
    """Escape any channel/role mention markup."""
    if not name:
        return ""
    return name.replace('<#', '<\u200B#').replace('<@&', '<\u200B@&')

def highlight_zero_values(player: dict) -> list:
    """Check if certain stats are zero or N/A, and return those fields."""
    fields = ["Kills", "Accuracy", "Shots Fired", "Shots Hit"]
    zero_fields = []
    for field in fields:
        val = str(player.get(field, 'N/A'))
        if val in ['0', '0.0', 'None', 'N/A']:
            zero_fields.append(field)
    return zero_fields

def validate_stat(field_name: str, raw_value: str):
    """Parses a stat value (int or float or 'N/A')."""
    raw_value = raw_value.strip()
    if raw_value.upper() == 'N/A':
        return 'N/A'
    if field_name in ['Kills', 'Shots Fired', 'Shots Hit', 'Deaths']:
        return int(raw_value)
    elif field_name == 'Accuracy':
        numeric_part = raw_value.replace('%', '')
        parsed = float(numeric_part)
        return f"{parsed:.1f}%"
    else:
        return raw_value

def clean_for_match(name):
    """
    Lowercases, removes spaces, punctuation, and common prefixes for best matching.
    """
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', '', name)  # Remove all non-alphanumeric
    name = re.sub(r'^(mr|ms|mrs|dr)', '', name)  # Remove titles at start
    return name

########################################
# EMBEDS
########################################

def build_single_embed(players_data: list, submitter_player_name: str) -> discord.Embed:
    """
    Builds ONE embed that shows the stats of ALL players in a single embed.
    """
    embed = discord.Embed(
        title="GPT FLEET STAT EXTRACTION",
        description=f"Submitted by: {submitter_player_name}",
        color=discord.Color.blue()
    )
    for index, player in enumerate(players_data, start=1):
        player_name = prevent_discord_formatting(player.get('player_name', 'Unknown'))
        clan_name = player.get('clan_name', 'N/A')
        kills = str(player.get('Kills', 'N/A'))
        deaths = str(player.get('Deaths', 'N/A'))
        shots_fired = str(player.get('Shots Fired', 'N/A'))
        shots_hit = str(player.get('Shots Hit', 'N/A'))
        accuracy = str(player.get('Accuracy', 'N/A'))
        melee_kills = str(player.get('Melee Kills', 'N/A'))
        player_info = (
            f"**Name**: {player_name}\n"
            f"**Clan**: {clan_name}\n"
            f"**Kills**: {kills}\n"
            f"**Deaths**: {deaths}\n"
            f"**Shots Fired**: {shots_fired}\n"
            f"**Shots Hit**: {shots_hit}\n"
            f"**Accuracy**: {accuracy}\n"
            f"**Melee Kills**: {melee_kills}\n")
        zero_vals = highlight_zero_values(player)
        if zero_vals:
            player_info += f"\n**Needs Confirmation**: {', '.join(zero_vals)}"

        embed.add_field(name=f"Player {index}", value=player_info, inline=False)

    return embed

def build_monitor_embed(players_data: list, submitter_name: str) -> discord.Embed:
    """
    Builds an embed for the final #monitor channel.
    """
    embed = discord.Embed(
        title="Saved Results",
        description=f"Submitted by: {submitter_name}",
        color=discord.Color.green()
    )
    for index, player in enumerate(players_data, start=1):
        player_name = prevent_discord_formatting(player.get('player_name', 'Unknown'))
        clan_name = player.get('clan_name', 'N/A')
        kills = str(player.get('Kills', 'N/A'))
        deaths = str(player.get('Deaths', 'N/A'))
        shots_fired = str(player.get('Shots Fired', 'N/A'))
        shots_hit = str(player.get('Shots Hit', 'N/A'))
        accuracy = str(player.get('Accuracy', 'N/A'))
        melee_kills = str(player.get('Melee Kills', 'N/A'))
        final_info = (
            f"**Name**: {player_name}\n"
            f"**Clan**: {clan_name}\n"
            f"**Kills**: {kills}\n"
            f"**Accuracy**: {accuracy}\n"
            f"**Shots Fired**: {shots_fired}\n"
            f"**Shots Hit**: {shots_hit}\n"
            f"**Deaths**: {deaths}\n"
            f"**Melee Kills**: {melee_kills}\n")
        embed.add_field(name=f"Player {index}", value=final_info, inline=False)

    return embed

########################################
# SHARED DATA & VIEW (CONFIRM/EDIT)
########################################

class SharedData:
    def __init__(
        self,
        players_data,
        submitter_player_name,
        registered_users,
        monitor_channel_id,
        screenshot_bytes=None,
        screenshot_filename=None
    ):

        self.players_data = players_data
        self.submitter_player_name = submitter_player_name
        self.registered_users = registered_users
        self.monitor_channel_id = monitor_channel_id

        self.selected_player_index = None
        self.selected_field = None
        self.message = None
        self.view = None
        self.screenshot_bytes = screenshot_bytes
        self.screenshot_filename = screenshot_filename

class ConfirmationView(discord.ui.View):
    def __init__(self, shared_data):
        super().__init__(timeout=None)
        self.shared_data = shared_data

    @discord.ui.button(label="YES", style=discord.ButtonStyle.green)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        """
        Confirm button -> Saves stats to DB, posts final embed in #monitor channel.
        """
        try:
            await interaction.response.defer(ephemeral=True)
            
            # If any zero or 'Needs Confirmation' fields remain, warn
            if any(highlight_zero_values(p) for p in self.shared_data.players_data):
                await interaction.followup.send(
                    "Some values are zero or missing. Please EDIT them before confirming.",
                    ephemeral=True
                )
                return 

            # Insert to DB
            from database import insert_player_data
            await insert_player_data(self.shared_data.players_data, self.shared_data.submitter_player_name)

            # Post to #monitor
            monitor_embed = build_monitor_embed( 
                self.shared_data.players_data,
 self.shared_data.submitter_player_name
            )
            file_to_send = None
            if self.shared_data.screenshot_bytes and self.shared_data.screenshot_filename:
 file_to_send = discord.File(BytesIO(self.shared_data.screenshot_bytes), filename=self.shared_data.screenshot_filename)


            monitor_channel = bot.get_channel(self.shared_data.monitor_channel_id)
            if monitor_channel:
                await monitor_channel.send(embed=monitor_embed)
            else:
                logger.error("Monitor channel not found or invalid ID in DB.")

            # Confirm success
            await self.shared_data.message.edit(
                content="Data confirmed and saved successfully!",
                embeds=[],
                view=None
            )
        except Exception as e:
            logger.error(f"Error in YES button callback: {e}")
            await interaction.followup.send("Error while confirming data.", ephemeral=True)

    @discord.ui.button(label="EDIT", style=discord.ButtonStyle.primary)
    async def edit(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.edit_player_selection(interaction)

    async def edit_player_selection(self, interaction: discord.Interaction):
        """
        Presents a select menu of players to edit stats for.
        """
        try:
            options = []
            for i, player in enumerate(self.shared_data.players_data):
                p_name = player.get('player_name', 'Unknown') or "Unknown"
                options.append(
                    discord.SelectOption(
                        label=f"Player {i + 1}",
                        description=p_name,
                        value=str(i)
                    )
                )
            player_select = PlayerSelect(options, self.shared_data)
            view = discord.ui.View()
            view.add_item(player_select)

            await interaction.response.edit_message(
                content="Choose a player to edit:",
                embeds=[],
                view=view
            )
        except Exception as e:
            logger.error(f"Error in edit_player_selection: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message("An error occurred while editing.", ephemeral=True)

class PlayerSelect(discord.ui.Select):
    def __init__(self, options, shared_data):
        super().__init__(placeholder="Select a player to edit", options=options)
        self.shared_data = shared_data

    async def callback(self, interaction: discord.Interaction):
        try:
            self.shared_data.selected_player_index = int(self.values[0])
            fields = ['player_name', 'Kills', 'Accuracy', 'Shots Fired', 'Shots Hit', 'Deaths', 'Melee Kills']
            field_options = [discord.SelectOption(label=f) for f in fields]

            field_select = FieldSelect(field_options, self.shared_data)
            view = discord.ui.View()
            view.add_item(field_select)

            await interaction.response.edit_message(
                content=(
                    f"Player {self.shared_data.selected_player_index + 1} selected. "
                    "Now select the field you want to edit:"
                ),
                embeds=[],
                view=view
            )
        except Exception as e:
            logger.error(f"Error in PlayerSelect callback: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message("An error occurred. Please try again.", ephemeral=True)

class FieldSelect(discord.ui.Select):
    def __init__(self, options, shared_data):
        super().__init__(placeholder="Select a field to edit", options=options)
        self.shared_data = shared_data

    async def callback(self, interaction: discord.Interaction):
        try:
            selected_field = self.values[0]
            self.shared_data.selected_field = selected_field

            await interaction.response.edit_message(
                content=(
                    f"Enter the new value for {selected_field} "
                    f"(Player {self.shared_data.selected_player_index + 1}):"
                ),
                embeds=[],
                view=None
            )

            def check(m: discord.Message):
                return m.author == interaction.user and m.channel == interaction.channel

            try:
                msg = await bot.wait_for('message', check=check, timeout=60.0)
                await msg.delete()
                new_value_str = msg.content.strip()

                try:
                    new_value = validate_stat(selected_field, new_value_str)
                except ValueError:
                    await interaction.followup.send(
                        f"Invalid input for {selected_field}. Must be numeric or 'N/A' or like '75.3%'.",
                        ephemeral=True
                    )
                    return

                player = self.shared_data.players_data[self.shared_data.selected_player_index]

                if selected_field == 'player_name':
                    from database import get_registered_users, get_clan_name_by_discord_server_id
                    cleaned_ocr_name = clean_ocr_result(new_value_str, 'Name')
                    if not cleaned_ocr_name:
                        player['player_name'] = None
                        player['discord_id'] = None
                        player['discord_server_id'] = None
                        player['clan_name'] = "N/A"
                    else:
                        registered_users = await get_registered_users()
                        db_names = [u["player_name"] for u in registered_users]
                        ocr_name_clean = clean_for_match(cleaned_ocr_name)
                        db_names_clean = [clean_for_match(n) for n in db_names]
                        from database import find_best_match
                        best_match_cleaned, match_score = find_best_match(
                            ocr_name_clean,
                            db_names_clean,
                            threshold=MATCH_SCORE_THRESHOLD
                        )
                        if best_match_cleaned and match_score >= MATCH_SCORE_THRESHOLD:
                            idx = db_names_clean.index(best_match_cleaned)
                            matched_user = registered_users[idx]
                            player['player_name'] = matched_user["player_name"]
                            player['discord_id'] = matched_user.get("discord_id")
                            player['discord_server_id'] = matched_user.get("discord_server_id")
                            if matched_user.get("discord_server_id"):
                                clan_name = await get_clan_name_by_discord_server_id(matched_user["discord_server_id"])
                                player['clan_name'] = clan_name
                            else:
                                player['clan_name'] = "N/A"
                        else:
                            player['player_name'] = None
                            player['discord_id'] = None
                            player['discord_server_id'] = None
                            player['clan_name'] = "N/A"
                else:
                    player[selected_field] = new_value

                # Rebuild single embed
                updated_embed = build_single_embed(
                    self.shared_data.players_data,
                    self.shared_data.submitter_player_name
                )

                await self.shared_data.message.edit(
                    content="**Updated Data:** Please confirm the updated data.",
                    embeds=[updated_embed],
                    view=ConfirmationView(self.shared_data)
                )

            except asyncio.TimeoutError:
                await interaction.followup.send("You took too long to respond. Please try again.", ephemeral=True)
            except Exception as e:
                logger.error(f"Error during data input: {e}")
                await interaction.followup.send("Something went wrong. Please try again.", ephemeral=True)
        except Exception as e:
            logger.error(f"Error in FieldSelect callback: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message("An error occurred. Please try again.", ephemeral=True)

########################################
# /EXTRACT COMMAND
########################################

@tree.command(name="extract", description="Submit mission stats for official report!")
async def extract(interaction: discord.Interaction, image: discord.Attachment):
    """
    This slash command is now multi-guild aware:
      1) Looks up the guild’s 'Server_Listing' by guild_id.
      2) Checks if the user has 'gpt_stat_access_role_id' from DB.
      3) Runs OCR on the provided image.
      4) Lets user confirm or edit stats before saving to DB.
      5) Posts final embed to the 'monitor_channel_id' from DB.
    """
    if not interaction.guild_id:
        await interaction.response.send_message("This command cannot be used in DMs.", ephemeral=True)
        return

    # Fetch the server listing for this guild
    server_data = await get_server_listing_by_id(interaction.guild_id)
    if not server_data:
        await interaction.response.send_message(
            "Server is not configured. Contact an admin or use the GuildManagementCog to set it up.",
            ephemeral=True
        )
        return

    gpt_stat_access_role_id = server_data.get("gpt_stat_access_role_id")
    monitor_channel_id = server_data.get("monitor_channel_id")
    gpt_channel_id = server_data.get("gpt_channel_id")

    if not gpt_stat_access_role_id or not monitor_channel_id:
        await interaction.response.send_message(
            "Server is missing required IDs (role or channel) in the database. Contact an admin.",
            ephemeral=True
        )
        return

    # Check if user has the GPT STAT ACCESS role
    role_ids = [r.id for r in interaction.user.roles] if hasattr(interaction.user, "roles") else []
    if gpt_stat_access_role_id not in role_ids:
        await interaction.response.send_message(
            "You do not have permission to use this command (missing GPT STAT ACCESS role).",
            ephemeral=True
        )
        return

    await interaction.response.defer(ephemeral=True)
    submitter_discord_id = interaction.user.id

    # Get the user’s "registered" data (if any)
    submitter_user = await get_registered_user_by_discord_id(submitter_discord_id)
    submitter_player_name = submitter_user.get('player_name', 'Unknown') if submitter_user else 'Unknown'

    if not any(image.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        await interaction.followup.send("Please upload a valid image file (png, jpg, jpeg).", ephemeral=True)
        return

    try:
        # 1) Read the raw image bytes
        img_bytes = await image.read()
        img_pil = Image.open(BytesIO(img_bytes))
        img_cv = np.array(img_pil)
        logger.info(f"Original image shape: {img_cv.shape}")

        # 2) Define regions based on the RAW shape
        regions = define_regions(img_cv.shape)

        # 3) Show the raw image ephemeral
        await interaction.followup.send(
            content="Here is the submitted image for stats extraction:",
            file=discord.File(BytesIO(img_bytes), filename=image.filename),
            ephemeral=True
        )

        # 4) Process OCR
        players_data = await asyncio.to_thread(process_for_ocr, img_cv, regions)
        logger.info(f"Players data extracted: {players_data}")

        # Filter out rows with junk or missing player names
        players_data = [
            p for p in players_data
            if p.get('player_name') and str(p.get('player_name')).strip() not in ["", "0", ".", "a"]
        ]

        # Check for minimum number of valid players
        if len(players_data) < 2:
            await interaction.followup.send("At least 2 players with valid names must be present in the image.", ephemeral=True)
            return

        # 5) Resolve recognized names -> DB
        registered_users = await get_registered_users()
        for player in players_data:
            ocr_name = player.get('player_name')
            if ocr_name:
                from database import find_best_match
                cleaned_ocr = clean_ocr_result(ocr_name, 'Name')

                # --- CLEAN BOTH NAMES BEFORE MATCHING ---
                db_names = [u["player_name"] for u in registered_users]
                ocr_name_clean = clean_for_match(cleaned_ocr)
                db_names_clean = [clean_for_match(n) for n in db_names]
                best_match_cleaned, match_score = find_best_match(
                    ocr_name_clean,
                    db_names_clean,
                    threshold=MATCH_SCORE_THRESHOLD
                )
                if best_match_cleaned and match_score is not None and match_score >= MATCH_SCORE_THRESHOLD:
                    idx = db_names_clean.index(best_match_cleaned)
                    matched_user = registered_users[idx]
                    player['player_name'] = matched_user["player_name"]
                    player['discord_id'] = matched_user.get("discord_id")
                    player['discord_server_id'] = matched_user.get("discord_server_id")
                    if matched_user.get("discord_server_id"):
                        clan_name = await get_clan_name_by_discord_server_id(matched_user["discord_server_id"])
                        player['clan_name'] = clan_name
                    else:
                        player['clan_name'] = "N/A"
                else:
                    player['player_name'] = None
                    player['discord_id'] = None
                    player['discord_server_id'] = None
                    player['clan_name'] = "N/A"
            else:
                player['player_name'] = None
                player['discord_id'] = None
                player['discord_server_id'] = None
                player['clan_name'] = "N/A"

        # Only keep matched (registered) players for all further steps:
        players_data = [p for p in players_data if p.get('player_name')]

        # Check for minimum number of registered players
        if len(players_data) < 2:
            await interaction.followup.send(
                "At least 2 registered players must be detected in the image. "
                "All reported players must be registered in the database.",
                ephemeral=True
            )
            return

        # 7) Build ephemeral embed for user confirmation
        single_embed = build_single_embed(players_data, submitter_player_name)
        shared_data = SharedData(
            players_data,
            submitter_player_name,
            registered_users,
            monitor_channel_id,
            screenshot_bytes=img_bytes,
            screenshot_filename=image.filename)
        view = ConfirmationView(shared_data)
        shared_data.view = view

        message = await interaction.followup.send(
            content="**Extracted Data:** Please confirm the extracted data.",
            embeds=[single_embed],
            view=view,
            ephemeral=True
        )
        shared_data.message = message

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        traceback_str = ''.join(traceback.format_tb(e.__traceback__))
        logger.error(f"Traceback: {traceback_str}")
        await interaction.followup.send("An error occurred while processing the image.", ephemeral=True)

########################################
# BOT EVENTS
########################################

@bot.event
async def on_ready():
    await tree.sync()
    logger.debug("Bot is starting...")
    print(f"Logged in as {bot.user}.")

########################################
# RUN THE BOT
########################################

bot.run(DISCORD_TOKEN)
