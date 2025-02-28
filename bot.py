import os
import discord
import logging

from discord.ext import commands
from dotenv import load_dotenv
from agent import MistralAgent
from check_loc import set_user_location, get_user_location
from conversation_tracker import (
    start_conversation, update_conversation, is_conversation_active, 
    end_conversation, check_for_ended_conversations
)

PREFIX = "!"

# Setup logging
logger = logging.getLogger("discord")

# Load the environment variables
load_dotenv()

# Create the bot with all intents
# The message content and members intent must be enabled in the Discord Developer Portal for the bot to work.
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Import the Mistral agent from the agent.py file
agent = MistralAgent()


# Get the token from the environment variables
token = os.getenv("DISCORD_TOKEN")


@bot.event
async def on_ready():
    """
    Called when the client is done preparing the data received from Discord.
    Prints message on terminal when bot successfully connects to discord.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_ready
    """
    logger.info(f"{bot.user} has connected to Discord!")


@bot.event
async def on_message(message: discord.Message):
    """
    Called when a message is sent in any channel the bot can see.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_message
    """
    # Don't delete this line! It's necessary for the bot to process commands.
    await bot.process_commands(message)

    # Ignore messages from self or other bots to prevent infinite loops.
    if message.author.bot or message.content.startswith("!"):
        return
    
    channel_id = str(message.channel.id)
    user_id = str(message.author.id)
    
    # Check if this message contains "sera" to start a conversation
    message_words = message.content.lower().split()
    is_addressing_sera = "sera" in message_words or any(
        word.startswith("sera,") or word.startswith("sera.") or 
        word.startswith("sera!") or word.startswith("sera?") 
        for word in message_words
    )
    
    # Start a new conversation if addressing Sera directly
    if is_addressing_sera:
        start_conversation(channel_id, user_id, message.content)
        logger.info(f"Starting conversation with {message.author} in channel {message.channel}")
    # Or continue an existing conversation if the message is relevant
    elif is_conversation_active(channel_id):
        # Check if the message is relevant to the ongoing conversation
        is_relevant = await agent.is_message_relevant(message, channel_id)
        if is_relevant:
            update_conversation(channel_id, user_id, message.content)
            logger.info(f"Continuing conversation with {message.author} in channel {message.channel}")
        else:
            # Message is not relevant to the conversation
            logger.info(f"Ignoring irrelevant message from {message.author}")
            return
    else:
        # Not addressing Sera and no active conversation
        return
    
    # Process the message with the agent
    logger.info(f"Processing message from {message.author}: {message.content}")
    response = await agent.run(message)
    
    # Send the response back to the channel
    await message.reply(response)
    
    # Check if conversations have ended due to timeout
    ended_conversations = check_for_ended_conversations()
    for ended_channel_id, participants in ended_conversations.items():
        channel = bot.get_channel(int(ended_channel_id))
        if channel:
            await channel.send("Seems like the party has ended, I am going back home. Feel free to call me again by saying 'Sera'!")


# Commands


# This example command is here to show you how to add commands to the bot.
# Run !ping with any number of arguments to see the command in action.
# Feel free to delete this if your project will not need commands.
@bot.command(name="ping", help="Pings the bot.")
async def ping(ctx, *, arg=None):
    if arg is None:
        await ctx.send("Pong!")
    else:
        await ctx.send(f"Pong! Your argument was {arg}")

@bot.command(name="setlocation", help="Set your current location (street, district, city, etc.)")
async def set_location(ctx, *, location=None):
    """Set the user's location."""
    if location is None:
        await ctx.send("Please provide a location. Example: !setlocation New York")
        return
    
    set_user_location(str(ctx.author.id), location)
    await ctx.send(f"Your location has been set to: {location}")

@bot.command(name="location", help="Check a user's location")
async def check_location(ctx, member: discord.Member = None):
    """Check a user's location."""
    # If no member is specified, use the command author
    if member is None:
        member = ctx.author
    
    location = get_user_location(str(member.id))
    if location:
        await ctx.send(f"{member.display_name}'s location is: {location}")
    else:
        await ctx.send(f"{member.display_name} hasn't set their location yet. They can use !setlocation to set it.")

# Start the bot, connecting it to the gateway
bot.run(token)


