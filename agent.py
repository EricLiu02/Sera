import os
import json
from mistralai import Mistral
import discord
from check_loc import get_user_location, set_user_location
from conversation_tracker import get_conversation_context, get_conversation_topic, set_conversation_topic
import asyncio
import random

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = """You are Sera, a helpful assistant that can help users with location-related queries.
You have access to the following functions:
- get_user_location: Get a user's saved location
- set_user_location: Set a user's location
- request_location_permission: Ask a user for permission to share their location

When a user asks about locations, determine which function to call and provide the necessary parameters.
Always remember that your name is Sera, and users will address you by this name.
"""

# Dictionary to store pending location requests
# Format: {request_id: {"requester": user_id, "target": user_id, "channel": channel_id}}
pending_location_requests = {}


class MistralAgent:
    def __init__(self):
        # Direct initialization for testing
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")  # Read from .env file
        self.client = Mistral(api_key=MISTRAL_API_KEY)

    async def run(self, message: discord.Message):
        # Check if this is a response to a pending location request
        response = await self._check_location_permission_response(message)
        if response:
            return response
            
        # Define available functions once
        functions = [
            {
                "type": "function",
                "function": {
                    "name": "get_user_location",
                    "description": "Get the location of the user who sent the message",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_user_location",
                    "description": "Set the location for the user who sent the message",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to set for the user"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "request_location_permission",
                    "description": "Request permission to share another user's location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mentioned_user_id": {
                                "type": "string",
                                "description": "The ID of the mentioned user"
                            }
                        },
                        "required": ["mentioned_user_id"]
                    }
                }
            }
        ]
        
        # Process the message with Mistral
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message.content}
        ]

        # Call Mistral with function definitions
        response = await self._call_mistral_with_retry(MISTRAL_MODEL, messages, functions, "auto")
        
        # Check if Mistral wants to call a function
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            
            try:
                arguments = json.loads(tool_call.function.arguments)
                
                # Handle different function calls
                if function_name == "get_user_location":
                    return await self._handle_get_location(message.author)
                    
                elif function_name == "set_user_location":
                    if "location" in arguments:
                        return await self._handle_set_location(message.author, arguments["location"])
                    else:
                        return "I need a location to set. Please specify where you are."
                        
                elif function_name == "request_location_permission":
                    # Find the mentioned user
                    mentioned_user = None
                    for user in message.mentions:
                        mentioned_user = user
                        break
                        
                    if mentioned_user:
                        return await self._handle_other_user_location_query(message, mentioned_user)
                    else:
                        return "I couldn't find the user you're asking about. Please mention them with @."
                
            except json.JSONDecodeError:
                pass
                
        # If no function call or if there was an error, return the regular response
        return response.choices[0].message.content
    
    async def is_message_relevant(self, message: discord.Message, channel_id: str) -> bool:
        """Determine if a message is relevant to the ongoing conversation."""
        # Get conversation context and topic
        context = get_conversation_context(channel_id)
        
        # Always consider messages relevant for short conversations
        if len(context) <= 5:
            return True
        
        # Simple keyword matching for obvious relevance
        topic = get_conversation_topic(channel_id)
        if topic:
            topic_keywords = topic.lower().split()
            message_text = message.content.lower()
            
            # Check if any topic keywords are in the message
            for keyword in topic_keywords:
                if len(keyword) > 3 and keyword in message_text:  # Only check keywords longer than 3 chars
                    return True
        
        # Only use API for ambiguous cases
        # Prepare the query for Mistral
        context_text = "\n".join(context[-5:])  # Use last 5 messages for context
        
        messages = [
            {"role": "system", "content": f"""You are analyzing a conversation to determine if a new message is relevant.
The conversation so far has been about: {topic if topic else 'various topics'}.
Previous messages:
{context_text}

Determine if the new message is relevant to this conversation. 
Respond with only 'yes' if it's relevant or 'no' if it's not relevant."""},
            {"role": "user", "content": f"New message: {message.content}"}
        ]
        
        response = await self._call_mistral_with_retry(MISTRAL_MODEL, messages)
        
        answer = response.choices[0].message.content.strip().lower()
        return "yes" in answer[:5]  # Check if the first few characters contain "yes"
    
    async def _determine_conversation_topic(self, initial_message: str, channel_id: str) -> None:
        """Determine the topic of a conversation based on the initial message."""
        messages = [
            {"role": "system", "content": "Analyze the following message and describe the main topic in 5 words or less."},
            {"role": "user", "content": initial_message}
        ]
        
        response = await self._call_mistral_with_retry(MISTRAL_MODEL, messages)
        
        topic = response.choices[0].message.content.strip()
        set_conversation_topic(channel_id, topic)
    
    async def _handle_get_location(self, user):
        """Handle getting a user's location."""
        user_location = get_user_location(str(user.id))
        if user_location:
            return f"Based on your saved information, your location is: {user_location}"
        else:
            return "I don't have your location information. You can set it using the !setlocation command or by telling me where you are."
    
    async def _handle_set_location(self, user, location):
        """Handle setting a user's location."""
        set_user_location(str(user.id), location)
        return f"I've updated your location to: {location}"
    
    async def _handle_other_user_location_query(self, message, mentioned_user):
        """Handle a query about another user's location."""
        # If the user is asking about their own location through a mention
        if str(message.author.id) == str(mentioned_user.id):
            return await self._handle_get_location(message.author)
        
        # Otherwise, proceed with the permission request
        request_id = f"{message.id}"
        
        # Store the request
        pending_location_requests[request_id] = {
            "requester": str(message.author.id),
            "requester_name": message.author.display_name,
            "target": str(mentioned_user.id),
            "channel": str(message.channel.id)
        }
        
        # Ask the mentioned user for permission
        return f"<@{mentioned_user.id}>, {message.author.display_name} is asking about your location. Do you want me to share it? Please reply with 'yes' or 'no'."
    
    async def _check_location_permission_response(self, message):
        """Check if this message is a response to a location permission request."""
        # Check all pending requests where this user is the target
        for request_id, request_data in list(pending_location_requests.items()):
            if request_data["target"] == str(message.author.id):
                response_text = message.content.lower().strip()
                
                # Check if the response is affirmative
                if response_text in ["yes", "y", "sure", "okay", "ok", "yep", "yeah"]:
                    # Get the location
                    location = get_user_location(str(message.author.id))
                    
                    # Remove the request
                    del pending_location_requests[request_id]
                    
                    if location:
                        return f"<@{request_data['requester']}>, {message.author.display_name}'s location is: {location}"
                    else:
                        return f"<@{request_data['requester']}>, {message.author.display_name} has given permission, but they haven't set their location yet."
                
                # Check if the response is negative
                elif response_text in ["no", "n", "nope", "nah", "negative"]:
                    # Remove the request
                    del pending_location_requests[request_id]
                    return f"<@{request_data['requester']}>, {message.author.display_name} has declined to share their location."
        
        return None

    async def _call_mistral_with_retry(self, model, messages, tools=None, tool_choice=None, max_retries=3):
        """Call Mistral API with retry logic for rate limits."""
        retries = 0
        while retries <= max_retries:
            try:
                if tools:
                    return await self.client.chat.complete_async(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice=tool_choice
                    )
                else:
                    return await self.client.chat.complete_async(
                        model=model,
                        messages=messages
                    )
            except Exception as e:
                if "429" in str(e) and retries < max_retries:
                    # Exponential backoff with jitter
                    wait_time = (2 ** retries) + random.uniform(0, 1)
                    print(f"Rate limited. Retrying in {wait_time:.2f} seconds...")
                    await asyncio.sleep(wait_time)
                    retries += 1
                else:
                    raise
