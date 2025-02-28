import time
from typing import Dict, Set, Optional

# Conversation tracking
active_conversations = {}  # channel_id -> {last_activity_time, participants, context}
CONVERSATION_TIMEOUT = 180  # 3 minutes of inactivity before ending conversation

def start_conversation(channel_id: str, user_id: str, initial_message: str) -> None:
    """Start or join a conversation in a channel."""
    active_conversations[channel_id] = {
        "last_activity": time.time(),
        "participants": {user_id},
        "active": True,
        "context": [initial_message],  # Keep track of conversation context
        "topic": ""  # Will be filled by the agent's analysis
    }

def update_conversation(channel_id: str, user_id: str, message: str) -> None:
    """Update an existing conversation with new activity."""
    if channel_id in active_conversations:
        active_conversations[channel_id]["last_activity"] = time.time()
        active_conversations[channel_id]["participants"].add(user_id)
        active_conversations[channel_id]["context"].append(message)
        # Limit context to last 10 messages to avoid excessive memory usage
        if len(active_conversations[channel_id]["context"]) > 10:
            active_conversations[channel_id]["context"] = active_conversations[channel_id]["context"][-10:]

def is_conversation_active(channel_id: str) -> bool:
    """Check if there's an active conversation in the channel."""
    if channel_id not in active_conversations:
        return False
        
    # Check if the conversation has timed out
    elapsed = time.time() - active_conversations[channel_id]["last_activity"]
    if elapsed > CONVERSATION_TIMEOUT:
        active_conversations[channel_id]["active"] = False
        return False
        
    return active_conversations[channel_id]["active"]

def get_conversation_context(channel_id: str) -> list:
    """Get the context of the current conversation."""
    if channel_id in active_conversations:
        return active_conversations[channel_id]["context"]
    return []

def set_conversation_topic(channel_id: str, topic: str) -> None:
    """Set the topic of the current conversation."""
    if channel_id in active_conversations:
        active_conversations[channel_id]["topic"] = topic

def get_conversation_topic(channel_id: str) -> str:
    """Get the topic of the current conversation."""
    if channel_id in active_conversations:
        return active_conversations[channel_id]["topic"]
    return ""

def end_conversation(channel_id: str) -> Optional[Set[str]]:
    """End a conversation and return the set of participants."""
    if channel_id in active_conversations and active_conversations[channel_id]["active"]:
        participants = active_conversations[channel_id]["participants"]
        active_conversations[channel_id]["active"] = False
        return participants
    return None

def check_for_ended_conversations() -> Dict[str, Set[str]]:
    """Check for conversations that have timed out and should be ended."""
    ended_conversations = {}
    current_time = time.time()
    
    for channel_id, data in list(active_conversations.items()):
        if data["active"] and (current_time - data["last_activity"]) > CONVERSATION_TIMEOUT:
            ended_conversations[channel_id] = data["participants"]
            data["active"] = False
            
    return ended_conversations 