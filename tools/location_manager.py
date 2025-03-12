import os
import json
import asyncio
from typing import Dict, Optional
import discord
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_core.tools.base import ArgsSchema

# JSON file to store user locations
LOCATION_FILE = "user_locations.json"

# Ensure the location file exists
if not os.path.exists(LOCATION_FILE):
    with open(LOCATION_FILE, "w") as f:
        json.dump({}, f)

class LocationInput(BaseModel):
    """
    Input for the location tool.
    """
    user_id: str = Field(
        description="The ID of the user whose location is being set or retrieved"
    )
    location: Optional[str] = Field(
        default=None,
        description="The location to set for the user (only needed for setting location)"
    )

class LocationTool(BaseTool):
    """
    A tool to manage user location data, storing and retrieving locations from a JSON file.
    """
    
    name: str = "location_manager"
    description: str = "Manages user location data for personalized responses"
    args_schema: Optional[ArgsSchema] = LocationInput
    return_direct: bool = True
    
    def _load_locations(self) -> Dict[str, str]:
        """Load the saved locations from the JSON file."""
        if os.path.exists(LOCATION_FILE):
            try:
                with open(LOCATION_FILE, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}
    
    def _save_locations(self, locations: Dict[str, str]) -> None:
        """Save the locations to the JSON file."""
        with open(LOCATION_FILE, 'w') as f:
            json.dump(locations, f)
    
    def _run(self, user_id: str, location: Optional[str] = None) -> str:
        """
        Sets or gets a user's location.
        
        Args:
            user_id (str): The ID of the user
            location (Optional[str]): The location to set (if None, retrieves the location)
            
        Returns:
            str: A message indicating success or the user's location
        """
        locations = self._load_locations()
        
        # If location is provided, set it
        if location:
            locations[user_id] = location
            self._save_locations(locations)
            return f"Location set to: {location}"
        
        # Otherwise, retrieve the location
        user_location = locations.get(user_id)
        if user_location:
            return user_location
        return "I notice you haven't set your location yet. Where are you located?"
    
    async def _arun(self, user_id: str, location: Optional[str] = None) -> str:
        """
        Async version of the location management functionality.
        """
        return self._run(user_id, location)
    
    async def wait_for_location(self, message):
        """
        Wait for a user to provide their location in Discord.
        
        Args:
            message: The Discord message object
            
        Returns:
            str: A confirmation message
        """
        user_id = str(message.author.id)
        
        # Try to get current location
        locations = self._load_locations()
        current_location = locations.get(user_id)
        
        if current_location:
            return f"Your current location is set to: {current_location}. You can update it by typing a new location."
        else:
            return "Where are you located? Please type your location."
    def get_user_location(self, user_id: str) -> str:
        """
        Retrieves the stored location of another user.
        """
        user_data = self._load_locations()
        if user_id in user_data:
            return f"{user_id} is located in {user_data[user_id]}."
        return "I don’t have their location stored yet."
