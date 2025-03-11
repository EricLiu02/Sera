import json
import os
from typing import Dict, Optional
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema

# File to store user locations
LOCATION_FILE = "user_locations.json"

class LocationInput(BaseModel):
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
    
    def __init__(self, **data):
        """
        Initializes the LocationTool.
        """
        super().__init__(**data)
    
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
        return "No location set for this user"
    
    async def _arun(self, user_id: str, location: Optional[str] = None) -> str:
        """
        Async version of the location management functionality.
        """
        return self._run(user_id, location)
    
    def set_user_location(self, user_id: str, location: str) -> None:
        """Set a user's location."""
        locations = self._load_locations()
        locations[user_id] = location
        self._save_locations(locations)
    
    def get_user_location(self, user_id: str) -> Optional[str]:
        """Get a user's location if it exists."""
        locations = self._load_locations()
        return locations.get(user_id)


# For backward compatibility with existing code
def load_locations() -> Dict[str, str]:
    """Load the saved locations from the JSON file."""
    if os.path.exists(LOCATION_FILE):
        try:
            with open(LOCATION_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_locations(locations: Dict[str, str]) -> None:
    """Save the locations to the JSON file."""
    with open(LOCATION_FILE, 'w') as f:
        json.dump(locations, f)

def set_user_location(user_id: str, location: str) -> None:
    """Set a user's location."""
    locations = load_locations()
    locations[user_id] = location
    save_locations(locations)

def get_user_location(user_id: str) -> Optional[str]:
    """Get a user's location if it exists."""
    locations = load_locations()
    return locations.get(user_id)


if __name__ == "__main__":
    # Example usage
    location_tool = LocationTool()
    
    # Set a location
    location_tool.set_user_location("123456789", "New York City")
    
    # Get a location
    location = location_tool.get_user_location("123456789")
    print(f"User location: {location}")
