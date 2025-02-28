import json
import os
from typing import Optional, Dict

# File to store user locations
LOCATION_FILE = "user_locations.json"

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
