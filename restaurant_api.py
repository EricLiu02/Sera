import os
import googlemaps
from datetime import datetime
from typing import Dict, List, Any, Optional

class RestaurantAPI:
    """
    A class to interact with the Google Places API to get restaurant information and reviews.
    """
    
    def __init__(self):
        """Initialize the Google Maps client with API key from environment variables."""
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is not set")
        
        self.client = googlemaps.Client(key=self.api_key)
    
    def search_restaurant(self, query: str, location: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for restaurants based on a query and optional location.
        
        Args:
            query: The search query (e.g., "pizza restaurant")
            location: Optional location to search near (e.g., "San Francisco, CA")
            
        Returns:
            A list of restaurant results
        """
        search_params = {
            'query': query,
            'type': 'restaurant'
        }
        
        # If location is provided, geocode it and add location bias
        if location:
            geocode_result = self.client.geocode(location)
            if geocode_result:
                lat = geocode_result[0]['geometry']['location']['lat']
                lng = geocode_result[0]['geometry']['location']['lng']
                search_params['location'] = (lat, lng)
                search_params['radius'] = 5000  # 5km radius
        
        # Perform the search
        places_result = self.client.places(**search_params)
        return places_result.get('results', [])
    
    def get_restaurant_details(self, place_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific restaurant.
        
        Args:
            place_id: The Google Place ID of the restaurant
            
        Returns:
            Detailed information about the restaurant
        """
        place_details = self.client.place(
            place_id=place_id,
            fields=['name', 'rating', 'formatted_address', 'formatted_phone_number', 
                   'opening_hours', 'website', 'price_level', 'reviews', 'photo', 
                   'user_ratings_total']
        )
        
        return place_details.get('result', {})
    
    def get_restaurant_reviews(self, place_id: str) -> List[Dict[str, Any]]:
        """
        Get reviews for a specific restaurant.
        
        Args:
            place_id: The Google Place ID of the restaurant
            
        Returns:
            A list of reviews for the restaurant
        """
        place_details = self.client.place(
            place_id=place_id,
            fields=['reviews']
        )
        
        return place_details.get('result', {}).get('reviews', [])
    
    def format_restaurant_info(self, restaurant: Dict[str, Any]) -> str:
        """
        Format restaurant information into a readable string.
        
        Args:
            restaurant: Restaurant details dictionary
            
        Returns:
            Formatted string with restaurant information
        """
        info = []
        info.append(f"🍽️ **{restaurant.get('name', 'Unknown Restaurant')}**")
        
        if 'rating' in restaurant:
            stars = "⭐" * int(restaurant.get('rating', 0))
            info.append(f"Rating: {restaurant.get('rating')} {stars} ({restaurant.get('user_ratings_total', 0)} reviews)")
        
        if 'formatted_address' in restaurant:
            info.append(f"Address: {restaurant.get('formatted_address')}")
        
        if 'formatted_phone_number' in restaurant:
            info.append(f"Phone: {restaurant.get('formatted_phone_number')}")
        
        if 'website' in restaurant:
            info.append(f"Website: {restaurant.get('website')}")
        
        if 'price_level' in restaurant:
            price_level = "💰" * restaurant.get('price_level', 0)
            info.append(f"Price Level: {price_level}")
        
        if 'opening_hours' in restaurant and 'weekday_text' in restaurant['opening_hours']:
            info.append("Hours:")
            for hours in restaurant['opening_hours']['weekday_text']:
                info.append(f"  {hours}")
        
        return "\n".join(info)
    
    def format_reviews(self, reviews: List[Dict[str, Any]], max_reviews: int = 5) -> str:
        """
        Format restaurant reviews into a readable string.
        
        Args:
            reviews: List of review dictionaries
            max_reviews: Maximum number of reviews to include
            
        Returns:
            Formatted string with reviews
        """
        if not reviews:
            return "No reviews available."
        
        formatted_reviews = ["**Recent Reviews:**"]
        
        for i, review in enumerate(reviews[:max_reviews]):
            stars = "⭐" * int(review.get('rating', 0))
            author = review.get('author_name', 'Anonymous')
            time = datetime.fromtimestamp(review.get('time', 0)).strftime('%Y-%m-%d')
            text = review.get('text', 'No comment')
            
            formatted_reviews.append(f"{i+1}. {author} - {stars} ({time})")
            formatted_reviews.append(f"   \"{text}\"")
            formatted_reviews.append("")
        
        return "\n".join(formatted_reviews)
    
    def prepare_reviews_for_summary(self, reviews: List[Dict[str, Any]], max_reviews: int = 5) -> str:
        """
        Prepare reviews for AI summarization.
        
        Args:
            reviews: List of review dictionaries
            max_reviews: Maximum number of reviews to include
            
        Returns:
            String with reviews formatted for AI summarization
        """
        if not reviews:
            return "No reviews available."
        
        reviews_text = []
        
        for i, review in enumerate(reviews[:max_reviews]):
            rating = review.get('rating', 0)
            text = review.get('text', 'No comment')
            
            reviews_text.append(f"Review {i+1} (Rating: {rating}/5): {text}")
        
        return "\n\n".join(reviews_text) 