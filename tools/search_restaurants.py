import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import googlemaps
from langchain_core.tools import BaseTool
from mistralai import Mistral
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
print("Mistral Key", MISTRAL_API_KEY)
MISTRAL_MODEL = "mistral-large-latest"

REVIEW_SUMMARY_PROMPT = """You are a helpful assistant specializing in summarizing restaurant reviews.
Provide a very concise 2-3 sentence summary that captures:
1. Overall sentiment and most mentioned positives
2. Any notable criticisms or areas for improvement
3. 1-2 most recommended dishes (if mentioned)

Keep your summary brief, direct, and informative."""


class SearchRestaurants:
    """
    A class to interact with the Google Places API to get restaurant information and reviews.
    """

    def __init__(self):
        """Initialize the Google Maps client with API key from environment variables."""
        self.api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_MAPS_API_KEY environment variable is not set")

        self.client = googlemaps.Client(key=self.api_key)

    def search_restaurant(
        self, query: str, location: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for restaurants based on a query and optional location.

        Args:
            query: The search query (e.g., "pizza restaurant")
            location: Optional location to search near (e.g., "San Francisco, CA")

        Returns:
            A list of restaurant results
        """
        search_params = {"query": query, "type": "restaurant"}

        # If location is provided, geocode it and add location bias
        if location:
            geocode_result = self.client.geocode(location)
            if geocode_result:
                lat = geocode_result[0]["geometry"]["location"]["lat"]
                lng = geocode_result[0]["geometry"]["location"]["lng"]
                search_params["location"] = (lat, lng)
                search_params["radius"] = 5000  # 5km radius

        # Perform the search
        places_result = self.client.places(**search_params)
        return places_result.get("results", [])

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
            fields=[
                "name",
                "rating",
                "formatted_address",
                "formatted_phone_number",
                "opening_hours",
                "website",
                "price_level",
                "reviews",
                "photo",
                "user_ratings_total",
            ],
        )

        return place_details.get("result", {})

    def get_restaurant_reviews(self, place_id: str) -> List[Dict[str, Any]]:
        """
        Get reviews for a specific restaurant.

        Args:
            place_id: The Google Place ID of the restaurant

        Returns:
            A list of reviews for the restaurant
        """
        place_details = self.client.place(
            place_id=place_id, fields=["reviews"])

        return place_details.get("result", {}).get("reviews", [])

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

        if "rating" in restaurant:
            stars = "⭐" * int(restaurant.get("rating", 0))
            info.append(
                f"Rating: {restaurant.get('rating')} {stars} ({restaurant.get('user_ratings_total', 0)} reviews)"
            )

        if "formatted_address" in restaurant:
            info.append(f"Address: {restaurant.get('formatted_address')}")

        if "formatted_phone_number" in restaurant:
            info.append(f"Phone: {restaurant.get('formatted_phone_number')}")

        if "website" in restaurant:
            info.append(f"Website: {restaurant.get('website')}")

        if "price_level" in restaurant:
            price_level = "💰" * restaurant.get("price_level", 0)
            info.append(f"Price Level: {price_level}")

        if (
            "opening_hours" in restaurant
            and "weekday_text" in restaurant["opening_hours"]
        ):
            info.append("Hours:")
            for hours in restaurant["opening_hours"]["weekday_text"]:
                info.append(f"  {hours}")

        return "\n".join(info)

    def format_reviews(
        self, reviews: List[Dict[str, Any]], max_reviews: int = 5
    ) -> str:
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
            stars = "⭐" * int(review.get("rating", 0))
            author = review.get("author_name", "Anonymous")
            time = datetime.fromtimestamp(
                review.get("time", 0)).strftime("%Y-%m-%d")
            text = review.get("text", "No comment")

            formatted_reviews.append(f"{i+1}. {author} - {stars} ({time})")
            formatted_reviews.append(f'   "{text}"')
            formatted_reviews.append("")

        return "\n".join(formatted_reviews)

    def prepare_reviews_for_summary(
        self, reviews: List[Dict[str, Any]], max_reviews: int = 5
    ) -> str:
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
            rating = review.get("rating", 0)
            text = review.get("text", "No comment")

            reviews_text.append(f"Review {i+1} (Rating: {rating}/5): {text}")

        return "\n\n".join(reviews_text)


class SearchRestaurantsTool(BaseTool):
    name: str = "search_restaurants"
    description: str = """Use this tool for ANY restaurant-related queries. This includes:
            - Finding restaurants by cuisine, name, or type
            - Getting restaurant information and reviews
            - Finding dining options in specific locations"""
    restaurant_api: SearchRestaurants = Field(
        default_factory=SearchRestaurants)
    client: Mistral = Field(
        default_factory=lambda: Mistral(api_key=MISTRAL_API_KEY))
    return_direct: bool = True

    def _run(
        self, query: str, location: Optional[str] = None, start_index: int = 0
    ) -> str:
        # Search for restaurants
        restaurants = self.restaurant_api.search_restaurant(query, location)
        if not restaurants:
            return f"I couldn't find any restaurants matching your search. Would you like to try a different query or location?"

        # Calculate the range of restaurants to show
        total_restaurants = len(restaurants)
        start = start_index
        end = min(start + 3, total_restaurants)

        if start >= total_restaurants:
            return "I've shown all available restaurants for this search. Would you like to try a different query?"

        response_parts = []
        place_id_map = []

        # Let the LLM handle the introduction text through the system prompt
        if start == 0:
            response_parts.append("Here are my recommendations:")
        else:
            response_parts.append("Here are more recommendations:")

        for i in range(start, end):
            restaurant = restaurants[i]
            place_id = restaurant["place_id"]
            place_id_map.append(place_id)

            # Add a separator between restaurants
            if i > start:
                response_parts.append("\n" + "-" * 30 + "\n")

            # Get details and reviews
            restaurant_details = self.restaurant_api.get_restaurant_details(
                place_id)
            reviews = self.restaurant_api.get_restaurant_reviews(place_id)

            # Format basic restaurant information
            info = []
            # Add name without any visible markers
            info.append(
                f"**{restaurant_details.get('name', 'Unknown Restaurant')}**")

            rating_parts = []
            if "rating" in restaurant_details:
                stars = "⭐" * int(restaurant_details.get("rating", 0))
                rating_parts.append(
                    f"{restaurant_details.get('rating')} {stars}")
            if "price_level" in restaurant_details:
                price_level = "💰" * restaurant_details.get("price_level", 0)
                rating_parts.append(price_level)
            if rating_parts:
                info.append(" | ".join(rating_parts))

            if "formatted_address" in restaurant_details:
                info.append(f"📍 {restaurant_details.get('formatted_address')}")

            response_parts.append("\n".join(info))

            # Generate AI summary of reviews if available
            if reviews:
                reviews_text = self.restaurant_api.prepare_reviews_for_summary(
                    reviews)

                # Generate summary using Mistral AI
                summary_messages = [
                    {"role": "system", "content": REVIEW_SUMMARY_PROMPT},
                    {"role": "user", "content": reviews_text},
                ]

                summary_response = self.client.chat.complete(
                    model=MISTRAL_MODEL,
                    messages=summary_messages,
                )

                response_parts.append(
                    f"\n💬 {summary_response.choices[0].message.content}"
                )
            else:
                response_parts.append("\nNo reviews available yet.")

        # Add information about additional results
        remaining_count = total_restaurants - end
        if remaining_count > 0:
            response_parts.append("\nWould you like to see more restaurant recommendations?")
        else:
            response_parts.append("\nThose are all the restaurants I found. Would you like to try a different search?")

        # Store metadata in a completely invisible way using zero-width spaces
        metadata = []
        for i, (pid, details) in enumerate(zip(place_id_map, [restaurants[i] for i in range(start, end)]), 1):
            metadata.append(f"{details.get('name', 'Unknown Restaurant')}:{i}:{pid}")
        
        # Add metadata with multiple zero-width spaces to ensure invisibility
        response_parts.append(f"\u200b\u200c\u200d{','.join(metadata)}\u200b\u200c\u200d")

        # Join all parts except metadata
        visible_content = "\n".join(response_parts[:-1])
        
        # If the visible content is too long, truncate it
        if len(visible_content) > 1900:
            visible_content = visible_content[:1850] + "\n\n[Some content truncated due to length]"
        
        # Return visible content with hidden metadata appended
        return visible_content + response_parts[-1]

    async def _arun(
        self, query: str, location: Optional[str] = None, start_index: int = 0
    ) -> str:
        """Async implementation of the tool"""
        return self._run(query, location, start_index)
