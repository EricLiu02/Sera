import os
import re
from typing import Any, Dict, Optional

from langchain_core.tools import BaseTool
from mistralai import Mistral
from pydantic import Field

from tools.search_restaurants import SearchRestaurants

# Constants
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-large-latest"
REVIEW_SUMMARY_PROMPT = """Please provide a one-sentence summary of this restaurant review, capturing the key sentiment and any specific highlights mentioned."""


class RestaurantDetailsTool(BaseTool):
    name: str = "get_restaurant_details"
    description: str = """Use this tool when users want more detailed information about a specific restaurant.
    This includes:
    - Full opening hours
    - Complete contact information
    - Website details
    - Detailed price information
    - Full address and location details
    - Popular times to visit
    - Menu information (if available)
    """
    restaurant_api: SearchRestaurants = Field(default_factory=SearchRestaurants)
    client: Mistral = Field(default_factory=lambda: Mistral(api_key=MISTRAL_API_KEY))
    return_direct: bool = True

    def _format_detailed_info(self, details: Dict[str, Any]) -> str:
        """Format detailed restaurant information into a readable string."""
        info_parts = []

        # Restaurant name and basic info
        info_parts.append(f"🏪 **{details.get('name', 'Unknown Restaurant')}**\n")

        # Rating and reviews
        if "rating" in details:
            stars = "⭐" * int(details.get("rating", 0))
            total_ratings = details.get("user_ratings_total", 0)
            info_parts.append(f"📊 Rating: {details.get('rating')} {stars}")
            info_parts.append(f"   Based on {total_ratings:,} reviews\n")

        # Price level with explanation
        if "price_level" in details:
            price_map = {
                1: "Inexpensive",
                2: "Moderate",
                3: "Expensive",
                4: "Very Expensive",
            }
            price_level = details.get("price_level")
            price_text = price_map.get(price_level, "Unknown")
            price_symbols = "💰" * price_level
            info_parts.append(f"💵 Price Level: {price_text} {price_symbols}\n")

        # Contact information
        contact_info = []
        if "formatted_phone_number" in details:
            contact_info.append(f"📞 Phone: {details.get('formatted_phone_number')}")
        if "website" in details:
            contact_info.append(f"🌐 Website: {details.get('website')}")
        if "formatted_address" in details:
            contact_info.append(f"📍 Address: {details.get('formatted_address')}")
        if contact_info:
            info_parts.append("\n".join(contact_info) + "\n")

        # Opening hours with current status
        if "opening_hours" in details:
            hours = details["opening_hours"]
            if "open_now" in hours:
                status = "🟢 Open now" if hours["open_now"] else "🔴 Closed now"
                info_parts.append(f"⏰ Status: {status}")
            if "weekday_text" in hours:
                info_parts.append("\n📅 Opening Hours:")
                for day_hours in hours["weekday_text"]:
                    info_parts.append(f"   {day_hours}")
            info_parts.append("")

        # Additional attributes
        if "serves_beer" in details:
            info_parts.append("🍺 Serves alcohol")
        if "serves_wine" in details:
            info_parts.append("🍷 Serves wine")
        if "serves_vegetarian_food" in details:
            info_parts.append("🥗 Vegetarian options available")
        if "wheelchair_accessible_entrance" in details:
            info_parts.append("♿ Wheelchair accessible")
        if "outdoor_seating" in details:
            info_parts.append("🪑 Outdoor seating available")
        if "delivery" in details:
            info_parts.append("🚚 Delivery available")
        if "takeout" in details:
            info_parts.append("📦 Takeout available")
        if "dine_in" in details:
            info_parts.append("🍽️ Dine-in available")

        # Add review samples if available
        if "reviews" in details:
            reviews = details["reviews"]
            if reviews:
                info_parts.append("\n📝 Recent Reviews:")
                # Sort reviews by rating (highest first) and recency
                sorted_reviews = sorted(
                    reviews,
                    key=lambda x: (x.get("rating", 0), x.get("time", 0)),
                    reverse=True,
                )
                # Take up to 3 most recent, highly-rated reviews
                for review in sorted_reviews[:3]:
                    stars = "⭐" * int(review.get("rating", 0))
                    author = review.get("author_name", "Anonymous")
                    time = review.get("relative_time_description", "")
                    text = review.get("text", "No comment").strip()

                    # Generate a concise summary using Mistral
                    summary_messages = [
                        {"role": "system", "content": REVIEW_SUMMARY_PROMPT},
                        {"role": "user", "content": text},
                    ]

                    try:
                        summary_response = self.client.chat.complete(
                            model=MISTRAL_MODEL,
                            messages=summary_messages,
                        )
                        summary = summary_response.choices[0].message.content.strip()
                    except:
                        # Fallback to truncation if summarization fails
                        summary = text[:150] + "..." if len(text) > 150 else text

                    info_parts.append(f"\n{author} - {stars} - {time}")
                    info_parts.append(f'"{summary}"')

        return "\n".join(info_parts)

    def _run(self, query: str) -> str:
        """Get detailed information about a specific restaurant.

        Args:
            query: Restaurant name from the results or a new restaurant name/description
        """
        try:
            # Check if the input contains the invisible metadata using unicode zero-width characters
            metadata_match = re.search(
                r"\u200b\u200c\u200d(.+?)\u200b\u200c\u200d", query
            )

            if metadata_match:
                # Extract metadata
                metadata = metadata_match.group(1).split(",")
                # Find the matching restaurant
                restaurant_name = query.split("**")[1] if "**" in query else query
                restaurant_name = restaurant_name.strip()

                for entry in metadata:
                    name, _, place_id = entry.split(":")
                    if name.strip() == restaurant_name:
                        details = self.restaurant_api.get_restaurant_details(place_id)
                        if details:
                            return self._format_detailed_info(details)

                return "Please select a restaurant from the search results."
            else:
                # If not from search results, search for the restaurant by name
                search_results = self.restaurant_api.search_restaurant(query)
                if not search_results:
                    return f"I couldn't find a restaurant matching '{query}'. Could you please try searching for it first?"
                place_id = search_results[0]["place_id"]  # Use the first match

            details = self.restaurant_api.get_restaurant_details(place_id)
            if not details:
                return "I couldn't find detailed information for this restaurant."

            return self._format_detailed_info(details)
        except Exception as e:
            return f"I encountered an error while fetching restaurant details: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async implementation of the tool"""
        return self._run(query)
