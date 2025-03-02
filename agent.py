import os
from mistralai import Mistral
import discord
import re
from restaurant_api import RestaurantAPI

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = """You are a helpful assistant. 
You can provide information about restaurants, including details and reviews.
When asked about restaurants, you'll use the restaurant API to provide accurate information.
"""

REVIEW_SUMMARY_PROMPT = """You are a helpful assistant specializing in summarizing restaurant reviews.
Based on the following reviews, provide a concise summary that captures:
1. The overall sentiment (positive/negative/mixed)
2. Common praise points
3. Common criticism points
4. Any standout dishes or experiences mentioned
5. Any consistent service observations

Keep your summary conversational, helpful, and under 150 words.
"""


class MistralAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

        self.client = Mistral(api_key=MISTRAL_API_KEY)
        self.restaurant_api = RestaurantAPI()

    async def run(self, message: discord.Message):
        content = message.content.strip()
        
        # Check if the message is asking about restaurants
        restaurant_patterns = [
            r"restaurant(?:s)?\s+(?:in|near|at)\s+(.+)",
            r"(?:find|search for|look for|get)\s+(?:a|some)?\s*restaurant(?:s)?\s+(?:in|near|at)\s+(.+)",
            r"(?:find|search for|look for|get)\s+(?:a|some)?\s*(.+)\s+restaurant(?:s)?",
            r"(?:what are|show me|tell me about)\s+(?:some)?\s*(?:good)?\s*restaurant(?:s)?\s+(?:in|near|at)\s+(.+)",
            r"reviews?\s+(?:for|of)\s+(.+)\s+restaurant",
            r"(.+)\s+restaurant\s+reviews?",
        ]
        
        restaurant_query = None
        location = None
        
        for pattern in restaurant_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                restaurant_query = match.group(1).strip()
                break
        
        # If it's a restaurant query, use the restaurant API
        if restaurant_query:
            try:
                # Check if there's a specific location mentioned
                location_match = re.search(r"(?:in|near|at)\s+(.+)(?:\s+restaurant|\s*$)", content, re.IGNORECASE)
                if location_match:
                    location = location_match.group(1).strip()
                
                # Search for restaurants
                restaurants = self.restaurant_api.search_restaurant(restaurant_query, location)
                
                if not restaurants:
                    return f"I couldn't find any restaurants matching '{restaurant_query}'" + (f" in {location}" if location else "") + ". Could you try a different search?"
                
                # Get details for the first restaurant
                first_restaurant = restaurants[0]
                place_id = first_restaurant['place_id']
                
                # Get detailed information and reviews
                restaurant_details = self.restaurant_api.get_restaurant_details(place_id)
                reviews = self.restaurant_api.get_restaurant_reviews(place_id)
                
                # Format the restaurant information
                formatted_info = self.restaurant_api.format_restaurant_info(restaurant_details)
                
                # Generate AI summary of reviews if reviews are available
                review_summary = ""
                if reviews:
                    # Prepare reviews for summarization
                    reviews_text = self.restaurant_api.prepare_reviews_for_summary(reviews)
                    
                    # Generate summary using Mistral AI
                    summary_messages = [
                        {"role": "system", "content": REVIEW_SUMMARY_PROMPT},
                        {"role": "user", "content": reviews_text},
                    ]
                    
                    summary_response = await self.client.chat.complete_async(
                        model=MISTRAL_MODEL,
                        messages=summary_messages,
                    )
                    
                    review_summary = f"\n\n**Review Summary:**\n{summary_response.choices[0].message.content}"
                    
                    # Also include a few raw reviews for reference
                    formatted_reviews = self.restaurant_api.format_reviews(reviews, max_reviews=3)
                    review_summary += f"\n\n{formatted_reviews}"
                else:
                    review_summary = "\n\nNo reviews available for this restaurant."
                
                # Combine information and review summary
                response = f"{formatted_info}{review_summary}"
                
                # If there are more results, mention them
                if len(restaurants) > 1:
                    response += f"\n\nI found {len(restaurants)} restaurants matching your query. This is the top result."
                
                return response
                
            except Exception as e:
                # If there's an error with the restaurant API, fall back to Mistral
                print(f"Error using restaurant API: {str(e)}")
                # Continue to use Mistral
        
        # Default: Use Mistral for non-restaurant queries or if restaurant API fails
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        response = await self.client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
        )

        return response.choices[0].message.content
