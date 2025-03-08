import os
import re
from mistralai import Mistral
import discord
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from restaurant_api import RestaurantAPI

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = """You are a helpful assistant with access to tools for finding restaurant information. 

When users ask about restaurants, food places, dining options, or specific eateries, use the search_restaurants tool to gather information. This includes queries like:
- Finding restaurants in specific locations
- Getting information about specific restaurants
- Looking for particular cuisines
- Finding dining options in an area
- Getting reviews or ratings for restaurants

For any restaurant-related query, ALWAYS use the search_restaurants tool to get accurate, up-to-date information.
For non-restaurant queries, respond normally without using the tool.

Remember to extract both the restaurant type/name and location (if provided) from the user's query when using the tool."""

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
        self.restaurant_api = RestaurantAPI()
        self.client = Mistral(api_key=MISTRAL_API_KEY)
        
        # Initialize the LangChain Mistral chat model
        self.llm = ChatMistralAI(
            api_key=MISTRAL_API_KEY,
            model=MISTRAL_MODEL
        )
        
        # Define tools using the @tool decorator
        @tool
        def search_restaurants(query: str, location: str = None) -> str:
            """Use this tool for ANY restaurant-related queries. This includes:
            - Finding restaurants by cuisine, name, or type
            - Getting restaurant information and reviews
            - Finding dining options in specific locations
            
            Args:
                query: Type of restaurant, cuisine, or specific restaurant name (e.g., 'Italian', 'pizza', 'Blue Bottle Coffee')
                location: Location to search in (e.g., 'San Francisco', 'Stanford', 'Palo Alto'). If not provided, will search generally.
            
            Returns:
                Concise information about multiple restaurants including basic info and review summaries.
            """
            # Search for restaurants
            restaurants = self.restaurant_api.search_restaurant(query, location)
            if not restaurants:
                return f"No restaurants found matching '{query}'" + (f" in {location}" if location else "")
            
            response_parts = ["Here are the top recommendations based on your request:"]
            
            # Process up to 3 restaurants
            num_restaurants = min(3, len(restaurants))
            
            for i in range(num_restaurants):
                restaurant = restaurants[i]
                place_id = restaurant['place_id']
                
                # Add a separator between restaurants
                if i > 0:
                    response_parts.append("\n" + "-"*30 + "\n")
                
                # Get details and reviews
                restaurant_details = self.restaurant_api.get_restaurant_details(place_id)
                reviews = self.restaurant_api.get_restaurant_reviews(place_id)
                
                # Format basic restaurant information (name, rating, address, price)
                info = []
                info.append(f"**{restaurant_details.get('name', 'Unknown Restaurant')}**")
                
                rating_parts = []
                if 'rating' in restaurant_details:
                    stars = "⭐" * int(restaurant_details.get('rating', 0))
                    rating_parts.append(f"{restaurant_details.get('rating')} {stars}")
                if 'price_level' in restaurant_details:
                    price_level = "💰" * restaurant_details.get('price_level', 0)
                    rating_parts.append(price_level)
                if rating_parts:
                    info.append(" | ".join(rating_parts))
                
                if 'formatted_address' in restaurant_details:
                    info.append(f"📍 {restaurant_details.get('formatted_address')}")
                
                response_parts.append("\n".join(info))
                
                # Generate AI summary of reviews if available
                if reviews:
                    reviews_text = self.restaurant_api.prepare_reviews_for_summary(reviews)
                    
                    # Generate summary using Mistral AI
                    summary_messages = [
                        {"role": "system", "content": REVIEW_SUMMARY_PROMPT},
                        {"role": "user", "content": reviews_text},
                    ]
                    
                    summary_response = self.client.chat.complete(
                        model=MISTRAL_MODEL,
                        messages=summary_messages,
                    )
                    
                    # Truncate review summary if too long
                    summary = summary_response.choices[0].message.content
                    if len(summary) > 300:
                        summary = summary[:297] + "..."
                    
                    response_parts.append(f"\n💬 {summary}")
                else:
                    response_parts.append("\nNo reviews available.")
            
            # Add information about additional results if any
            remaining_count = len(restaurants) - num_restaurants
            if remaining_count > 0:
                response_parts.append(f"\n\nThere are {remaining_count} more restaurants matching your query. Would you like to see more options?")
            
            return "\n".join(response_parts)

        #TODO: Add more tools when we merge code into main
        # Bind the tools to the LLM
        self.tools = [search_restaurants]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def run(self, message: discord.Message):
        try:
            content = message.content.strip()
            print(f"Processing message: {content}")  # Debug logging
            
            # Check if it's a restaurant query
            if any(keyword in content.lower() for keyword in ['restaurant', 'food', 'eat', 'dining']):
                print("Detected restaurant query, using search_restaurants tool directly")  # Debug logging
                
                # Parse query and location from the message
                query = None
                location = None
                
                # Extract location if "near", "in", or "at" is in the message
                location_matches = re.findall(r'(?:near|in|at)\s+([^,\.!?]+)', content, re.IGNORECASE)
                if location_matches:
                    location = location_matches[0].strip()
                
                # Extract query (type of restaurant or cuisine)
                if 'italian' in content.lower():
                    query = 'Italian'
                elif 'chinese' in content.lower():
                    query = 'Chinese'
                elif 'japanese' in content.lower():
                    query = 'Japanese'
                # Add more cuisine types as needed
                
                # If no specific cuisine was found but "restaurant" is in the query
                if not query and 'restaurant' in content.lower():
                    # Extract words before "restaurant"
                    matches = re.findall(r'(\w+)\s+restaurant', content, re.IGNORECASE)
                    if matches:
                        query = matches[0]
                
                # If still no query found, use a generic term
                if not query:
                    query = "restaurant"
                
                print(f"Extracted query: {query}, location: {location}")  # Debug logging
                
                # Directly invoke the search_restaurants tool
                tool_output = self.tools[0].invoke({
                    "query": query,
                    "location": location
                })
                
                # Split response into chunks if it's too long
                if len(tool_output) > 1900:  # Leave room for formatting
                    chunks = []
                    current_chunk = []
                    current_length = 0
                    
                    # Split by restaurant entries (separated by our delimiter)
                    entries = tool_output.split("\n" + "-"*30 + "\n")
                    
                    for i, entry in enumerate(entries):
                        # Add header to first chunk
                        if i == 0 and entry.startswith("Here are"):
                            current_chunk.append(entry)
                            current_length = len(entry)
                            continue
                            
                        entry_length = len(entry)
                        if current_length + entry_length + 2 <= 1900:  # +2 for newlines
                            current_chunk.append(entry)
                            current_length += entry_length + 2
                        else:
                            # Finalize current chunk
                            chunks.append("\n\n".join(current_chunk))
                            # Start new chunk
                            current_chunk = [entry]
                            current_length = entry_length
                    
                    # Add any remaining content
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                    
                    # Send each chunk as a separate message
                    first_chunk = True
                    for chunk in chunks:
                        if first_chunk:
                            await message.reply(chunk)
                            first_chunk = False
                        else:
                            await message.channel.send(chunk)
                    return None  # We've handled the sending ourselves
                
                return tool_output
            
            # If not a restaurant query, proceed with normal LangChain processing
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=content)
            ]
            
            # Get the AI response with potential tool calls
            ai_msg = await self.llm_with_tools.ainvoke(messages)
            messages.append(ai_msg)
            
            # Handle any tool calls
            if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call["name"].lower()
                    tool_args = tool_call["args"]
                    
                    # Execute the search_restaurants tool
                    if tool_name == "search_restaurants":
                        tool_output = self.tools[0].invoke(tool_args)
                        
                        # Handle long responses
                        if len(tool_output) > 1900:
                            chunks = []
                            current_chunk = []
                            current_length = 0
                            
                            entries = tool_output.split("\n" + "-"*30 + "\n")
                            
                            for i, entry in enumerate(entries):
                                if i == 0 and entry.startswith("Here are"):
                                    current_chunk.append(entry)
                                    current_length = len(entry)
                                    continue
                                    
                                entry_length = len(entry)
                                if current_length + entry_length + 2 <= 1900:
                                    current_chunk.append(entry)
                                    current_length += entry_length + 2
                                else:
                                    chunks.append("\n\n".join(current_chunk))
                                    current_chunk = [entry]
                                    current_length = entry_length
                            
                            if current_chunk:
                                chunks.append("\n\n".join(current_chunk))
                            
                            first_chunk = True
                            for chunk in chunks:
                                if first_chunk:
                                    await message.reply(chunk)
                                    first_chunk = False
                                else:
                                    await message.channel.send(chunk)
                            return None
                    
                    # Add the tool response to messages
                    messages.append(ToolMessage(
                        content=tool_output,
                        tool_call_id=tool_call["id"]
                    ))
                
                # Get final response incorporating tool outputs
                final_response = await self.llm_with_tools.ainvoke(messages)
                return final_response.content
            
            return ai_msg.content

        except Exception as e:
            print(f"Error in run method: {str(e)}")  # Debug logging
            # Fallback to regular chat completion
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message.content},
            ]
            response = await self.client.chat.complete_async(
                model=MISTRAL_MODEL,
                messages=messages,
            )
            return response.choices[0].message.content
