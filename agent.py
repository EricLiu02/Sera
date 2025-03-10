import os
import re
from mistralai import Mistral
import discord
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from utils import extract_image_base64

# Import Tools
from tools.search_restaurants import SearchRestaurantsTool
from tools.split_bill import SplitBill
from tools.reservation_agent import ReservationAgent

from typing import Dict, List, Optional

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = """You are a helpful assistant with access to a search_restaurants tool for finding restaurant information.

Your primary task is to help users find and learn about restaurants. When users ask about restaurants, whether directly or indirectly, use the search_restaurants tool to provide accurate information.

Examples of when to use the tool:
1. Direct restaurant queries:
   - "Find Italian restaurants in San Francisco"
   - "What are some good restaurants near Stanford?"
   - "Show me Chinese food places in Palo Alto"

2. Indirect restaurant queries:
   - "Where can I get sushi around here?"
   - "I'm hungry for Mexican food"
   - "What's a good place to eat near downtown?"

3. Specific restaurant inquiries:
   - "Tell me about Blue Bottle Coffee"
   - "What are the reviews like for Pizzeria Delfina?"

4. Pagination requests:
   - When users say "yes", "show more", "more options", or similar phrases in response to being asked if they want to see more restaurants
   - In these cases, use the search_restaurants tool with the same query and location from the context, and the appropriate start_index
   - The context will be provided as a dictionary with 'type': 'search_context', 'query', 'location', and 'results_shown' fields

When using the tool, extract:
- query: The type of restaurant, cuisine, or specific restaurant name
- location: The area to search in (if provided)
- start_index: For pagination requests, use the results_shown value from the context

For non-restaurant queries, respond normally without using the tool.

Remember to maintain context for pagination when users ask to see more options.

If the user asks to split a bill, use the split_bill tool. Assume that the image of the bill will be
provided to the tool"""

REVIEW_SUMMARY_PROMPT = """You are a helpful assistant specializing in summarizing restaurant reviews.
Provide a very concise 2-3 sentence summary that captures:
1. Overall sentiment and most mentioned positives
2. Any notable criticisms or areas for improvement
3. 1-2 most recommended dishes (if mentioned)

Keep your summary brief, direct, and informative."""


class MistralAgent:
    def __init__(self):
        # Initialize the LangChain Mistral chat model
        self.llm = ChatMistralAI(api_key=MISTRAL_API_KEY, model=MISTRAL_MODEL)
        
        # Initialize restaurant API for pagination context
        self.restaurant_api = SearchRestaurantsTool()

        # State tracking for restaurant queries
        # channel_id -> query info
        self.last_restaurant_query: Dict[int, Dict] = {}

        # Initialize tools
        self.tools = [SearchRestaurantsTool(), SplitBill(), ReservationAgent()]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def run(self, message: discord.Message):
        # Create messages with context about previous searches if available
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Add context about previous search if it exists
        last_query = self.last_restaurant_query.get(message.channel.id)
        if last_query:
            context = {
                "type": "search_context",
                "query": last_query["query"],
                "location": last_query["location"],
                "results_shown": last_query["last_index"] + 3,
                "total_results": len(
                    self.restaurant_api.search_restaurant(
                        last_query["query"], last_query["location"]
                    )
                ),
            }
            messages.append(SystemMessage(content=str(context)))

        messages.append(HumanMessage(content=message.content))

        # Let LangChain handle all the query interpretation
        ai_msg = await self.llm_with_tools.ainvoke(messages)
        if not ai_msg:
            return "I apologize, but I couldn't process that request. Could you please try rephrasing it?"

        messages.append(ai_msg)

        # Check for tool calls first
        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"].lower()
                tool_args = tool_call["args"]

                if tool_name == "split_bill":
                    try:
                        base64_image = await extract_image_base64(message)
                    except Exception as e:
                        return str(e)

                    # Use provided user instructions if available in tool_args, else fall back to message content
                    user_instructions = tool_args.get(
                        "user_instructions", message.content
                    )

                    split_bill_tool = next(
                        (t for t in self.tools if t.name.lower() == "split_bill"),
                        None,
                    )

                    return await split_bill_tool.ainvoke(
                        {
                            "user_instructions": user_instructions,
                            "image": base64_image,
                        }
                    )
                # Execute the search_restaurants tool
                if tool_name == "search_restaurants":
                    # Validate tool arguments
                    if not tool_args.get("query") and (
                        not last_query or not last_query.get("query")
                    ):
                        return "I need to know what kind of restaurant you're looking for. Could you please specify?"

                    # Store or update query information for pagination
                    self.last_restaurant_query[message.channel.id] = {
                        "query": tool_args.get(
                            "query", last_query["query"] if last_query else None
                        ),
                        "location": tool_args.get(
                            "location",
                            last_query["location"] if last_query else None,
                        ),
                        "last_index": tool_args.get("start_index", 0),
                    }

                    # try:
                    # Get restaurant recommendations
                    tool_output = self.tools[0].invoke(tool_args)
                    if not tool_output:
                        return "I couldn't find any restaurants matching your criteria. Would you like to try a different search?"

                    # If the response is too long, truncate each restaurant's review summary
                    if len(tool_output) > 1900:
                        entries = tool_output.split("\n" + "-" * 30 + "\n")

                        # Keep the header
                        formatted_entries = [entries[0]]

                        # Process each restaurant entry
                        for entry in entries[1:]:
                            # Find the review section (after the 💬 emoji)
                            parts = entry.split("\n💬 ", 1)
                            if len(parts) > 1:
                                # Keep the restaurant info and truncate the review
                                restaurant_info = parts[0]
                                review = parts[1]
                                truncated_review = (
                                    review[:200] + "..."
                                    if len(review) > 200
                                    else review
                                )
                                formatted_entries.append(
                                    f"{restaurant_info}\n💬 {truncated_review}"
                                )
                            else:
                                formatted_entries.append(entry)

                        # Join everything back together with separators
                        tool_output = "\n" + "-" * 30 + "\n".join(formatted_entries)

                        # If still too long, truncate the whole message
                        if len(tool_output) > 1900:
                            tool_output = (
                                tool_output[:1850]
                                + "\n\n[Some content truncated due to length]"
                            )

                    return tool_output

                    # except Exception as e:
                    #     print(f"Error processing restaurant search: {repr(e)}")
                    #     return "I encountered an error while searching for restaurants. Would you like to try again?"

                if tool_name == "make_restaurant_reservation":
                    return await self.tools[2].ainvoke(
                        {
                            "message": message,
                        }
                    )

        # If no tool calls, check for content in AI response
        if not ai_msg.content or not ai_msg.content.strip():
            return "I'm not sure how to help with that. Could you please rephrase your question?"

        return ai_msg.content

        # except Exception as e:
        #     print(f"Error in run method: {str(e)}")  # Debug logging
        #     return "I'm having trouble processing your request. Could you please try again?"
