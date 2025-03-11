import os
import re
from mistralai import Mistral
import discord
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# Import Tools
from tools.search_restaurants import SearchRestaurantsTool
from tools.split_bill import SplitBill, get_image_text
from tools.restaurant_details import RestaurantDetailsTool
from tools.reservation_agent import ReservationAgent

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MISTRAL_MODEL = "mistral-large-latest"
# SYSTEM_PROMPT = """You are a helpful assistant with access to a search_restaurants tool for finding restaurant information.

# Your primary task is to help users find and learn about restaurants. When users ask about restaurants, whether directly or indirectly, use the search_restaurants tool to provide accurate information.

# Examples of when to use the tool:
# 1. Direct restaurant queries:
#    - "Find Italian restaurants in San Francisco"
#    - "What are some good restaurants near Stanford?"
#    - "Show me Chinese food places in Palo Alto"

# 2. Indirect restaurant queries:
#    - "Where can I get sushi around here?"
#    - "I'm hungry for Mexican food"
#    - "What's a good place to eat near downtown?"

# 3. Specific restaurant inquiries:
#    - "Tell me about Blue Bottle Coffee"
#    - "What are the reviews like for Pizzeria Delfina?"

# 4. Pagination requests:
#    - When users say "yes", "show more", "more options", or similar phrases in response to being asked if they want to see more restaurants
#    - In these cases, use the search_restaurants tool with the same query and location from the context, and the appropriate start_index
#    - The context will be provided as a dictionary with 'type': 'search_context', 'query', 'location', and 'results_shown' fields

# When using the tool, extract:
# - query: The type of restaurant, cuisine, or specific restaurant name
# - location: The area to search in (if provided)
# - start_index: For pagination requests, use the results_shown value from the context

# For non-restaurant queries, respond normally without using the tool.

# Remember to maintain context for pagination when users ask to see more options.

# If the user asks to split a bill, use the split_bill tool. Assume that the image of the bill will be
# provided to the tool"""

SYSTEM_PROMPT = """
You are a helpful assistant that can help users find and learn about restaurants. You have available to you the following tools:
- search_restaurants: Use this to find restaurants and get basic information
- get_restaurant_details: Use this when users want more detailed information about a specific restaurant
- split_bill: Use this to split a bill
- make_restaurant_reservation: Use this to make a restaurant reservation

You can use the search_restaurants tool to find restaurants, the get_restaurant_details tool to get more information about a specific restaurant, the split_bill tool to split a bill, and the make_restaurant_reservation tool to make a restaurant reservation.
The output of each tool is a string, and you should directly return the output of the tool in your response. Do not include any other text in your response or otherwise modify the output of the tool.

You should use the search_restaurants tool to find restaurants when the user asks about a specific restaurant or cuisine, or when they ask for recommendations, or when they ask for a list of restaurants in a specific area, or when they ask for a list of restaurants in a specific category, directly or indirectly.

You should use the get_restaurant_details tool when:
1. A user asks for more information about a specific restaurant they found
2. A user wants to know specific details like opening hours, contact info, or amenities
3. A user wants to verify if a restaurant has certain features (outdoor seating, wheelchair access, etc.)

When using get_restaurant_details:
1. If the user is asking about a restaurant from search results, extract and use the place_id from the hidden comment in the search results
2. If the user is asking about a restaurant directly by name, pass the name to the tool and it will search for it

You should use the split_bill tool to split a bill when the user asks to split a bill.

You should use the make_restaurant_reservation tool to make a restaurant reservation when the user asks to make a restaurant reservation.

You should not use the search_restaurants tool to find restaurants when the user asks to make a restaurant reservation. You should not make a restaurant reservation unless the user explicitly asks to make a restaurant reservation.
Use the tool most appropriate for the user's request. Do not call into tools unless you are sure that the user's request is best handled by that tool.

Make sure to directly return the output of the tool in your response. Do not include any other text in your response or otherwise modify the output of the tool.
If the user's request is not best handled by a tool, respond normally without using a tool. Even though you are a restaurant expert, you can still respond normally without using a tool to other non-restaurant related questions.
Only call into at most one tool per response.
"""


REVIEW_SUMMARY_PROMPT = """You are a helpful assistant specializing in summarizing restaurant reviews.
Provide a very concise 2-3 sentence summary that captures:
1. Overall sentiment and most mentioned positives
2. Any notable criticisms or areas for improvement
3. 1-2 most recommended dishes (if mentioned)

Keep your summary brief, direct, and informative."""


class MistralAgent:
    def __init__(self):
        self.chat_history = [
            SystemMessage(content=SYSTEM_PROMPT),
        ]

        llm = ChatMistralAI(api_key=MISTRAL_API_KEY, model=MISTRAL_MODEL)
        tools = [
            SearchRestaurantsTool(),
            RestaurantDetailsTool(),
            SplitBill(),
            ReservationAgent(),
        ]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt)
        self.agent = AgentExecutor(agent=agent, tools=tools)

    async def run(self, message: discord.Message):
        image_text = await get_image_text(message)
        human_message = f"{message.content} {image_text}"

        self.chat_history.append(HumanMessage(content=human_message))

        output = await self.agent.ainvoke(
            {
                "input": human_message,
                "chat_history": self.chat_history,
            }
        )

        self.chat_history.append(AIMessage(content=output["output"]))
        return output["output"]


# class MistralAgent:
#     def __init__(self):
#         # Initialize the LangChain Mistral chat model
#         self.llm = ChatMistralAI(api_key=MISTRAL_API_KEY, model=MISTRAL_MODEL)

#         # Initialize restaurant API for pagination context
#         self.restaurant_api = SearchRestaurantsTool()

#         # State tracking for restaurant queries
#         # channel_id -> query info
#         self.last_restaurant_query: Dict[int, Dict] = {}

#         # Initialize tools
#         self.tools = [SearchRestaurantsTool(), SplitBill(), ReservationAgent()]
#         self.llm_with_tools = self.llm.bind_tools(self.tools)

#     async def run(self, message: discord.Message):
#         # Create messages with context about previous searches if available
#         messages = [SystemMessage(content=SYSTEM_PROMPT)]

#         # Add context about previous search if it exists
#         last_query = self.last_restaurant_query.get(message.channel.id)
#         if last_query:
#             context = {
#                 "type": "search_context",
#                 "query": last_query["query"],
#                 "location": last_query["location"],
#                 "results_shown": last_query["last_index"] + 3,
#                 "total_results": len(
#                     self.restaurant_api.restaurant_api.search_restaurant(
#                         last_query["query"], last_query["location"]
#                     )
#                 ),
#             }
#             messages.append(SystemMessage(content=str(context)))

#         messages.append(HumanMessage(content=message.content))

#         # Let LangChain handle all the query interpretation
#         ai_msg = await self.llm_with_tools.ainvoke(messages)
#         if not ai_msg:
#             return "I apologize, but I couldn't process that request. Could you please try rephrasing it?"

#         messages.append(ai_msg)

#         # Check for tool calls first
#         if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
#             for tool_call in ai_msg.tool_calls:
#                 tool_name = tool_call["name"].lower()
#                 tool_args = tool_call["args"]

#                 if tool_name == "split_bill":
#                     try:
#                         base64_image = await extract_image_base64(message)
#                     except Exception as e:
#                         return str(e)

#                     # Use provided user instructions if available in tool_args, else fall back to message content
#                     user_instructions = tool_args.get(
#                         "user_instructions", message.content
#                     )

#                     split_bill_tool = next(
#                         (t for t in self.tools if t.name.lower() == "split_bill"),
#                         None,
#                     )

#                     return await split_bill_tool.ainvoke(
#                         {
#                             "user_instructions": user_instructions,
#                             "image": base64_image,
#                         }
#                     )
#                 # Execute the search_restaurants tool
#                 if tool_name == "search_restaurants":
#                     # Validate tool arguments
#                     if not tool_args.get("query") and (
#                         not last_query or not last_query.get("query")
#                     ):
#                         return "I need to know what kind of restaurant you're looking for. Could you please specify?"

#                     # Store or update query information for pagination
#                     self.last_restaurant_query[message.channel.id] = {
#                         "query": tool_args.get(
#                             "query", last_query["query"] if last_query else None
#                         ),
#                         "location": tool_args.get(
#                             "location",
#                             last_query["location"] if last_query else None,
#                         ),
#                         "last_index": tool_args.get("start_index", 0),
#                     }

#                     # try:
#                     # Get restaurant recommendations
#                     tool_output = self.tools[0].invoke(tool_args)
#                     if not tool_output:
#                         return "I couldn't find any restaurants matching your criteria. Would you like to try a different search?"

#                     return tool_output

#                     # except Exception as e:
#                     #     print(f"Error processing restaurant search: {repr(e)}")
#                     #     return "I encountered an error while searching for restaurants. Would you like to try again?"

#                 if tool_name == "make_restaurant_reservation":
#                     return await self.tools[2].ainvoke(
#                         {
#                             "message": message,
#                         }
#                     )

#         # If no tool calls, check for content in AI response
#         if not ai_msg.content or not ai_msg.content.strip():
#             return "I'm not sure how to help with that. Could you please rephrase your question?"

#         return ai_msg.content

#         # except Exception as e:
#         #     print(f"Error in run method: {str(e)}")  # Debug logging
#         #     return "I'm having trouble processing your request. Could you please try again?"
