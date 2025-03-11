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
