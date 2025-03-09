INITIAL_PROMPT = """
Below is a transcription from a restaurant bill. Parse it into the following Python dict format. 

Response Format Example:
{ 
   "Sherry": [("Burguer", 11.25), ("Cheese", 19.75)],
   "Rishi": [("Burguer", 11.25), ("Egg Roll", 8.9)],
   "<tax>": 10.3
   "<tip>": 11.0
   "<total>": 72.45
}

"Sherry" and "Rishi" refer to the names of the group members, and <tax>, <tip> and <total> should always be in
the dict. In the example, the there was only one burguer that was shared between Sherry and Rishi, you should make
this split if the user asks to. The names "Rishi" and "Sherry" are just examples. Please use the actual names of the 
members below. If they say "I", just use "You."

DO NOT include any extra text—only output the list.
DO NOT include anything else (such as ```python), as your response should be parsable by python's ast.

This is how it should be split, and the bill comes right after:
"""

FINAL_PROMPT = """
Generate a concise, human-readable breakdown in clear and formatted text.

Example Output:
**Sherry** pays **$13** for Burguer, Cheese, and shared tax/tip.  
**You** pay **$48** for Burguer, Egg Roll, and shared tax/tip.  
**Total Bill:** *$31.00*.
Follow this format strictly. Do not add explanations or extra text.
"""
