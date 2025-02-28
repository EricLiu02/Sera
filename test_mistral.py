import os
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("MISTRAL_API_KEY")
print(f"API key loaded: {api_key[:5]}...{api_key[-5:]}")

# Initialize client
client = Mistral(api_key=api_key)

# Test a simple completion
try:
    response = client.chat.complete(
        model="mistral-tiny",
        messages=[{"role": "user", "content": "Hello, how are you?"}]
    )
    print("Success! Response:", response.choices[0].message.content)
except Exception as e:
    print("Error:", e) 