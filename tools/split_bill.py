import os
import base64
from typing import Dict
from PIL import Image
import ast
import io
from dotenv import load_dotenv
from prompts.split_bill_prompts import INITIAL_PROMPT, FINAL_PROMPT
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class SplitBill():
    """
    A class to process and split a bill based on a user prompt and an image of the receipt.
    """

    def __init__(self):
        """
        Initializes the SplitBill class with an OpenAI-powered agent for processing.
        """
        self.agent = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, model_name="gpt-4o")

    async def run(self, user_prompt: str, image: Image.Image) -> str:
        """
        Processes the given receipt image and user prompt to return a formatted bill split.

        Args:
            user_prompt (str): The user's description of how to split the bill.
            image (Image.Image): The image of the receipt.

        Returns:
            str: A formatted response indicating how the bill is split.
        """
        # Extract text from image using OpenAI Vision API
        image_raw_text = await self.__image_to_raw_text(image)

        # Construct the prompt using system instructions, user prompt, and OCR output
        prompt = f"{INITIAL_PROMPT} {user_prompt} {image_raw_text}"
        initial_response_text = await self.agent.ainvoke(prompt)

        # Parse response to extract the breakdown
        breakdown_dict = self.__raw_text_to_breakdown(
            initial_response_text.content)
        split = self.__perform_split(breakdown_dict)

        # Ask LLM to generate a nicely formatted answer to the user
        final_prompt = f"""
            {FINAL_PROMPT} 
            Breakdown: 
            {initial_response_text.content} 
            Split: 
            {split}
        """
        final_response_text = await self.agent.ainvoke(final_prompt)
        return final_response_text.content

    async def __image_to_raw_text(self, image: Image.Image) -> str:
        """
        Extracts raw text from the given receipt image using OpenAI's vision capabilities.

        Args:
            image (Image.Image): The receipt image.

        Returns:
            str: The extracted text from the image.
        """
        buffered = io.BytesIO()
        image.save(buffered, format='JPEG')
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        response = await self.agent.ainvoke([
            HumanMessage(content=[
                {"type": "text", "text": "Extract the text from this receipt image."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"}}
            ])
        ])
        return response.content

    def __raw_text_to_breakdown(self, raw_text: str) -> Dict[str, any]:
        """
        Parses the raw text into a structured breakdown using `ast.literal_eval`.

        Args:
            raw_text (str): The raw response text containing breakdown details.

        Returns:
            Dict[str, any]: The parsed breakdown of the receipt items.
        """
        return ast.literal_eval(raw_text)

    def __perform_split(self, breakdown: Dict[str, any]) -> Dict[str, float]:
        """
        Computes the bill split among individuals based on the parsed breakdown.

        Args:
            breakdown (Dict[str, any]): The structured breakdown of receipt items.

        Returns:
            Dict[str, float]: A dictionary mapping names to the amount each person owes.
        """
        persons = [key for key in breakdown if key not in [
            "<tax>", "<tip>", "<total>"]]

        subtotals = {}
        overall_subtotal = 0.0
        for name in persons:
            person_total = sum(price for _, price in breakdown[name])
            subtotals[name] = person_total
            overall_subtotal += person_total

        tax = breakdown.get("<tax>", 0)
        tip = breakdown.get("<tip>", 0)

        split = {}
        for name in persons:
            person_subtotal = subtotals[name]
            ratio = person_subtotal / overall_subtotal if overall_subtotal else 0
            person_tax = tax * ratio
            person_tip = tip * ratio
            split[name] = round(person_subtotal + person_tax + person_tip, 2)

        return split


if __name__ == "__main__":
    import asyncio

    async def main():
        """
        Test function to run the bill splitting functionality using sample receipt images.
        """
        test_bills = [
            {
                "filename": "data/test_images/img1.jpeg",
                "prompt": "Split this bill. Sherry got the queijo quente and I got the rest"
            },
            {
                "filename": "data/test_images/img2.jpeg",
                "prompt": "I got the Fish and Chips, Kohi the bread, Liz the coke and we all shared the fries."
            },
            {
                "filename": "data/test_images/img3.jpeg",
                "prompt": "John got the 2 bw rolls and a facial tissue, I got all the rest."
            }
        ]
        bill = test_bills[2]
        split_bill = SplitBill()
        result = await split_bill.run(user_prompt=bill["prompt"], image=Image.open(bill["filename"]))
        print(result)

    asyncio.run(main())
