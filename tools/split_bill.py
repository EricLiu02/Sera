import ast
import asyncio
import base64
import io
import os
from typing import Dict, Optional

import discord
from dotenv import load_dotenv
from langchain_core.tools import BaseTool, ToolException
from langchain_core.tools.base import ArgsSchema
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from PIL import Image
from pydantic import BaseModel, Field, PrivateAttr

from tools.prompts.split_bill_prompts import INITIAL_PROMPT, FINAL_PROMPT

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class SplitBillInput(BaseModel):
    user_instructions: str = Field(
        description="Instructions the user provided in their message in order to split the bill"
    )
    bill: str = Field(description="The full bill transcription")


class SplitBill(BaseTool):
    """
    A class to process and split a bill based on a user prompt and an image of the receipt.
    """

    _agent: ChatOpenAI = PrivateAttr()
    name: str = "split_bill"
    description: str = (
        "Useful for when the user wants to split the bill/check for the restaurant."
    )
    args_schema: Optional[ArgsSchema] = SplitBillInput
    return_direct: bool = True

    def __init__(self, **data):
        """
        Initializes the SplitBill class with an OpenAI-powered agent for processing.
        """
        super().__init__(**data)
        self._agent = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, model_name="gpt-4o")

    async def _arun(self, user_instructions: str, bill: str) -> str:
        """
        Processes the given receipt image and user prompt to return a formatted bill split.

        Args:
            user_prompt (str): The user's description of how to split the bill.
            bill (str): The image of the receipt.

        Returns:
            str: A formatted response indicating how the bill is split.
        """
        # Construct the prompt using system instructions, user prompt, and OCR output
        prompt = f"{INITIAL_PROMPT} {user_instructions} The following is a transcription of the bill. Please consider all the items: {bill}"
        initial_response_text = await self._agent.ainvoke(prompt)
        print(bill)
        # Parse response to extract the breakdown
        breakdown_dict = self.__raw_text_to_breakdown(
            initial_response_text.content)

        print(breakdown_dict)

        split = self.__perform_split(breakdown_dict)

        print(split)

        # Ask LLM to generate a nicely formatted answer to the user
        final_prompt = f"""
            {FINAL_PROMPT} 
            Breakdown: 
            {initial_response_text.content} 
            Split: 
            {split}
        """
        final_response_text = await self._agent.ainvoke(final_prompt)
        return final_response_text.content

    def _run(self, user_instructions: str, image: str) -> str:
        """
        Synchronous version of the bill splitting functionality.
        This would need asyncio.run() or similar to execute the async version.
        """
        return asyncio.run(self._arun(user_instructions, image))

    def __raw_text_to_breakdown(self, raw_text: str) -> Dict[str, any]:
        """
        Parses the raw text into a structured breakdown using `ast.literal_eval`.

        Args:
            raw_text (str): The raw response text containing breakdown details.

        Returns:
            Dict[str, any]: The parsed breakdown of the receipt items.
        """
        try:
            return ast.literal_eval(raw_text)
        except (SyntaxError, ValueError) as e:
            raise ToolException(f"Failed to parse breakdown: {str(e)}")

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

        # Verify that the computed splits add up to the reported total (if provided)
        computed_total = round(sum(split.values()), 2)
        if "<total>" in breakdown:
            reported_total = round(breakdown["<total>"], 2)
            difference = round(reported_total - computed_total, 2)
            if abs(difference) >= 0.01:
                # Adjust the person with the highest share to account for the rounding difference
                largest_person = max(split, key=split.get)
                split[largest_person] = round(
                    split[largest_person] + difference, 2)
        return split


async def get_image_text(message: discord.Message):
    if not message.attachments or "image" not in message.attachments[0].content_type:
        return ""

    image_url = message.attachments[0].url
    gpt = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o")

    try:
        response = await gpt.ainvoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": """
                                You are a helpful agent that transcribes the text from bill images.
                                Extract the text from this bill image. 
                                Don't send anything else other than the extracted text. 
                                This is a bill, so make sure the values add up.""",
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]
                )
            ]
        )
        return "Here is a transcription of the bill: " + response.content
    except (SyntaxError, ValueError) as e:
        return ""


async def test_split_bill():
    """
    Test function to run the bill splitting functionality using sample receipt images.
    """
    test_bills = [
        {
            "filename": "data/test_images/img1.jpeg",
            "prompt": "Split this bill. Sherry got the queijo quente and I got the rest",
        },
        {
            "filename": "data/test_images/img2.jpeg",
            "prompt": "I got the Fish and Chips, Kohi the bread, Liz the coke and we all shared the fries.",
        },
        {
            "filename": "data/test_images/img3.jpeg",
            "prompt": "John got the 2 bw rolls and a facial tissue, I got all the rest.",
        },
    ]
    bill = test_bills[2]
    split_bill = SplitBill()

    # Convert image
    image = Image.open(bill["filename"])
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Call function
    result = await split_bill.ainvoke(
        user_instructions=bill["prompt"], image=img_base64
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(test_split_bill())
