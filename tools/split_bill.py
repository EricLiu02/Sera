import os
import base64
from typing import Dict
from PIL import Image
import ast
import io
from dotenv import load_dotenv
from tools.prompts.split_bill_prompts import INITIAL_PROMPT, FINAL_PROMPT
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.tools import BaseTool
from typing import Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun, CallbackManagerForToolRun)
from langchain_core.tools import BaseTool, ToolException
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field, PrivateAttr


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class SplitBillInput(BaseModel):
    user_instructions: str = Field(
        description="Instructions the user provided in their message in order to split the bill")
    image: str = Field(description="Base64 encoded image of the receipt")


class SplitBill(BaseTool):
    """
    A class to process and split a bill based on a user prompt and an image of the receipt.
    """
    _agent: ChatOpenAI = PrivateAttr()
    name: str = "split_bill"
    description: str = "Useful for when the user wants to split the bill/check for the restaurant."
    args_schema: Optional[ArgsSchema] = SplitBillInput
    return_direct: bool = True

    def __init__(self, **data):
        """
        Initializes the SplitBill class with an OpenAI-powered agent for processing.
        """
        super().__init__(**data)
        self._agent = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY, model_name="gpt-4o")

    async def _arun(self, user_instructions: str, image: str) -> str:
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
        prompt = f"{INITIAL_PROMPT} {user_instructions} {image_raw_text}"
        initial_response_text = await self._agent.ainvoke(prompt)

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
        final_response_text = await self._agent.ainvoke(final_prompt)
        return final_response_text.content

    def _run(self, user_instructions: str, image: str) -> str:
        """
        Synchronous version of the bill splitting functionality.
        This would need asyncio.run() or similar to execute the async version.
        """
        raise NotImplementedError("Please use the async version of this tool")

    async def __image_to_raw_text(self, image: str) -> str:
        """
        Extracts raw text from the given receipt image using OpenAI's vision capabilities.

        Args:
            image (Image.Image): The receipt image.

        Returns:
            str: The extracted text from the image.
        """
        image_url = ""
        if not image.startswith('data:image'):
            image_url = f"data:image/jpeg;base64,{image}"
        else:
            image_url = image

        response = await self._agent.ainvoke([
            HumanMessage(content=[
                {"type": "text", "text": "Extract the text from this receipt image."},
                {"type": "image_url", "image_url": {"url": image_url}}
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

        # Convert image
        image = Image.open(bill["filename"])
        buffered = io.BytesIO()
        image.save(buffered, format='JPEG')
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Call function
        result = await split_bill.ainvoke(user_instructions=bill["prompt"], image=img_base64)
        print(result)

    asyncio.run(main())
