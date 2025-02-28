from types import List
from agent import MistralAgent
from PIL import Image
import pytesseract


class SplitBill(MistralAgent):
    def __init__(self):
        MistralAgent.__init__(self)

    # Public Methods
    def run(self, user_prompt: str, image: Image.Image) -> str:
        image_raw_text = self.__image_to_raw_text(image)

        
        final_breakdown = ""
        return final_breakdown

    # Private Methods
    def __image_to_raw_text(self, image: Image.Image) -> str:
        return pytesseract.image_to_string(image)

    def __raw_text_to_breakdown(self) -> List[tuple[str, int]]:
        pass

    def __verify_breakdown(self) -> bool:
        pass


if __name__ == "__main__":
    # Open Image
    filename = "data/test_images/img1.jpeg"
    image = Image.open(filename)

    # Create user prompt
    prompt = "Split this bill. Sherry got the queijo quente and I got the rest"

    # Get breakdown
    split_bill = SplitBill()
    print(split_bill.run(user_prompt=prompt, image=image))
