from typing import Dict
from pydantic import BaseModel, PrivateAttr
from langchain.tools import BaseTool
from mistralai.client import Mistral
from twilio_reservation_agent import TwilioReservationAgent
from reservation_details import ReservationDetails
import os


class RestaurantAgent(BaseTool):
    name: str = "restaurant_reservation"
    description: str = "A tool for making restaurant reservations through phone calls"

    _mistal_client: Mistral = PrivateAttr()
    reservation_agent: TwilioReservationAgent = PrivateAttr()
    active_conversations: Dict[str, ReservationDetails] = PrivateAttr()

    def __init__(self):
        super().__init__()
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self._mistal_client = Mistral(api_key=MISTRAL_API_KEY)
        self.reservation_agent = TwilioReservationAgent()
        self.active_conversations = {}
