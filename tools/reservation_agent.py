import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, Optional
from dataclasses import asdict

import discord
from mistralai import Mistral
from openai import AsyncOpenAI
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
import httpx
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr


from tools.prompts.reservation_prompts import (
    ReservationDetails,
    get_extract_reservation_details_prompt,
    get_restaurant_conversation_prompt,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MISTRAL_MODEL = "mistral-large-latest"
OPENAI_REALTIME_MODEL = "gpt-4o-realtime-preview"


class TwilioReservationAgent:
    def __init__(self):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        if not all(
            [self.account_sid, self.auth_token, self.twilio_number, OPENAI_API_KEY]
        ):
            raise ValueError(
                "Missing required Twilio and OpenAI credentials in environment variables"
            )

        self.client = Client(self.account_sid, self.auth_token)
        self._mistal_client = Mistral(api_key=MISTRAL_API_KEY)
        self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.active_conversations = {}
        self.conversations = {}

        try:
            with open("webhook_url.txt", "r") as f:
                self.webhook_base_url = f.read().strip()
        except FileNotFoundError:
            raise ValueError(
                "Webhook URL file not found - must run reservation_server.py first"
            )

    def _format_phone_number(self, phone: str) -> str:
        """Formats phone number to E.164 format, assuming US if no country code"""
        # Remove any non-digit characters
        digits = re.sub(r"\D", "", phone)

        # If number starts with '+', keep as is
        if phone.startswith("+"):
            return f"+{digits}"

        # For US numbers without country code
        if len(digits) == 10:
            return f"+1{digits}"
        elif len(digits) == 11 and digits.startswith("1"):
            return f"+{digits}"

        raise ValueError(
            "Invalid phone number format. Please provide a 10-digit US number or international number with country code."
        )

    def datetime_to_str(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    def reservation_to_dict(self, reservation: ReservationDetails) -> dict:
        data = asdict(reservation)
        # Convert datetime to string
        if data.get("reservation_time"):
            data["reservation_time"] = self.datetime_to_str(data["reservation_time"])
        return data

    async def handle_conversation(
        self,
        reservation: ReservationDetails,
        speech_input: str = None,
        is_initial: bool = False,
    ) -> str:
        """Handle all conversation with OpenAI, both initial greeting and ongoing"""
        try:
            logger.info("Starting OpenAI conversation")
            if is_initial:
                logger.info("Generating initial greeting")
            else:
                logger.info(f"Processing restaurant response: '{speech_input}'")

            async with self.openai_client.beta.realtime.connect(
                model=OPENAI_REALTIME_MODEL
            ) as connection:
                await connection.session.update(session={"modalities": ["text"]})

                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": get_restaurant_conversation_prompt(
                                    reservation, is_initial
                                ),
                            }
                        ],
                    }
                )

                # Add user message if this isn't the initial greeting
                if not is_initial:
                    await connection.conversation.item.create(
                        item={
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "\n".join(reservation.chat_history),
                                }
                            ],
                        }
                    )
                else:
                    await connection.conversation.item.create(
                        item={
                            "type": "message",
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "Start the conversation with your initial greeting.",
                                }
                            ],
                        }
                    )

                await connection.response.create()

                full_response = []
                async for event in connection:
                    if event.type == "response.text.delta":
                        full_response.append(event.delta)
                        print(event.delta, end="", flush=True)
                    elif event.type == "response.done":
                        print()
                        break

                ai_response = "".join(full_response)
                logger.info(f"AI response: '{ai_response}'")

                reservation.chat_history.append("AI Assistant: " + ai_response)

                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{self.webhook_base_url}/set_reservation",
                        json=self.reservation_to_dict(reservation),
                    )

                # Generate TwiML with the AI response
                response = VoiceResponse()
                gather = response.gather(
                    input="speech",
                    action=f"{self.webhook_base_url}/gather",
                    method="POST",
                    language="en-US",
                    speechTimeout="auto",
                )
                gather.say(ai_response, voice="woman")

                return str(response)

        except Exception as e:
            logger.error(f"Error in OpenAI conversation: {str(e)}", exc_info=True)
            response = VoiceResponse()
            response.say(
                "I apologize for the technical difficulty. Could you please repeat that?",
                voice="woman",
            )
            return str(response)

    async def handle_restaurant_response(
        self, speech_input: str, reservation: ReservationDetails
    ) -> str:
        """Handle ongoing conversation"""
        return await self.handle_conversation(
            reservation, speech_input, is_initial=False
        )

    async def parse_reservation_request(
        self, message: str
    ) -> tuple[bool, Optional[ReservationDetails], Optional[str]]:
        """
        Returns (is_complete, reservation_details, missing_fields_message)
        """
        response = await self._mistal_client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": get_extract_reservation_details_prompt()},
                {"role": "user", "content": f"Message: {message}\nOutput:"},
            ],
            response_format={"type": "json_object"},
        )

        try:
            result = json.loads(response.choices[0].message.content)

            if not result["complete"]:
                return (
                    False,
                    None,
                    result.get(
                        "error_message",
                        "Please provide all the required information for the reservation.",
                    ),
                )

            details = result["details"]
            try:
                # Format phone number, assuming US if no country code
                formatted_phone = self._format_phone_number(details["phone_number"])

                reservation = ReservationDetails(
                    restaurant_phone=formatted_phone,
                    party_size=details["party_size"],
                    reservation_time=datetime.fromisoformat(
                        details["reservation_time"]
                    ),
                    customer_name=details["customer_name"],
                    special_requests=details["special_requests"],
                    chat_history=[],
                )
                return True, reservation, None

            except ValueError as e:
                return False, None, str(e)

        except Exception as e:
            return (
                False,
                None,
                "I couldn't process that request. Please provide the restaurant's phone number (10 digits for US or with country code), party size, time, and name for the reservation.",
            )

    async def make_reservation_call(self, reservation: ReservationDetails) -> str:
        """Make the initial call to the restaurant"""
        logger.info("Starting reservation call")
        try:
            formatted_restaurant_phone = self._format_phone_number(
                reservation.restaurant_phone
            )
            logger.info(f"Formatted phone number: {formatted_restaurant_phone}")

            # Generate initial TwiML
            twiml = await self.handle_conversation(reservation, is_initial=True)
            logger.info(f"Generated TwiML: {twiml}")

            # Make the call
            call = self.client.calls.create(
                twiml=twiml,
                to=formatted_restaurant_phone,
                from_=self.twilio_number,
                record=True,
            )
            logger.info(f"Call created with SID: {call.sid}")

            return (
                f"✓ Starting call with {formatted_restaurant_phone} for your reservation.\n"
                f"I'll handle the conversation and update you on the result.\n"
                f"Reservation details:\n"
                f"- Party size: {reservation.party_size}\n"
                f"- Time: {reservation.reservation_time.strftime('%I:%M %p on %A, %B %d')}\n"
                f"- Name: {reservation.customer_name}\n"
                f"- Special requests: {reservation.special_requests or 'None'}"
            )

        except Exception as e:
            logger.error(f"Error making reservation call: {str(e)}", exc_info=True)
            return f"❌ An unexpected error occurred: {str(e)}"


class RestaurantAgent(BaseTool):
    name: str = "restaurant_reservation"
    description: str = "A tool for making restaurant reservations through phone calls"

    _mistal_client: Mistral = PrivateAttr()
    reservation_agent: TwilioReservationAgent = None
    active_conversations: Dict[str, ReservationDetails] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self._mistal_client = Mistral(api_key=MISTRAL_API_KEY)
        self.reservation_agent = TwilioReservationAgent()
        self.active_conversations = {}

    async def run(self, message: discord.Message):
        if "make a reservation" in message.content.lower():
            is_complete, details, error_msg = (
                await self.reservation_agent.parse_reservation_request(message.content)
            )

            if not is_complete:
                return error_msg

            return await self.reservation_agent.make_reservation_call(details)

    async def _arun(self, message: str):
        return await self.run(message)

    def _run(self, message: str):
        """
        Synchronous version of the reservation agent.
        This would need asyncio.run() or similar to execute the async version.
        """
        raise NotImplementedError("Please use the async version of this tool")
