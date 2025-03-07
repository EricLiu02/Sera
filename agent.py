import os
from mistralai import Mistral
import discord
from twilio.rest import Client
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict
import re
import json
from twilio.twiml.voice_response import VoiceResponse, Connect
from openai import OpenAI, AsyncOpenAI
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass
class ReservationDetails:
    restaurant_phone: str
    party_size: int
    reservation_time: datetime
    customer_name: str
    special_requests: Optional[str] = None


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
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.active_conversations = {}
        self.conversations = {}

        with open("webhook_url.txt", "r") as f:
            self.webhook_base_url = f.read().strip()

    def _format_phone_number(self, phone: str) -> str:
        """Formats phone number to E.164 format"""
        # Remove any non-digit characters
        digits = re.sub(r"\D", "", phone)

        # Add US country code if not present
        if len(digits) == 10:
            digits = f"1{digits}"

        return f"+{digits}"

    def _generate_twiml(self, reservation: ReservationDetails) -> str:
        """Generates TwiML for the automated call"""
        time_str = reservation.reservation_time.strftime("%I:%M %p")
        date_str = reservation.reservation_time.strftime("%A, %B %d")

        special_requests_script = (
            f" We have some special requests: {reservation.special_requests}."
            if reservation.special_requests
            else ""
        )

        return f"""<?xml version="1.0" encoding="UTF-8"?><Response><Say voice="alice">Hello, I'm calling to make a reservation. I would like to make a reservation for {reservation.party_size} people on {date_str} at {time_str}. The reservation will be under the name {reservation.customer_name}.{special_requests_script} Please confirm if this reservation time works.</Say><Pause length="2"/><Record maxLength="30" playBeep="true" transcribe="true"/><Say voice="alice">Thank you for your response. We will confirm the reservation details with the customer. Have a great day!</Say></Response>"""

    def _generate_initial_greeting(self, reservation: ReservationDetails) -> str:
        """Generate the initial greeting TwiML"""
        logger.info("Generating initial greeting TwiML")
        response = VoiceResponse()

        time_str = reservation.reservation_time.strftime("%I:%M %p")
        date_str = reservation.reservation_time.strftime("%A, %B %d")

        # Initial message
        message = (
            f"Hello, I'm an AI assistant calling to make a reservation for "
            f"{reservation.party_size} people on {date_str} at {time_str} "
            f"under the name {reservation.customer_name}. "
            "Please respond with yes if this time works, or no if you need a different time."
        )

        # Simple gather with speech input
        gather = response.gather(
            input="speech", action=f"{self.webhook_base_url}/gather", method="POST"
        )
        gather.say(message, voice="alice")

        twiml = str(response)
        logger.info(f"Generated TwiML: {twiml}")
        return twiml

    async def make_reservation_call(self, reservation: ReservationDetails) -> str:
        """Make the initial call to the restaurant"""
        logger.info(f"Starting reservation call for {reservation.customer_name}")
        try:
            formatted_restaurant_phone = self._format_phone_number(
                reservation.restaurant_phone
            )
            logger.info(f"Calling phone number: {formatted_restaurant_phone}")
            logger.info(f"Using Twilio number: {self.twilio_number}")

            twiml = self._generate_initial_greeting(reservation)
            logger.info(f"Using TwiML: {twiml}")

            # Make the call
            call = self.client.calls.create(
                twiml=twiml, to=formatted_restaurant_phone, from_=self.twilio_number
            )
            logger.info(f"Call created with SID: {call.sid}")

            return (
                f"✓ Starting call to {formatted_restaurant_phone}\n"
                f"Reservation details:\n"
                f"- Party size: {reservation.party_size}\n"
                f"- Time: {reservation.reservation_time.strftime('%I:%M %p on %A, %B %d')}\n"
                f"- Name: {reservation.customer_name}"
            )

        except Exception as e:
            logger.error(f"Error making reservation call: {str(e)}", exc_info=True)
            return f"❌ An unexpected error occurred: {str(e)}"

    def _get_prompt(self) -> str:
        """Returns the prompt with the current date/time context"""
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")

        datetime_context = f"""
    Extract reservation details from the message and convert relative dates/times to absolute dates.
    Current date/time reference: {current_datetime}

    Validate the following:
    1. All required fields are present
    2. Reservation time must be in the future
    3. Phone number must be valid (with country code)
    4. Party size must be a positive number
    """

        prompt_template = """
    Required fields: phone_number, party_size, reservation_time, customer_name
    Optional fields: special_requests

    Handle these date/time formats:
    - Relative: "tomorrow", "next Tuesday", "this Friday"
    - Time: "7pm", "7:30 PM", "19:30"
    - Combined: "tomorrow at 7", "next Friday at 8:30 PM"

    Return in JSON format:
    {
        "complete": true/false,
        "missing_fields": ["field1", "field2"],
        "error_message": "Specific error message if validation fails, null if complete is true",
        "details": {
            "phone_number": "phone number with country code",
            "party_size": number,
            "reservation_time": "YYYY-MM-DD HH:MM",
            "customer_name": "name",
            "special_requests": "requests or null"
        }
    }

    Examples:
    Message: Make a reservation at +1234567890 for 4 people tomorrow at 7:30 PM under John Doe
    Response: {
        "complete": true,
        "missing_fields": [],
        "error_message": null,
        "details": {
            "phone_number": "+1234567890",
            "party_size": 4,
            "reservation_time": "2024-03-08 19:30",
            "customer_name": "John Doe",
            "special_requests": null
        }
    }

    Message: Make a reservation at +1234567890 for 4 people yesterday at 7:30 PM under John Doe
    Response: {
        "complete": false,
        "missing_fields": [],
        "error_message": "Reservation time must be in the future. Please specify a future date and time.",
        "details": {
            "phone_number": "+1234567890",
            "party_size": 4,
            "reservation_time": "2024-03-06 19:30",
            "customer_name": "John Doe",
            "special_requests": null
        }
    }

    Message: Reserve for 3 people today at 6 under Mike
    Response: {
        "complete": false,
        "missing_fields": ["phone_number"],
        "error_message": "Missing restaurant phone number. Please provide a contact number for the restaurant.",
        "details": {
            "phone_number": null,
            "party_size": 3,
            "reservation_time": "2024-03-07 18:00",
            "customer_name": "Mike",
            "special_requests": null
        }
    }
    """
        return datetime_context + prompt_template

    async def parse_reservation_request(
        self, message: str
    ) -> tuple[bool, Optional[ReservationDetails], Optional[str]]:
        """
        Returns (is_complete, reservation_details, missing_fields_message)
        """
        response = await self.mistral_client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": self._get_prompt()},
                {"role": "user", "content": f"Message: {message}\nOutput:"},
            ],
            response_format={"type": "json_object"},
        )

        try:
            result = json.loads(response.choices[0].message.content)

            if not result["complete"]:
                error_msg = result.get("error_message", "Invalid reservation details")
                missing = ", ".join(result["missing_fields"])
                if missing:
                    error_msg = f"{error_msg}\nMissing information: {missing}"
                return False, None, error_msg

            details = result["details"]
            reservation = ReservationDetails(
                restaurant_phone=details["phone_number"],
                party_size=details["party_size"],
                reservation_time=datetime.fromisoformat(details["reservation_time"]),
                customer_name=details["customer_name"],
                special_requests=details["special_requests"],
            )
            return True, reservation, None

        except Exception as e:
            return False, None, f"Error parsing reservation details: {str(e)}"

    def _get_call_system_prompt(self, reservation: ReservationDetails) -> str:
        """Generate system prompt with reservation context"""
        time_str = reservation.reservation_time.strftime("%I:%M %p")
        date_str = reservation.reservation_time.strftime("%A, %B %d")

        return f"""You are an AI assistant MAKING a reservation at a restaurant. You are the one calling them.
Current situation: You have called the restaurant and are trying to make the following reservation:
- Party size: {reservation.party_size} people
- Date: {date_str}
- Time: {time_str}
- Name: {reservation.customer_name}
- Special requests: {reservation.special_requests or 'None'}

Your role: You are the caller, not the restaurant. You need to:
1. Get confirmation for this specific reservation time
2. If they say yes/confirm, thank them and end positively
3. If they say no/busy, ask if a time 30 minutes earlier or later would work
4. If they have questions, answer them professionally
5. Keep responses brief and clear

Remember: YOU are making the reservation, they are the restaurant answering the phone.
Do not say you've lost track of the reservation - you have all the details above.

Example good responses:
- If they say "yes": "Wonderful, thank you for confirming. We'll see you on {date_str} at {time_str}."
- If they say "no": "I understand that time isn't available. Would 30 minutes earlier or later work better?"
- If they ask about party size: "Yes, it would be for {reservation.party_size} people. Is that manageable?"

Keep the conversation focused on confirming this reservation."""

    async def handle_restaurant_response(
        self, speech_input: str, reservation: ReservationDetails
    ) -> str:
        """Handle the restaurant's response using OpenAI"""
        try:
            logger.info(f"Processing restaurant response: '{speech_input}'")

            async with self.openai_client.beta.realtime.connect(
                model="gpt-4o-realtime-preview"
            ) as connection:
                # Set up session for text conversation
                await connection.session.update(session={"modalities": ["text"]})

                # Create system message with reservation context
                system_prompt = self._get_call_system_prompt(reservation)
                logger.info(f"System prompt:\n{system_prompt}")

                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    }
                )

                # Add context about what the restaurant said
                user_context = f"Restaurant employee said: '{speech_input}'"
                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_context}],
                    }
                )

                # Get response
                await connection.response.create()

                # Collect response
                full_response = []
                async for event in connection:
                    if event.type == "response.text.delta":
                        full_response.append(event.delta)
                        print(event.delta, end="", flush=True)  # Real-time printing
                    elif event.type == "response.done":
                        print()  # New line after response
                        break

                response_text = "".join(full_response)
                logger.info(f"AI response: '{response_text}'")

                # Generate TwiML with the AI response
                response = VoiceResponse()
                gather = response.gather(
                    input="speech",
                    action=f"{self.webhook_base_url}/gather",
                    method="POST",
                    language="en-US",
                    speechTimeout="auto",
                )
                gather.say(response_text, voice="alice")

                return str(response)

        except Exception as e:
            logger.error(f"Error in OpenAI conversation: {str(e)}", exc_info=True)
            response = VoiceResponse()
            response.say(
                "I apologize for the technical difficulty. Let me repeat: I'm calling to make a reservation. Would this time work for you?",
                voice="alice",
            )
            return str(response)


class RestaurantAgent:
    def __init__(self):
        MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)
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

        # Future enhancement: Handle follow-up messages for incomplete reservations
        # You could store conversation state in self.active_conversations
        # and implement a back-and-forth flow to collect missing information
