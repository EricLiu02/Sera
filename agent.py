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

    async def _generate_ai_greeting(self, reservation: ReservationDetails) -> str:
        """Generate initial greeting using OpenAI"""
        try:
            logger.info("Generating AI greeting")

            async with self.openai_client.beta.realtime.connect(
                model="gpt-4o-realtime-preview"
            ) as connection:
                await connection.session.update(session={"modalities": ["text"]})

                system_prompt = f"""You are an AI assistant making a restaurant reservation call.
You need to make the INITIAL greeting to start the conversation.

Reservation details:
- Party size: {reservation.party_size} people
- Date: {reservation.reservation_time.strftime('%A, %B %d')}
- Time: {reservation.reservation_time.strftime('%I:%M %p')}
- Name: {reservation.customer_name}
- Special requests: {reservation.special_requests or 'None'}

Your task:
1. Introduce yourself as an AI assistant
2. Clearly state you're calling to make a reservation
3. State all the reservation details
4. Ask if this time works for them
5. Keep it professional but conversational

Remember: YOU are making the call TO the restaurant."""

                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
                    }
                )

                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Generate the initial greeting for the call.",
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

                greeting = "".join(full_response)
                logger.info(f"Generated AI greeting: {greeting}")
                return greeting

        except Exception as e:
            logger.error(f"Error generating AI greeting: {str(e)}", exc_info=True)
            return (
                "Hello, I'm an AI assistant calling to make a reservation. "
                "Would you be able to help me with that?"
            )

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
                model="gpt-4o-realtime-preview"
            ) as connection:
                await connection.session.update(session={"modalities": ["text"]})

                system_prompt = f"""You are an AI assistant making a restaurant reservation call.
You are the one MAKING the call TO the restaurant.

Current reservation details:
- Party size: {reservation.party_size} people
- Date: {reservation.reservation_time.strftime('%A, %B %d')}
- Time: {reservation.reservation_time.strftime('%I:%M %p')}
- Name: {reservation.customer_name}
- Special requests: {reservation.special_requests or 'None'}

Your task:
{
    "If this is the initial greeting:" if is_initial else "You're in the middle of the call:"
}
1. {"Introduce yourself professionally as an AI assistant" if is_initial else "Respond naturally to what they just said"}
2. {"Clearly state all reservation details" if is_initial else "Stay focused on confirming the reservation"}
3. {"Ask if the time works for them" if is_initial else "Handle their response appropriately"}
4. If they say no/busy: Ask about 30 minutes earlier/later
5. If they have questions: Answer professionally
6. Keep responses conversational but focused

Remember: YOU are making the reservation, they are answering your call.
Keep the conversation focused on confirming this reservation."""

                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}],
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
                                    "text": f"Restaurant employee said: '{speech_input}'",
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

                # Generate TwiML with the AI response
                response = VoiceResponse()
                gather = response.gather(
                    input="speech",
                    action=f"{self.webhook_base_url}/gather",
                    method="POST",
                    language="en-US",
                    speechTimeout="auto",
                )
                gather.say(ai_response, voice="alice")

                return str(response)

        except Exception as e:
            logger.error(f"Error in OpenAI conversation: {str(e)}", exc_info=True)
            response = VoiceResponse()
            response.say(
                "I apologize for the technical difficulty. Could you please repeat that?",
                voice="alice",
            )
            return str(response)

    def _generate_initial_greeting(self, reservation: ReservationDetails) -> str:
        """Generate the initial TwiML with AI greeting"""
        return asyncio.run(self.handle_conversation(reservation, is_initial=True))

    async def handle_restaurant_response(
        self, speech_input: str, reservation: ReservationDetails
    ) -> str:
        """Handle ongoing conversation"""
        return await self.handle_conversation(
            reservation, speech_input, is_initial=False
        )

    def _get_prompt(self) -> str:
        """Returns the prompt with the current date/time context"""
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")

        datetime_context = f"""
    Extract reservation details from the message and convert relative dates/times to absolute dates.
    Current date/time reference: {current_datetime}

    Validate the following:
    1. All required fields are present
    2. Reservation time must be in the future
    3. Phone number must be either:
       - 10 digits for US numbers (will add +1)
       - Full international format with + and country code
    4. Party size must be a positive number
    """

        prompt_template = """
    Required fields: phone_number, party_size, reservation_time, customer_name
    Optional fields: special_requests

    Phone number format:
    - US numbers: 10 digits (e.g., "1234567890" or "123-456-7890")
    - International: Include + and country code (e.g., "+441234567890")
    Note: US numbers will automatically get +1 prefix if not provided.

    Handle these date/time formats:
    - Relative: "tomorrow", "next Tuesday", "this Friday"
    - Time: "7pm", "7:30 PM", "19:30"
    - Combined: "tomorrow at 7", "next Friday at 8:30 PM"

    If information is missing, provide a helpful error message explaining what's needed.
    Example error messages:
    - "Please provide a phone number for the restaurant"
    - "I need to know the name for the reservation"
    - "Could you specify what time you'd like the reservation?"
    - "How many people will be dining?"

    Return in JSON format:
    {
        "complete": true/false,
        "missing_fields": ["field1", "field2"],
        "error_message": "A helpful, conversational message explaining what information is needed",
        "details": {
            "phone_number": "phone number with country code",
            "party_size": number,
            "reservation_time": "YYYY-MM-DD HH:MM",
            "customer_name": "name",
            "special_requests": "requests or null"
        }
    }

    Examples:
    Message: Make a reservation for 4 people tomorrow at 7:30 PM
    Response: {
        "complete": false,
        "missing_fields": ["phone_number", "customer_name"],
        "error_message": "I can help make that reservation for 4 people tomorrow evening. I just need the restaurant's phone number and the name to put the reservation under. Could you provide those details?",
        "details": {
            "phone_number": null,
            "party_size": 4,
            "reservation_time": "2024-03-08 19:30",
            "customer_name": null,
            "special_requests": null
        }
    }

    Message: Book a table at +1234567890 for tomorrow night under Mike
    Response: {
        "complete": false,
        "missing_fields": ["party_size"],
        "error_message": "I'll be happy to make a reservation for tomorrow night under Mike's name. How many people will be in your party?",
        "details": {
            "phone_number": "+1234567890",
            "party_size": null,
            "reservation_time": "2024-03-08 19:00",
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
