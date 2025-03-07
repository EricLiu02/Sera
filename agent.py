import os
from mistralai import Mistral
import discord
from twilio.rest import Client
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict
import re
import json

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

        if not all([self.account_sid, self.auth_token, self.twilio_number]):
            raise ValueError(
                "Missing required Twilio credentials in environment variables"
            )

        self.client = Client(self.account_sid, self.auth_token)
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)

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

    async def make_reservation_call(self, reservation: ReservationDetails) -> str:
        try:
            formatted_restaurant_phone = self._format_phone_number(
                reservation.restaurant_phone
            )
            twiml = self._generate_twiml(reservation)

            call = self.client.calls.create(
                twiml=twiml,
                to=formatted_restaurant_phone,
                from_=self.twilio_number,
                record=True,
            )

            return (
                f"✓ Calling {formatted_restaurant_phone} for your reservation.\n"
                f"Reservation details:\n"
                f"- Party size: {reservation.party_size}\n"
                f"- Time: {reservation.reservation_time.strftime('%I:%M %p on %A, %B %d')}\n"
                f"- Name: {reservation.customer_name}\n"
                f"- Special requests: {reservation.special_requests or 'None'}"
            )

        except Exception as e:
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
