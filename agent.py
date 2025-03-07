import os
from mistralai import Mistral
import discord
from twilio.rest import Client
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict
import re

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

        if not all([self.account_sid, self.auth_token, self.twilio_number]):
            raise ValueError(
                "Missing required Twilio credentials in environment variables"
            )

        self.client = Client(self.account_sid, self.auth_token)

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

    def parse_reservation_request(self, message: str) -> Optional[ReservationDetails]:
        """
        Parses a reservation request message
        Returns ReservationDetails or None if parsing fails

        Example message format:
        "Make a reservation at +1234567890 for 4 people tomorrow at 7:30 PM under John Doe"
        """
        try:
            # This is a simple parser - you might want to use a more sophisticated NLP solution
            phone_match = re.search(r"\+?\d[\d-]{9,}", message)
            party_size_match = re.search(r"(\d+)\s+people", message)
            name_match = re.search(r"under\s+([A-Za-z\s]+)", message)

            if not all([phone_match, party_size_match, name_match]):
                return None

            # For this example, we're assuming the time is for tomorrow at the specified time
            # You'd want to enhance this with proper datetime parsing
            time_match = re.search(
                r"at\s+(\d{1,2}):?(\d{2})?\s*(AM|PM)", message, re.IGNORECASE
            )
            if not time_match:
                return None

            # Extract special requests if any
            special_match = re.search(
                r"special\s+requests?:?\s+([^.]+)", message, re.IGNORECASE
            )

            return ReservationDetails(
                restaurant_phone=phone_match.group(0),
                party_size=int(party_size_match.group(1)),
                reservation_time=datetime.now(),  # You'd want to properly parse this
                customer_name=name_match.group(1).strip(),
                special_requests=special_match.group(1) if special_match else None,
            )

        except Exception:
            return None


class RestaurantAgent:
    def __init__(self):
        self.reservation_agent = TwilioReservationAgent()

    async def run(self, message: discord.Message):
        # Check if this is a reservation request
        if "make a reservation" in message.content.lower():
            # Parse the reservation request
            reservation_details = self.reservation_agent.parse_reservation_request(
                message.content
            )

            if not reservation_details:
                return (
                    "I couldn't understand the reservation details. Please use this format:\n"
                    "Make a reservation at +1234567890 for 4 people tomorrow at 7:30 PM under John Doe"
                )

            # Make the reservation call
            return await self.reservation_agent.make_reservation_call(
                reservation_details
            )
