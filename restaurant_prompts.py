from datetime import datetime
from typing import Optional
from dataclasses import dataclass


@dataclass
class ReservationDetails:
    restaurant_phone: str
    party_size: int
    reservation_time: datetime
    customer_name: str
    special_requests: Optional[str] = None


EXTRACT_RESERVATION_DETAILS_PROMPT = """
    Validate the following:
    1. All required fields are present
    2. Reservation time must be in the future
    3. Phone number must be either:
       - 10 digits for US numbers (will add +1)
       - Full international format with + and country code
    4. Party size must be a positive number

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
EXTRACT_RESERVATION_DETAILS_PROMPT_HEADER = """
    Extract reservation details from the message and convert relative dates/times to absolute dates.
    Current date/time reference: {cur_time}
    """


def get_extract_reservation_details_prompt():
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        EXTRACT_RESERVATION_DETAILS_PROMPT_HEADER.format(cur_time=current_datetime)
        + EXTRACT_RESERVATION_DETAILS_PROMPT
    )


def get_restaurant_conversation_prompt(
    reservation: ReservationDetails, is_initial: bool
):
    return f"""
    You are an AI assistant making a restaurant reservation call.
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
    Keep the conversation focused on confirming this reservation.
    """
