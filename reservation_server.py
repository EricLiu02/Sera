import os
from datetime import datetime
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from pyngrok import ngrok, conf
from twilio.twiml.voice_response import VoiceResponse
import uvicorn

from tools.reservation_agent import TwilioReservationAgent, ReservationDetails

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure ngrok
ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")
if not ngrok_auth_token:
    raise ValueError("NGROK_AUTH_TOKEN not found in environment variables")

# Set up ngrok
conf.get_default().auth_token = ngrok_auth_token
# Start ngrok
http_tunnel = ngrok.connect(8000)
WEBHOOK_BASE_URL = http_tunnel.public_url
print(f"Webhook URL: {WEBHOOK_BASE_URL}")

# When ngrok starts
with open("webhook_url.txt", "w") as f:
    f.write(http_tunnel.public_url)

app = FastAPI()
agent = TwilioReservationAgent()
current_reservation = None


@app.post("/set_reservation")
async def set_reservation(reservation: dict):
    global current_reservation
    current_reservation = ReservationDetails(**reservation)
    # current_reservation = ReservationDetails(
    #     party_size=reservation["party_size"],
    #     reservation_time=datetime.fromisoformat(reservation["reservation_time"]),
    #     customer_name=reservation["customer_name"],
    #     restaurant_phone=reservation["restaurant_phone"],
    #     special_requests=reservation.get(
    #         "special_requests"
    #     ),  # Using .get() since it's optional
    #     chat_history=reservation.get("chat_history", []),
    # )
    logger.info(f"Reservation set: {current_reservation}")
    print(f"Reservation set: {current_reservation}")
    return {"status": "success"}


@app.post("/gather")
async def handle_gather(request: Request):
    try:
        form_data = await request.form()
        speech_result = form_data.get("SpeechResult", "")
        call_sid = form_data.get("CallSid")
        logger.info(f"Received speech: {speech_result}")
        logger.info(f"Form data: {form_data}")
        logger.info(f"Call SID: {call_sid}")

        if not current_reservation:
            logger.error(f"No reservation found for call {call_sid}")
            response = VoiceResponse()
            response.say(
                "I apologize, but I've lost track of the conversation. Goodbye.",
                voice="alice",
            )
            return Response(content=str(response), media_type="application/xml")

        current_reservation.chat_history.append(f"User: {speech_result}")

        twiml_response = await agent.handle_restaurant_response(
            speech_result, current_reservation
        )
        return Response(content=twiml_response, media_type="application/xml")

    except Exception as e:
        logger.error(f"Error in gather handler: {str(e)}", exc_info=True)
        response = VoiceResponse()
        response.say(
            "I apologize for the technical difficulty. Please try again.", voice="alice"
        )
        return Response(content=str(response), media_type="application/xml")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
