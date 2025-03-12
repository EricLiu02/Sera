import os
import logging
import atexit
from typing import Dict

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
TWILIO_VOICE = os.getenv("TWILIO_VOICE")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
if not NGROK_AUTH_TOKEN:
    raise ValueError("NGROK_AUTH_TOKEN not found in environment variables")

# Set up ngrok
conf.get_default().auth_token = NGROK_AUTH_TOKEN
http_tunnel = ngrok.connect(8000)
WEBHOOK_BASE_URL = http_tunnel.public_url
print(f"Webhook URL: {WEBHOOK_BASE_URL}")

# When ngrok starts
with open("webhook_url.txt", "w") as f:
    f.write(http_tunnel.public_url)

app = FastAPI()
agent = TwilioReservationAgent()

current_reservation = None
active_calls: Dict[str, ReservationDetails] = {}


@app.post("/set_reservation")
async def set_reservation(reservation: dict):
    global current_reservation
    current_reservation = ReservationDetails(**reservation)
    logger.info(f"Reservation set: {current_reservation}")
    return {"status": "success"}


@app.post("/call_status")
async def handle_call_status(request: Request):
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        call_status = form_data.get("CallStatus")
        call_duration = form_data.get("CallDuration")
        recording_url = form_data.get("RecordingUrl")

        logger.info(f"Call {call_sid} ended with status: {call_status}")

        if call_sid in active_calls:
            call_context = active_calls[call_sid]
            call_context.status = call_status

            print(f"Call {call_sid} ended with status: {call_status}")
            print(f"Call duration: {call_duration}")
            print(f"Recording URL: {recording_url}")

            await agent.analyze_call_outcome(call_context)

            del active_calls[call_sid]

    except Exception as e:
        logger.error(f"Error handling call status: {str(e)}", exc_info=True)

    return Response(content="", media_type="application/xml")


@app.post("/gather")
async def handle_gather(request: Request):
    try:
        form_data = await request.form()
        speech_result = form_data.get("SpeechResult", "")
        call_sid = form_data.get("CallSid")
        logger.info(f"Received speech: {speech_result}")
        logger.info(f"Form data: {form_data}")
        logger.info(f"Call SID: {call_sid}")

        if call_sid in active_calls:
            reservation = active_calls[call_sid]
        elif current_reservation:
            reservation = current_reservation
            active_calls[call_sid] = reservation
        else:
            logger.error(f"No reservation found for call {call_sid}")
            response = VoiceResponse()
            response.say(
                "I apologize, but I've lost track of the conversation. Goodbye.",
                voice=TWILIO_VOICE,
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
            "I apologize for the technical difficulty. Please try again.",
            voice=TWILIO_VOICE,
        )
        return Response(content=str(response), media_type="application/xml")


def cleanup():
    if os.path.exists("webhook_url.txt"):
        os.remove("webhook_url.txt")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    atexit.register(cleanup)
