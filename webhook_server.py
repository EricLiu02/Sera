from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn
from pyngrok import ngrok, conf
import os
from restaurant_agent import TwilioReservationAgent, ReservationDetails
from dotenv import load_dotenv
from io import BytesIO
import logging
from twilio.twiml.voice_response import VoiceResponse
from datetime import datetime

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


@app.post("/stream")
async def handle_stream(request: Request):
    logger.info("Received stream request")
    try:
        # Simple test response
        test_message = "This is a test response."
        speech_response = await agent.openai_client.audio.speech.create(
            model="tts-1", voice="alloy", input=test_message
        )

        return StreamingResponse(
            BytesIO(speech_response.content), media_type="audio/mpeg"
        )
    except Exception as e:
        logger.error(f"Error in stream handler: {str(e)}", exc_info=True)
        return {"error": str(e)}


@app.post("/status")
async def call_status(request: Request):
    form_data = await request.form()
    logger.info(f"Call status update: {dict(form_data)}")
    return {"success": True}


@app.post("/gather")
async def handle_gather(request: Request):
    try:
        # Get speech input
        form_data = await request.form()
        speech_result = form_data.get("SpeechResult", "")
        logger.info(f"Received speech: {speech_result}")

        # Get reservation details from the call
        call_sid = form_data.get("CallSid")
        logger.info(f"Call SID: {call_sid}")

        # Create a test reservation for now (you'll need to store/retrieve the actual reservation)
        reservation = ReservationDetails(
            restaurant_phone="+1234567890",
            party_size=4,
            reservation_time=datetime.now(),
            customer_name="John Doe",
        )

        # Let the agent handle everything
        twiml_response = await agent.handle_restaurant_response(
            speech_result, reservation
        )
        return Response(content=twiml_response, media_type="application/xml")

    except Exception as e:
        logger.error(f"Error in gather handler: {str(e)}", exc_info=True)
        response = VoiceResponse()
        response.say(
            "I apologize for the technical difficulty. Please try again.",
            voice="Polly.Russell",
        )
        return Response(content=str(response), media_type="application/xml")


@app.post("/retry")
async def handle_retry(request: Request):
    response = VoiceResponse()
    response.say("I didn't receive any input. Goodbye.", voice="Polly.Russell")
    return Response(content=str(response), media_type="application/xml")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
