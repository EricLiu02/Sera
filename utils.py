import base64
import discord


async def extract_image_base64(message: discord.Message) -> str:
    """
    Extracts and converts the first valid image attachment in a Discord message to a base64-encoded string.

    Args:
        message (discord.Message): The Discord message object potentially containing image attachments.

    Returns:
        str: A base64-encoded string representing the image.

    Raises:
        ValueError: If no valid image attachment is found.
        Exception: If an error occurs during image download or conversion.
    """
    if not message.attachments:
        raise ValueError("No image attachment found. Please attach an image.")

    # Filter for valid image attachments (e.g., .png, .jpg, .jpeg)
    image_attachment = next(
        (att for att in message.attachments if att.filename.lower().endswith(
            ('.png', '.jpg', '.jpeg'))),
        None
    )

    if image_attachment is None:
        raise ValueError(
            "No valid image attachment found. Please attach a jpg or png image.")

    try:
        # Asynchronously read the image bytes from the attachment
        image_bytes = await image_attachment.read()
        # Convert the image bytes to a base64 encoded string
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return base64_image
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")
