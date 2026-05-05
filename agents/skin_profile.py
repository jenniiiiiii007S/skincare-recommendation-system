import json
import re
import base64


def load_image_base64(image_path):
    """Load an image from disk and encode it as base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def skin_profile_agent(client, user_input, image_path=None):
    """Parse user input (and optional facial image) into a structured skin profile.

    Args:
        client: Google GenAI client instance.
        user_input: Raw natural language string from the user.
        image_path: Optional path to a facial image (jpg/png) for visual analysis.

    Returns:
        dict with keys: skin_type, concerns, allergies, age, current_products,
        goals, routine_request. Visual observations from the image are merged
        into concerns if an image is provided.
    """

    text_prompt = """You are a skincare specialist. Extract a structured skin profile from the user input below.
If a facial image is also provided, analyze it for visible skin conditions such as acne, redness, pigmentation, or under-eye bags, and incorporate these observations into the profile — even if the user did not mention them in text.

Return ONLY valid JSON with these exact fields:
{
  "skin_type": "oily" | "dry" | "combination" | "normal" | "sensitive" | null,
  "concerns": ["acne", "redness", "dark spots", ...],
  "allergies": ["fragrance", "parabens", ...],
  "age": number or null,
  "current_products": ["product name", ...],
  "goals": ["hydration", "anti-aging", ...],
  "routine_request": "full routine" | "morning only" | "evening only" | "product suggestion" | null,
  "image_observations": "brief description of visible skin conditions detected from image, or null if no image"
}

Rules:
- skin_type: infer from user description or image if possible, else null
- concerns: combine user-reported concerns with any conditions visibly detected in the image
- allergies: only from user text, never infer from image
- image_observations: fill this field only if an image was provided, summarize what you see
- Return null for fields you cannot determine
- No markdown, no backticks, just the JSON

USER INPUT: """ + user_input

    try:
        if image_path:
            # Load and encode image
            image_data = load_image_base64(image_path)

            # Determine media type from extension
            ext = image_path.lower().split(".")[-1]
            media_type = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"

            # Send text + image to Gemini
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    {
                        "parts": [
                            {"text": text_prompt},
                            {
                                "inline_data": {
                                    "mime_type": media_type,
                                    "data": image_data
                                }
                            }
                        ]
                    }
                ]
            )
        else:
            # Text only
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=text_prompt
            )

        # Parse JSON response
        text = re.sub(r'```json|```', '', response.text.strip()).strip()
        profile = json.loads(text)

        # Ensure all expected fields exist with safe defaults
        profile.setdefault("skin_type", None)
        profile.setdefault("concerns", [])
        profile.setdefault("allergies", [])
        profile.setdefault("age", None)
        profile.setdefault("current_products", [])
        profile.setdefault("goals", [])
        profile.setdefault("routine_request", None)
        profile.setdefault("image_observations", None)

        # Normalize None values
        if not profile["concerns"]:
            profile["concerns"] = []
        if not profile["allergies"]:
            profile["allergies"] = []
        if not profile["goals"]:
            profile["goals"] = []
        if not profile["current_products"]:
            profile["current_products"] = []

        # Print image observations if detected
        if profile.get("image_observations"):
            print(f"[skin_profile] Image analysis: {profile['image_observations']}")

        return profile

    except json.JSONDecodeError as e:
        print(f"[skin_profile] JSON parse failed: {e}")
        return {
            "skin_type": None, "concerns": [], "allergies": [],
            "age": None, "current_products": [], "goals": [],
            "routine_request": None, "image_observations": None
        }
    except Exception as e:
        print(f"[skin_profile] Agent failed: {e}")
        return {
            "skin_type": None, "concerns": [], "allergies": [],
            "age": None, "current_products": [], "goals": [],
            "routine_request": None, "image_observations": None
        }
