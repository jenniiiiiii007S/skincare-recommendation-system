import json
import re


def skin_profile_agent(client, user_input):
    """Parse raw user text into a structured skin profile using Gemini.

    Args:
        client: Google GenAI client instance.
        user_input: Raw natural language string from the user.

    Returns:
        dict with keys: skin_type, concerns, allergies, age,
        current_products, goals, routine_request.
    """
    prompt = f"""You are a skincare intake specialist. Analyze the following user input and extract
a structured skin profile. Return ONLY valid JSON with no markdown formatting, no backticks, no explanation.

User input: "{user_input}"

Return this exact JSON structure (use null for any field not mentioned):
{{
    "skin_type": "oily/dry/combination/sensitive/normal or null",
    "concerns": ["list of skin concerns mentioned, e.g., acne, dark circles, redness, wrinkles"],
    "allergies": ["list of any allergies or ingredients to avoid"],
    "age": null,
    "current_products": ["list of any products the user currently uses"],
    "goals": ["what the user wants to achieve, e.g., reduce acne, brighten skin"],
    "routine_request": "full routine / specific product / product suggestion"
}}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:
        profile = json.loads(response.text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            profile = json.loads(json_match.group())
        else:
            profile = {"error": "Could not parse profile", "raw": response.text}

    return profile
