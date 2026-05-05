import json
import re
from agents.budget_agent import format_budget_for_prompt


def routine_builder_agent(client, profile, retrieved_products, conflict_report, routine_pref=None, budget_profile=None):
    """Assemble retrieved products into a structured skincare routine using Gemini.

    Args:
        client: Google GenAI client instance.
        profile: dict from skin_profile_agent (skin_type, concerns, allergies, etc.).
        retrieved_products: list of dicts with keys 'name', 'brand', 'product_type', 'ingredients', 'price'.
        conflict_report: dict with keys 'conflicts' (list) and 'allergy_flags' (list) from conflict checker.
        routine_pref: 'AM', 'PM', or 'both' from pipeline routine preference detection.
        budget_profile: dict from budget_agent with overall_limit, tier, per_category, raw_mention.

    Returns:
        dict with keys: morning_routine, evening_routine, notes, warnings.
    """

    # Format product list for the prompt
    product_list_str = ""
    for i, p in enumerate(retrieved_products, 1):
        product_list_str += f"{i}. {p['name']} (Brand: {p['brand']}, Type: {p['product_type']}, Price: {p['price']})\n"
        product_list_str += f"   Ingredients: {p['ingredients'][:200]}...\n"

    # Format conflict info
    conflict_str = "None detected." if not conflict_report["conflicts"] else ""
    for c in conflict_report["conflicts"]:
        conflict_str += f"- AVOID: {c['ingredient_a']} + {c['ingredient_b']} — {c['reason']}\n"

    allergy_str = "None detected." if not conflict_report["allergy_flags"] else ""
    for a in conflict_report["allergy_flags"]:
        allergy_str += f"- Product '{a['product']}' contains allergen: {a['allergen']}\n"

    beneficial_str = ""
    if conflict_report.get("beneficial"):
        for b in conflict_report["beneficial"]:
            beneficial_str += f"- {b['ingredient_a']} + {b['ingredient_b']}: {b['benefit']}\n"
    if not beneficial_str:
        beneficial_str = "None identified."

    # Format budget info
    budget_str = format_budget_for_prompt(budget_profile) if budget_profile else "No budget constraint specified."

    prompt = f"""You are a skincare routine specialist. Based on the user's skin profile,
retrieved product candidates, and ingredient safety analysis, build a personalized
morning and evening skincare routine.

USER PROFILE:
- Skin type: {profile.get('skin_type') or 'unknown'}
- Concerns: {', '.join(profile.get('concerns') or []) or 'none specified'}
- Allergies: {', '.join(profile.get('allergies') or []) or 'none'}
- Age: {profile.get('age') or 'not specified'}
- Goals: {', '.join(profile.get('goals') or []) or 'general skincare'}
- Routine request: {profile.get('routine_request') or 'full routine'}

BUDGET:
{budget_str}

AVAILABLE PRODUCTS:
{product_list_str}

INGREDIENT CONFLICTS DETECTED:
{conflict_str}

ALLERGY FLAGS:
{allergy_str}

BENEFICIAL COMBINATIONS:
{beneficial_str}

INSTRUCTIONS:
1. BUDGET IS THE TOP PRIORITY. The user cannot afford products outside their budget,
   so a routine they can't pay for is useless. NEVER select a product whose price
   exceeds the user's per-category limit (if set), overall limit (if set), or tier limit.
   If a category has no in-budget option, OMIT that step entirely and note it in warnings —
   do NOT recommend an over-budget product as a fallback. The user can decide whether to
   add that step later when they have more to spend.
2. Select appropriate products from the list above for a morning and evening routine,
   choosing the best skin-type and concern match WITHIN budget.
3. DO NOT include any products flagged for allergy conflicts.
4. If ingredient conflicts were detected, separate conflicting products into different routines
   (e.g., one in morning, one in evening) or exclude one.
5. Order products in standard skincare order, for example: cleanser → toner → serum/treatment → moisturizer → sunscreen (AM only).
6. Explain WHY each product was chosen based on the user's concerns, goals, AND how it fits the budget.

Return ONLY valid JSON with no markdown formatting, no backticks, no explanation outside the JSON.
Use this exact structure:
{{
    "morning_routine": [
        {{
            "step": 1,
            "product_type": "Cleanser",
            "product_name": "...",
            "brand": "...",
            "why": "brief explanation of why this product suits the user"
        }}
    ],
    "evening_routine": [
        {{
            "step": 1,
            "product_type": "Cleanser",
            "product_name": "...",
            "brand": "...",
            "why": "brief explanation"
        }}
    ],
    "notes": ["any general skincare tips relevant to this user's profile"],
    "warnings": ["any flagged conflicts or allergies the user should be aware of"]
}}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    try:
        routine = json.loads(response.text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            try:
                routine = json.loads(json_match.group())
            except json.JSONDecodeError:
                routine = {"error": "Could not parse routine", "raw": response.text}
        else:
            routine = {"error": "Could not parse routine", "raw": response.text}

    return routine
