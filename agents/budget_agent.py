import json
import re

PRODUCT_TYPES = ["Cleanser", "Serum", "Moisturizer", "Sun protect", "Retinol", "Face oil"]


def budget_agent(client, user_input):
    """Parse user input into a structured budget profile.

    Returns:
        dict with keys:
            - overall_limit: float or None
            - tier: "under_50" / "under_100" / "under_200" / "200_plus" / "any"
            - per_category: dict mapping product type to float limit or None
            - raw_mention: the budget-related phrase the user said, or None
    """

    prompt = """You are a skincare budget specialist. Extract budget information from the user input.

User input: \"""" + user_input + """\"

Product categories to consider: Cleanser, Serum, Moisturizer, Sun protect, Retinol, Face oil.
(Note: "sunscreen" or "SPF" maps to Sun protect.)

EXTRACT THESE FIELDS:

1. overall_limit: total budget in dollars as a number, or null if not mentioned.
   Examples:
     "I have $150 to spend total"           -> 150
     "keep the whole routine under $80"     -> 80
     "no budget for the whole thing"        -> null (this is a 200_plus tier signal, not a number)

2. tier: classify the OVERALL budget into one value:
     under_50  -> cheap, budget, drugstore, affordable, under $50
     under_100 -> mid-range, moderate, under $100
     under_200 -> under $200
     200_plus  -> luxury, high-end, splurge, no budget limit, $200+
     any       -> no budget mentioned at all

3. per_category: a dict where each key is a product type. For each one,
   extract a dollar limit IF AND ONLY IF the user mentioned a specific
   amount tied to that product. Otherwise null.

   Catch these phrasings:
     "I'd spend $60 on a serum"                       -> Serum: 60
     "willing to splurge on moisturizer up to $120"   -> Moisturizer: 120
     "cleanser should be cheap, like $15 max"         -> Cleanser: 15
     "keep sunscreen under $25"                       -> Sun protect: 25
     "retinol can be expensive, $80 ok"               -> Retinol: 80
     "face oil no more than $40"                      -> Face oil: 40
     "$50 for cleanser, $70 for serum, $100 for moisturizer" -> Cleanser: 50, Serum: 70, Moisturizer: 100
     "spend more on serum and moisturizer, less on cleanser" -> null for all (no numbers given)
     "drugstore cleanser but high-end serum"          -> Cleanser: 50, Serum: null (qualitative only on serum)

   IMPORTANT: only fill a per_category value when there is a NUMERIC dollar
   amount tied to that specific product type. If the user only gives a
   qualitative preference (cheap, expensive, splurge) without a number
   for that category, leave it as null and let the tier handle it.

4. raw_mention: copy the exact budget-related phrase from the user, or null.

If the user mentions no budget at all, return tier "any" and everything else null.

Return ONLY valid JSON, no markdown, no backticks, no explanation:
{
    "overall_limit": null,
    "tier": "any",
    "per_category": {
        "Cleanser": null,
        "Serum": null,
        "Moisturizer": null,
        "Sun protect": null,
        "Retinol": null,
        "Face oil": null
    },
    "raw_mention": null
}"""

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = re.sub(r'```json|```', '', response.text.strip()).strip()
        budget_profile = json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            try:
                budget_profile = json.loads(json_match.group())
            except json.JSONDecodeError:
                budget_profile = _default_budget_profile()
        else:
            budget_profile = _default_budget_profile()
    except Exception as e:
        print("[budget_agent] Gemini call failed: " + str(e))
        budget_profile = _default_budget_profile()

    budget_profile = _validate_budget_profile(budget_profile)
    return budget_profile


def _default_budget_profile():
    """Return a safe default profile with no budget constraint."""
    return {
        "overall_limit": None,
        "tier": "any",
        "per_category": {ptype: None for ptype in PRODUCT_TYPES},
        "raw_mention": None
    }


def _validate_budget_profile(profile):
    """Ensure all expected keys exist and values are the right types."""
    defaults = _default_budget_profile()

    for key in defaults:
        if key not in profile:
            profile[key] = defaults[key]

    # Fill any missing product types in per_category
    for ptype in PRODUCT_TYPES:
        if ptype not in profile["per_category"]:
            profile["per_category"][ptype] = None

    # Coerce overall_limit to float
    try:
        if profile["overall_limit"] is not None:
            profile["overall_limit"] = float(profile["overall_limit"])
    except (ValueError, TypeError):
        profile["overall_limit"] = None

    # Coerce per-category limits to float
    for ptype in profile["per_category"]:
        try:
            if profile["per_category"][ptype] is not None:
                profile["per_category"][ptype] = float(profile["per_category"][ptype])
        except (ValueError, TypeError):
            profile["per_category"][ptype] = None

    # Validate tier value
    valid_tiers = ["under_50", "under_100", "under_200", "200_plus", "any"]
    if profile["tier"] not in valid_tiers:
        profile["tier"] = "any"

    return profile


def filter_products_by_budget(products, budget_profile):
    """Filter products using per-category limits with overall limit as fallback.

    Priority order for each product:
      1. Per-category limit if set for that product type
      2. Overall limit if set
      3. Tier-derived limit if overall_limit is null
      4. No filter if tier is 'any' and nothing else is set

    If a product type has no in-budget option, the cheapest available product
    is added as a fallback and a notice is recorded so the user is informed
    that no cheaper alternatives exist in that category.
    """

    # Reset fallback tracker each call
    filter_products_by_budget.last_fallbacks = []

    # Short circuit when no budget mentioned at all
    if budget_profile["tier"] == "any" and budget_profile["overall_limit"] is None:
        has_any_cat_limit = any(v is not None for v in budget_profile["per_category"].values())
        if not has_any_cat_limit:
            return products

    TIER_LIMITS = {
        "under_50":  50.0,
        "under_100": 100.0,
        "under_200": 200.0,
        "200_plus":  None,
        "any":       None,
    }

    tier_limit    = TIER_LIMITS.get(budget_profile["tier"])
    overall_limit = budget_profile["overall_limit"]
    per_category  = budget_profile["per_category"]

    def get_limit_for_type(ptype):
        cat_limit = per_category.get(ptype)
        if cat_limit is not None:
            return cat_limit
        if overall_limit is not None:
            return overall_limit
        return tier_limit

    def parse_price(price_str):
        try:
            return float(str(price_str).replace("$", "").replace(",", "").strip())
        except (ValueError, TypeError):
            return None

    def price_ok(p, limit):
        if limit is None:
            return True
        price = parse_price(p.get("price"))
        if price is None:
            return True
        if budget_profile["tier"] == "200_plus" and per_category.get(p["product_type"]) is None:
            return price >= 200.0
        return price <= limit

    filtered = []
    for p in products:
        limit = get_limit_for_type(p["product_type"])
        if price_ok(p, limit):
            filtered.append(p)

    # Fallback: when a category has no in-budget option, add the cheapest
    # available product and record what the user's limit was vs. its actual price
    types_before = set(p["product_type"] for p in products)
    types_after  = set(p["product_type"] for p in filtered)
    missing      = types_before - types_after

    fallback_records = []
    for ptype in missing:
        candidates = [p for p in products if p["product_type"] == ptype]
        candidates.sort(key=lambda p: parse_price(p.get("price")) or 9999)
        cheapest = candidates[0]
        cheapest_price = parse_price(cheapest.get("price"))
        attempted_limit = get_limit_for_type(ptype)

        filtered.append(cheapest)

        record = {
            "product_type": ptype,
            "product_name": cheapest.get("name"),
            "actual_price": cheapest_price,
            "user_limit": attempted_limit,
        }
        fallback_records.append(record)

        print(
            "[budget_agent] No products in budget for " + ptype +
            " (limit: $" + str(int(attempted_limit)) + "). " +
            "Using cheapest available: " + str(cheapest.get("name")) +
            " at $" + str(cheapest_price) + "."
        )

    filter_products_by_budget.last_fallbacks = fallback_records

    return filtered

def format_budget_for_prompt(budget_profile):
    """Return a human-readable budget summary to inject into the routine builder prompt."""
    tier_labels = {
        "under_50":  "under $50 (budget-friendly / drugstore)",
        "under_100": "under $100 (mid-range)",
        "under_200": "under $200",
        "200_plus":  "$200+ (luxury / premium)",
        "any":       "no specific budget constraint",
    }

    lines = []

    if budget_profile["raw_mention"]:
        lines.append("User said: \"" + budget_profile["raw_mention"] + "\"")

    if budget_profile["overall_limit"] is not None:
        lines.append("Overall limit: $" + str(int(budget_profile["overall_limit"])))
    else:
        lines.append("Budget tier: " + tier_labels.get(budget_profile["tier"], "not specified"))

    cat_limits = [
        k + ": $" + str(int(v))
        for k, v in budget_profile["per_category"].items()
        if v is not None
    ]
    if cat_limits:
        lines.append("Per-category limits: " + ", ".join(cat_limits))

    if budget_profile.get("budget_fallbacks"):
        lines.append("\nNOTE: Some categories had no in-budget options. The cheapest available product was used:")
        for fb in budget_profile["budget_fallbacks"]:
            lines.append(
                "  - " + fb["product_type"] +
                ": user limit was $" + str(int(fb["user_limit"])) +
                " but cheapest available is " + str(fb["product_name"]) +
                " at $" + str(fb["actual_price"]) +
                " (no cheaper option exists)"
            )

    return "\n".join(lines) if lines else "No budget constraint specified."
