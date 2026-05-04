import json
import re

# Ingredient interaction rules grounded in the Skincare Knowledge Graph dataset
# Rules are split into two categories:
#   (1) conflict_rules_raw: ingredient combinations to AVOID (adverse effects)
#   (2) beneficial_pairs: ingredient combinations that WORK WELL together

conflict_rules_raw = [
    {"ingredient_a": "retinol", "ingredient_b": "vitamin c", "reason": "Cancel out effects."},
    {"ingredient_a": "retinol", "ingredient_b": "aha", "reason": "Cancel out effects and cause irritation."},
    {"ingredient_a": "retinol", "ingredient_b": "bha", "reason": "May cause breakouts, dry skin, and irritation."},
    {"ingredient_a": "retinol", "ingredient_b": "benzoyl peroxide", "reason": "Too harsh for skin and cancel out effects."},
    {"ingredient_a": "salicylic acid", "ingredient_b": "benzoyl peroxide", "reason": "Can cause skin irritation."},
]

beneficial_pairs = [
    {"ingredient_a": "hyaluronic acid", "ingredient_b": "polyglutamic acid", "benefit": "Better hydration."},
    {"ingredient_a": "retinol", "ingredient_b": "niacinamide", "benefit": "Improves skin blemishes, diminishes ageing, and evens out skin tone."},
    {"ingredient_a": "retinol", "ingredient_b": "peptides", "benefit": "Produces collagen and hyaluronic acid, reduces inflammation, and increases cell turnover."},
    {"ingredient_a": "vitamin c", "ingredient_b": "vitamin e", "benefit": "Can help prevent photodamage."},
    {"ingredient_a": "vitamin c", "ingredient_b": "ferulic acid", "benefit": "Ferulic acid stabilizes vitamin C and fends off free radicals."},
]

# Build bidirectional conflict lookup so checking either ingredient finds the rule
conflict_lookup = {}
for rule in conflict_rules_raw:
    a, b = rule["ingredient_a"], rule["ingredient_b"]
    conflict_lookup.setdefault(a, []).append({"conflicts_with": b, "reason": rule["reason"]})
    conflict_lookup.setdefault(b, []).append({"conflicts_with": a, "reason": rule["reason"]})

# Build bidirectional beneficial lookup similarly
beneficial_lookup = {}
for pair in beneficial_pairs:
    a, b = pair["ingredient_a"], pair["ingredient_b"]
    beneficial_lookup.setdefault(a, []).append({"pairs_with": b, "benefit": pair["benefit"]})
    beneficial_lookup.setdefault(b, []).append({"pairs_with": a, "benefit": pair["benefit"]})

# Alias mapping: generic terms → actual INCI ingredient name variants
INGREDIENT_ALIASES = {
    "aha": ["glycolic acid", "lactic acid", "mandelic acid"],
    "bha": ["salicylic acid", "beta hydroxy acid"],
    "vitamin c": ["ascorbic acid", "l-ascorbic acid", "sodium ascorbyl phosphate",
                  "ascorbyl glucoside", "ascorbyl tetraisopalmitate", "3-o-ethyl ascorbic acid"],
    "retinol": ["retinol", "retinyl palmitate", "retinyl acetate", "retinal",
                "hydroxypinacolone retinoate"],
    "benzoyl peroxide": ["benzoyl peroxide"],
    "niacinamide": ["niacinamide", "nicotinamide"],
}

def normalize_ingredient(ing):
    """Map an ingredient name to its canonical conflict_lookup key if an alias exists."""
    ing_lower = ing.lower().strip()
    for canonical, aliases in INGREDIENT_ALIASES.items():
        if ing_lower == canonical or ing_lower in aliases:
            return canonical
    return ing_lower

def rag_conflict_check(client, ingredient_collection, ingredients_list):
    uncovered = [i for i in ingredients_list if i not in conflict_lookup]
    if not uncovered:
        return []

    # Query ChromaDB per ingredient for better retrieval
    all_docs = []
    for ing in uncovered[:15]:
        res = ingredient_collection.query(query_texts=[ing], n_results=3)
        if res["documents"][0]:
            all_docs.extend(res["documents"][0])

    context = "\n\n".join(all_docs)
    if not context.strip():
        return []

    prompt = f"""You are a cosmetic chemist and dermatologist reviewing a MULTI-PRODUCT skincare routine.
The ingredients below come from DIFFERENT products assigned to DIFFERENT routine steps (cleanser, serum, moisturizer, etc.).
The routine builder will AUTOMATICALLY separate conflicting ingredients across AM and PM routines.

Flag a conflict ONLY if ALL of the following are true:
1. A dermatologist would tell a patient to NEVER have both products in their routine on the same day, even at different times.
2. The harm is documented and clinically significant — causes real skin damage, not just reduced efficacy.
3. The interaction is well-established in skincare literature, not theoretical.

DO NOT flag:
- Ingredients that are simply suboptimal or slightly less effective together
- Combinations that are fine when used at different times of day (AM vs PM)
- AHA + BHA combinations — these are commonly used in full routines with AM/PM separation
- Formulation-level incompatibilities (e.g. cationic/anionic ingredient reactions that only matter inside a bottle during manufacturing)
- Ingredients you are uncertain about
- Any combination that is merely "less than ideal" rather than genuinely harmful
- Two AHA ingredients (e.g. glycolic acid + lactic acid) from different products — manageable with routine ordering
- Alcohol denat combined with AHA or BHA — not a clinically significant same-day hazard
- Any combination that the routine builder can resolve by separating into AM vs PM

FULL ROUTINE INGREDIENTS: {ingredients_list}
INGREDIENTS TO FOCUS ON (not covered by hardcoded rules): {uncovered[:15]}

INCI DESCRIPTIONS:
{context}

Return ONLY a valid JSON list of SEVERE, clinically documented, same-day conflicts. Each item must have:
- ingredient_a
- ingredient_b
- reason
- severity: must be "severe"

If uncertain or if no same-day severe conflicts exist, return []. No markdown, no explanation, just the JSON list."""

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = re.sub(r'```json|```', '', response.text.strip()).strip()
        rag_conflicts = json.loads(text)

        # Safety filter: drop anything not explicitly labeled severe
        rag_conflicts = [c for c in rag_conflicts if c.get("severity", "").lower() == "severe"]

        # Deduplicate (a,b) and (b,a) pairs
        seen = set()
        deduped = []
        for c in rag_conflicts:
            key = frozenset([c["ingredient_a"], c["ingredient_b"]])
            if key not in seen:
                seen.add(key)
                c["source"] = "rag"
                deduped.append(c)

        return deduped

    except json.JSONDecodeError as e:
        print(f"[conflict_checker] Failed to parse Gemini response: {e}")
        return []
    except Exception as e:
        print(f"[conflict_checker] RAG check failed: {e}")
        return []
