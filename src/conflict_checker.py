import json
import re

# Ingredient interaction rules grounded in the Skincare Knowledge Graph dataset
# Rules are split into two categories:
#   - conflict_rules_raw: ingredient combinations to AVOID (adverse effects)
#   - beneficial_pairs: ingredient combinations that WORK WELL together

conflict_rules_raw = [
    {"ingredient_a": "retinol", "ingredient_b": "vitamin c", "reason": "Cancel out effects."},
    {"ingredient_a": "retinol", "ingredient_b": "aha", "reason": "Cancel out effects and cause irritation."},
    {"ingredient_a": "retinol", "ingredient_b": "bha", "reason": "May cause breakouts, dry skin, and irritation."},
    {"ingredient_a": "retinol", "ingredient_b": "citric acid", "reason": "Excessive dryness, redness, sensitivity, or a rash."},
    {"ingredient_a": "retinol", "ingredient_b": "benzoyl peroxide", "reason": "Too harsh for skin and cancel out effects."},
    {"ingredient_a": "aha", "ingredient_b": "niacinamide", "reason": "Can cause redness."},
    {"ingredient_a": "bha", "ingredient_b": "niacinamide", "reason": "Can cause redness."},
    {"ingredient_a": "aha", "ingredient_b": "vitamin c", "reason": "Can cause irritation."},
    {"ingredient_a": "bha", "ingredient_b": "vitamin c", "reason": "Can cause irritation."},
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

    prompt = f"""You are a cosmetic chemist. Based on the INCI ingredient information below,
identify any combinations from the routine that should NOT be used together.

FULL ROUTINE INGREDIENTS: {ingredients_list}
INGREDIENTS TO FOCUS ON (not covered by hardcoded rules): {uncovered[:15]}

INCI DESCRIPTIONS:
{context}

Return ONLY a valid JSON list. Each conflict must have:
- ingredient_a
- ingredient_b
- reason
- severity: one of "mild", "moderate", "severe"

If no conflicts, return []. No markdown, no explanation, just the JSON list."""

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = re.sub(r'```json|```', '', response.text.strip()).strip()
        rag_conflicts = json.loads(text)

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
