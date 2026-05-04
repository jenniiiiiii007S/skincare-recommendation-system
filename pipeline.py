import json
import re
from agents.skin_profile import skin_profile_agent
from agents.retrieval import search_products
from agents.conflict_checker import conflict_lookup, beneficial_lookup, rag_conflict_check
from agents.routine_builder import routine_builder_agent


def detect_routine_preference(client, user_input):
    """Use Gemini to detect if user wants AM, PM, or both routines."""

    prompt = f"""A user is asking for skincare advice. Determine if they want a morning routine, evening routine, or both.

USER INPUT: "{user_input}"

Rules:
- If they explicitly mention morning, daytime, AM, or waking up → return AM
- If they explicitly mention night, evening, PM, bedtime, or before bed → return PM
- If they mention both, neither, or just ask for a general/full routine → return both

Return ONLY one word: AM, PM, or both. Nothing else."""

    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        result = response.text.strip()
        if result in ["AM", "PM", "both"]:
            return result
        return "both"
    except Exception as e:
        print(f"[detect_routine_preference] Gemini call failed: {e}")
        return "both"


def run_conflict_check(client, ingredient_collection, products, profile):
    conflicts_found = []
    beneficial_found = []
    allergy_flags = []

    all_ingredients = set()
    for p in products:
        for ing in p["ingredients"].lower().split(", "):
            all_ingredients.add(ing.strip())
    all_ingredients_list = list(all_ingredients)

    # Step 1: hardcoded rules (fast)
    for ing in all_ingredients_list:
        if ing in conflict_lookup:
            for conflict in conflict_lookup[ing]:
                partner = conflict["conflicts_with"]
                if partner in all_ingredients:
                    pair = tuple(sorted([ing, partner]))
                    entry = {"ingredient_a": pair[0], "ingredient_b": pair[1],
                             "reason": conflict["reason"], "source": "rules"}
                    if entry not in conflicts_found:
                        conflicts_found.append(entry)

    # Step 2: RAG fallback for ingredients not in hardcoded rules
    rag_conflicts = rag_conflict_check(client, ingredient_collection, all_ingredients_list)
    for c in rag_conflicts:
        pair = tuple(sorted([c["ingredient_a"], c["ingredient_b"]]))
        entry = {"ingredient_a": pair[0], "ingredient_b": pair[1],
                 "reason": c["reason"], "source": "rag"}
        if entry not in conflicts_found:
            conflicts_found.append(entry)

    # Beneficial pairs
    for ing in all_ingredients_list:
        if ing in beneficial_lookup:
            for pair in beneficial_lookup[ing]:
                partner = pair["pairs_with"]
                if partner in all_ingredients:
                    sorted_pair = tuple(sorted([ing, partner]))
                    entry = {"ingredient_a": sorted_pair[0], "ingredient_b": sorted_pair[1],
                             "benefit": pair["benefit"]}
                    if entry not in beneficial_found:
                        beneficial_found.append(entry)

    # Allergy check
    user_allergies = [a.lower() for a in (profile.get("allergies") or [])]
    for p in products:
        for allergen in user_allergies:
            if allergen in p["ingredients"].lower():
                allergy_flags.append({"product": p["name"], "allergen": allergen})

    return {"conflicts": conflicts_found, "allergy_flags": allergy_flags, "beneficial": beneficial_found}


def full_pipeline(client, product_collection, ingredient_collection, user_input, products_per_type=3):
    """Run the full skincare recommendation pipeline.

    Agent 1 (Skin Profile) -> Agent 2 (Product Retrieval) -> Agent 3 (Conflict Check) -> Agent 4 (Routine Builder)

    Args:
        client: Google GenAI client instance.
        product_collection: ChromaDB collection of skincare products.
        ingredient_collection: ChromaDB collection of INCI ingredients.
        user_input: Raw natural language string from the user.
        products_per_type: Number of products to retrieve per category (default 3).

    Returns:
        dict with keys: profile, retrieved_products, conflict_report, routine, raw_input, routine_preference.
    """

    # Step 0: Detect routine preference before any agent runs
    routine_pref = detect_routine_preference(client, user_input)
    print(f"Routine preference detected: {routine_pref}")

    # Agent 1: Parse user input into structured skin profile
    profile = skin_profile_agent(client, user_input)

    # Agent 2: Retrieve candidate products based on routine preference
    am_types = ["Cleanser", "Serum", "Moisturizer", "Sun protect"]
    pm_types = ["Cleanser", "Serum", "Moisturizer", "Retinol", "Face oil"]

    if routine_pref == "AM":
        product_types = am_types
    elif routine_pref == "PM":
        product_types = pm_types
    else:
        product_types = list(set(am_types + pm_types))  # all unique types for both

    skin_type = profile.get("skin_type", "")
    concerns = " ".join(profile.get("concerns") or [])

    all_products = []
    for ptype in product_types:
        query = f"{skin_type} skin {concerns} {ptype.lower()}"
        results = search_products(product_collection, query, n_results=products_per_type, product_type=ptype)
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            all_products.append({
                "name": meta["name"],
                "brand": meta["brand"],
                "product_type": meta["product_type"],
                "ingredients": results["documents"][0][i],
                "price": meta["price"],
                # Tag each product with its intended time of use
                "routine_time": "AM" if ptype == "Sun protect" else
                               "PM" if ptype in ["Retinol", "Face oil"] else "both"
            })

    # Agent 3: Check for conflicts and allergies
    conflict_report = run_conflict_check(client, ingredient_collection, all_products, profile)

    # Agent 4: Build the routine (passes routine_pref so Gemini structures AM/PM correctly)
    routine = routine_builder_agent(client, profile, all_products, conflict_report, routine_pref=routine_pref)

    return {
        "raw_input": user_input,
        "profile": profile,
        "retrieved_products": all_products,
        "conflict_report": conflict_report,
        "routine": routine,
        "routine_preference": routine_pref
    }
