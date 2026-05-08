import os
import sys
import tempfile
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from google import genai
from data_loader import load_and_clean_data, build_vector_database
from pipeline import full_pipeline


st.set_page_config(page_title="Facial Skincare Routine Assistant", page_icon="🧴", layout="centered")

st.title("🧴 Facial Skincare Routine Assistant")
st.caption("Tell me about your facial skincare needs. You can upload a selfie, mention allergies, and set a budget."")

@st.cache_resource(show_spinner="Loading data and building vector database...")
def init_pipeline():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY is not set.")
        st.stop()
    client = genai.Client(api_key=api_key)
    products_df, inci_df = load_and_clean_data()
    product_collection, ingredient_collection = build_vector_database(products_df, inci_df)
    return client, product_collection, ingredient_collection


client, product_collection, ingredient_collection = init_pipeline()

INITIAL_MESSAGE = {
    "role": "assistant",
    "content": "Hi! Tell me about your skin — type, concerns, allergies, and budget. You can upload a selfie too.\n\nExamples:\n- *I have oily acne-prone skin and want a routine under $80*\n- *Dry sensitive skin, allergic to fragrance, drugstore budget*\n- *Anti-aging routine for combination skin, around $150*",
}


def format_routine(result):
    if "error" in result:
        return result["error"]

    parts = []
    profile = result.get("profile", {}) or {}

    if profile.get("image_observations"):
        parts.append("**📸 From your photo:** " + profile["image_observations"])

    # FIX 2: double newlines so each profile field renders on its own line
    profile_lines = []
    if profile.get("skin_type"):
        profile_lines.append("Skin type: " + str(profile["skin_type"]))
    if profile.get("concerns"):
        profile_lines.append("Concerns: " + ", ".join(profile["concerns"]))
    if profile.get("allergies"):
        profile_lines.append("Allergies: " + ", ".join(profile["allergies"]))
    if profile_lines:
        parts.append("**Profile detected**\n\n" + "\n\n".join(profile_lines))

    budget = result.get("budget_profile", {}) or {}
    if budget.get("overall_limit"):
        parts.append("**Budget:** $" + str(int(budget["overall_limit"])) + " total")
    elif budget.get("tier") and budget["tier"] != "any":
        parts.append("**Budget tier:** " + budget["tier"].replace("_", " "))

    # FIX 1: build price lookup from retrieved products
    price = price_lookup.get(step.get("product_name", ""), "")
    if price:
        # strip trailing .0 for cleaner display
        price_clean = price.rstrip("0").rstrip(".") if "." in price else price
        line += " — " + price_clean

    routine = result.get("routine", {}) or {}

    if routine.get("morning_routine"):
        parts.append("### ☀️ Morning Routine")
        for step in routine["morning_routine"]:
            line = ("**" + str(step.get("step", "")) + ". " +
                    str(step.get("product_type", "")) + "** — " +
                    str(step.get("product_name", "")))
            if step.get("brand"):
                line += " *(" + str(step["brand"]) + ")*"
            price = price_lookup.get(step.get("product_name", ""), "")
            if price:
                line += " — " + str(price)
            parts.append(line)
            if step.get("why"):
                parts.append("> " + step["why"])

    if routine.get("evening_routine"):
        parts.append("### 🌙 Evening Routine")
        for step in routine["evening_routine"]:
            line = ("**" + str(step.get("step", "")) + ". " +
                    str(step.get("product_type", "")) + "** — " +
                    str(step.get("product_name", "")))
            if step.get("brand"):
                line += " *(" + str(step["brand"]) + ")*"
            price = price_lookup.get(step.get("product_name", ""), "")
            if price:
                line += " — " + str(price)
            parts.append(line)
            if step.get("why"):
                parts.append("> " + step["why"])

    if routine.get("warnings"):
        parts.append("### ⚠️ Warnings")
        for w in routine["warnings"]:
            parts.append("- " + str(w))

    if routine.get("notes"):
        parts.append("### 💡 Notes")
        for n in routine["notes"]:
            parts.append("- " + str(n))

    if budget.get("budget_fallbacks"):
        parts.append("### 💰 Budget Notice")
        for fb in budget["budget_fallbacks"]:
            parts.append(
                "- **" + fb["product_type"] + "**: cheapest available is $" +
                str(fb["actual_price"]) + " (your limit was $" +
                str(int(fb["user_limit"])) + ")"
            )

    return "\n\n".join(parts) if parts else "Could not generate a routine. Try rephrasing."


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


if "messages" not in st.session_state:
    st.session_state.messages = [INITIAL_MESSAGE]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image_path"):
            try:
                st.image(msg["image_path"], width=200)
            except Exception:
                pass
        st.markdown(msg["content"])


with st.sidebar:
    st.header("Optional facial photo")
    uploaded = st.file_uploader(
        "Upload a clear photo of your face",
        type=["png", "jpg", "jpeg", "webp"],
        help="Photos go to Google Gemini for analysis."
    )
    if uploaded is not None:
        st.image(uploaded, caption="Will be analyzed with your next message", width=200)
    st.caption("Privacy: photos are not stored by this app, but processed by Google's Gemini API.")
    st.caption("This tool is for educational purposes and is not medical advice.")

    st.divider()

    # FIX 3: Start over button
    if st.button("🔄 Start over", use_container_width=True):
        st.session_state.messages = [INITIAL_MESSAGE]
        st.rerun()


user_message = st.chat_input("Describe your skin...")

if user_message:
    image_path = save_uploaded_file(uploaded) if uploaded is not None else None

    st.session_state.messages.append({
        "role": "user",
        "content": user_message,
        "image_path": image_path,
    })
    with st.chat_message("user"):
        if image_path:
            try:
                st.image(image_path, width=200)
            except Exception:
                pass
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Building your routine..."):
            try:
                result = full_pipeline(
                    client, product_collection, ingredient_collection,
                    user_input=user_message,
                    image_path=image_path,
                )
                response_text = format_routine(result)
            except Exception as e:
                response_text = "Sorry, something went wrong: " + str(e)

        st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
