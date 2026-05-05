import pandas as pd
import ast
import chromadb

DATA_DIR = "data"

# lookfantastic is a UK retailer, prices are in GBP - convert to USD
GBP_TO_USD = 1.27   # approximate exchange rate, update if you want a different one

def parse_ingredients_list(ingred_str):
    """Parse ingredient list from string representation of Python list."""
    try:
        ingredients = ast.literal_eval(ingred_str)
        return ", ".join([i.strip().lower() for i in ingredients])
    except (ValueError, SyntaxError, TypeError):
        return ""


def extract_brand(name):
    """Extract brand name from product name for lookfantastic products."""
    brand_mapping = {
        "The Ordinary": "The Ordinary",
        "CeraVe": "CeraVe",
        "AMELIORATE": "AMELIORATE",
        "La Roche-Posay": "La Roche-Posay",
        "Peter Thomas Roth": "Peter Thomas Roth",
        "Paula's Choice": "Paula's Choice",
        "Drunk Elephant": "Drunk Elephant",
    }
    if pd.isna(name):
        return "Unknown"
    for brand in brand_mapping:
        if brand.lower() in name.lower():
            return brand_mapping[brand]
    parts = name.split(" ")
    return " ".join(parts[:2]) if len(parts) >= 2 else parts[0]


def clean_list_string(val):
    """Parse string representation of list from INCI dataset."""
    if pd.isna(val):
        return ""
    try:
        items = ast.literal_eval(val)
        return ", ".join([i.strip() for i in items if i.strip()])
    except:
        return str(val).strip()

def convert_gbp_to_usd(price_val):
    """Convert a GBP price string or number to a USD-formatted string."""
    if pd.isna(price_val):
        return price_val
    cleaned = str(price_val).replace("£", "").replace("$", "").replace(",", "").strip()
    try:
        gbp = float(cleaned)
        usd = round(gbp * GBP_TO_USD, 2)
        return f"${usd}"
    except (ValueError, TypeError):
        return price_val

def format_usd(price_val):
    """Normalize a USD price to a consistent $XX.XX string format."""
    if pd.isna(price_val):
        return price_val
    cleaned = str(price_val).replace("$", "").replace(",", "").strip()
    try:
        return f"${float(cleaned)}"
    except (ValueError, TypeError):
        return price_val

def load_and_clean_data():
    """Load all datasets, clean, merge, and return unified dataframes."""

    # Load raw datasets
    skincare_df = pd.read_csv(f"{DATA_DIR}/skincare_products_clean.csv")
    cosmetics_df = pd.read_csv(f"{DATA_DIR}/cosmetics.csv")
    inci_df = pd.read_csv(f"{DATA_DIR}/ingredientsList.csv")

    # Clean skincare_products_clean
    skincare_df["ingredients_clean"] = skincare_df["clean_ingreds"].apply(parse_ingredients_list)
    skincare_df_clean = skincare_df[["product_name", "product_type", "ingredients_clean", "price"]].copy()
    skincare_df_clean.columns = ["name", "product_type", "ingredients", "price"]
    skincare_df_clean["brand"] = "Unknown"
    skincare_df_clean["source"] = "lookfantastic"
    skincare_df_clean["brand"] = skincare_df_clean["name"].apply(lambda x: x.split(" ")[0] if pd.notna(x) else "Unknown")
    for col in ["combination", "dry", "normal", "oily", "sensitive"]:
        skincare_df_clean[col] = None
        
    skincare_df_clean["price"] = skincare_df_clean["price"].apply(convert_gbp_to_usd)
    
    # Clean cosmetics dataset
    cosmetics_df["ingredients_clean"] = cosmetics_df["Ingredients"].apply(
        lambda x: ", ".join([i.strip().lower() for i in str(x).split(",")]) if pd.notna(x) else ""
    )
    cosmetics_df_clean = cosmetics_df[["Name", "Label", "ingredients_clean", "Price", "Brand",
                                        "Combination", "Dry", "Normal", "Oily", "Sensitive"]].copy()
    cosmetics_df_clean.columns = ["name", "product_type", "ingredients", "price", "brand",
                                   "combination", "dry", "normal", "oily", "sensitive"]
    cosmetics_df_clean["source"] = "sephora"

    cosmetics_df_clean["price"] = cosmetics_df_clean["price"].apply(format_usd)

    # Merge
    products_df = pd.concat([skincare_df_clean, cosmetics_df_clean], ignore_index=True)

    # Fix brand extraction for lookfantastic products
    mask = products_df["source"] == "lookfantastic"
    products_df.loc[mask, "brand"] = products_df.loc[mask, "name"].apply(extract_brand)

    # Standardize product type names
    product_type_mapping = {
        "Moisturiser": "Moisturizer",
        "Eye Care": "Eye cream",
    }
    exclude_types = ["Body Wash", "Bath Salts", "Bath Oil"]
    products_df = products_df[~products_df["product_type"].isin(exclude_types)].reset_index(drop=True)
    products_df["product_type"] = products_df["product_type"].replace(product_type_mapping)
    products_df = products_df[products_df["ingredients"].str.len() > 0].reset_index(drop=True)

    # Clean INCI
    inci_df["good_for"] = inci_df["who_is_it_good_for"].apply(clean_list_string)
    inci_df["avoid_for"] = inci_df["who_should_avoid"].apply(clean_list_string)
    inci_df["full_description"] = inci_df.apply(lambda row:
        f"Ingredient: {row['name']}. "
        f"Description: {row['short_description'] if pd.notna(row['short_description']) else 'N/A'}. "
        f"Function: {row['what_does_it_do'] if pd.notna(row['what_does_it_do']) else 'N/A'}. "
        f"Good for: {row['good_for']}. "
        f"Avoid if: {row['avoid_for']}.",
        axis=1
    )

    print(f"Loaded {len(products_df)} products and {len(inci_df)} ingredients.")
    return products_df, inci_df


def build_vector_database(products_df, inci_df):
    """Create ChromaDB collections for products and ingredients."""

    chroma_client = chromadb.Client()

    # Product collection
    try:
        chroma_client.delete_collection("skincare_products")
    except:
        pass

    product_collection = chroma_client.create_collection(
        name="skincare_products",
        metadata={"hnsw:space": "cosine"}
    )

    product_documents = []
    product_metadata = []
    product_ids = []

    for idx, row in products_df.iterrows():
        skin_types = []
        for st in ["combination", "dry", "normal", "oily", "sensitive"]:
            if row.get(st) == 1:
                skin_types.append(st)
        skin_type_str = ", ".join(skin_types) if skin_types else "not specified"

        doc = (
            f"Product: {row['name']}. "
            f"Brand: {row['brand']}. "
            f"Type: {row['product_type']}. "
            f"Price: {row['price']}. "
            f"Suitable for skin types: {skin_type_str}. "
            f"Ingredients: {row['ingredients']}."
        )
        product_documents.append(doc)
        product_metadata.append({
            "name": str(row["name"]),
            "brand": str(row["brand"]),
            "product_type": str(row["product_type"]),
            "price": str(row["price"]),
            "source": str(row["source"]),
            "skin_types": skin_type_str
        })
        product_ids.append(f"product_{idx}")

    batch_size = 500
    for i in range(0, len(product_documents), batch_size):
        end = min(i + batch_size, len(product_documents))
        product_collection.add(
            documents=product_documents[i:end],
            metadatas=product_metadata[i:end],
            ids=product_ids[i:end]
        )

    # Ingredient collection
    try:
        chroma_client.delete_collection("skincare_ingredients")
    except:
        pass

    ingredient_collection = chroma_client.create_collection(
        name="skincare_ingredients",
        metadata={"hnsw:space": "cosine"}
    )

    ingredient_ids = []
    idx = 0
    while idx < len(inci_df):
        ingredient_ids.append("ingredient_" + str(idx))
        idx = idx + 1

    ingredient_metadata = []
    for idx, row in inci_df.iterrows():
        ingredient_metadata.append({
            "name": str(row["name"]) if pd.notna(row["name"]) else "",
            "good_for": str(row["good_for"]),
            "avoid_for": str(row["avoid_for"])
        })

    ingredient_collection.add(
        documents=inci_df["full_description"].tolist(),
        metadatas=ingredient_metadata,
        ids=ingredient_ids
    )

    print(f"Indexed {product_collection.count()} products and {ingredient_collection.count()} ingredients.")
    return product_collection, ingredient_collection
