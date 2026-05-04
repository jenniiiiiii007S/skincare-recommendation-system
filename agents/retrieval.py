import json
import re

def search_products(product_collection, query, n_results=5, product_type=None, skin_type=None):
    """Search the product vector database using a natural language query.

    Args:
        product_collection: ChromaDB collection of skincare products.
        query: Natural language search string.
        n_results: Number of products to return.
        product_type: Optional filter, such as 'Cleanser' or 'Moisturizer'.
        skin_type: Optional skin type to filter results, such as 'oily' or 'dry'.

    Returns:
        ChromaDB query results dict with ids, documents, metadatas, distances.
    """
    where_filter = {"product_type": product_type} if product_type else None

    # Fetch extra candidates so we still have enough products after skin type filtering
    fetch_n = n_results * 3 if skin_type else n_results

    results = product_collection.query(
        query_texts=[query],
        n_results=fetch_n,
        where=where_filter
    )

    if not skin_type:
        return {
            "ids": [results["ids"][0][:n_results]],
            "documents": [results["documents"][0][:n_results]],
            "metadatas": [results["metadatas"][0][:n_results]],
            "distances": [results["distances"][0][:n_results]]
        }

    skin_type_lower = skin_type.lower().strip()
    filtered = {"ids": [], "documents": [], "metadatas": [], "distances": []}
    fallback = {"ids": [], "documents": [], "metadatas": [], "distances": []}

    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        st = meta.get("skin_types", "not specified").lower()

        if st == "not specified" or skin_type_lower in st:
            filtered["ids"].append(results["ids"][0][i])
            filtered["documents"].append(results["documents"][0][i])
            filtered["metadatas"].append(meta)
            filtered["distances"].append(results["distances"][0][i])
        else:
            fallback["ids"].append(results["ids"][0][i])
            fallback["documents"].append(results["documents"][0][i])
            fallback["metadatas"].append(meta)
            fallback["distances"].append(results["distances"][0][i])

        if len(filtered["ids"]) >= n_results:
            break

    if len(filtered["ids"]) < n_results:
        needed = n_results - len(filtered["ids"])
        filtered["ids"] += fallback["ids"][:needed]
        filtered["documents"] += fallback["documents"][:needed]
        filtered["metadatas"] += fallback["metadatas"][:needed]
        filtered["distances"] += fallback["distances"][:needed]

    return {
        "ids": [filtered["ids"]],
        "documents": [filtered["documents"]],
        "metadatas": [filtered["metadatas"]],
        "distances": [filtered["distances"]]
    }
