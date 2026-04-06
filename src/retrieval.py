import json
import re

def search_products(product_collection, query, n_results=5, product_type=None):
    """Search the product vector database using a natural language query.

    Args:
        product_collection: ChromaDB collection of skincare products.
        query: Natural language search string.
        n_results: Number of products to return.
        product_type: Optional filter (e.g., 'Cleanser', 'Moisturizer').

    Returns:
        ChromaDB query results dict with ids, documents, metadatas, distances.
    """
    filter = {"product_type": product_type} if product_type else None

    results = product_collection.query(
        query_texts=[query],
        n_results=n_results,
        where=filter
    )
    return results
