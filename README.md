# Multimodal Retrieval-Augmented AI System for Personalized Skincare Recommendations

**DS-UA 301: Advanced Topics in Data Science — Group 12**  
Trinity Chan · Jennifer Ran · Savanna Thomas

## Overview
A domain-specialized multi-agent pipeline that generates personalized, conflict-free 
skincare routine recommendations. The system accepts both text descriptions and facial 
images as input, retrieves real products from a vector database of 2,418 products, 
checks for ingredient conflicts and allergens, and builds structured AM/PM routines.

## Pipeline
1. **Skin Profile Agent** — Parses user text and optional facial image into a structured profile using Gemini 2.5 Flash vision
2. **Product Retrieval Agent** — Retrieves candidate products from ChromaDB using embedding similarity, filtered by skin type
3. **Conflict Checker Agent** — Detects ingredient conflicts via hardcoded rules + RAG, flags allergens using synonym mapping
4. **Budget Agent** — Extracts budget constraints and filters products by price
5. **Routine Builder Agent** — Assembles final AM/PM routine using Gemini 2.5 Flash

## Repository Structure
```
├── agents/
│   ├── __init__.py
│   ├── skin_profile.py       # Agent 1: multimodal input → structured profile
│   ├── retrieval.py          # Agent 2: ChromaDB semantic search
│   ├── conflict_checker.py   # Agent 3: conflict detection + allergen flagging
│   ├── budget_agent.py       # Agent 4: budget extraction and filtering
│   └── routine_builder.py    # Agent 5: routine assembly
├── pipeline.py               # End-to-end pipeline chaining all agents
├── data_loader.py            # Data cleaning, merging, ChromaDB indexing
├── data/
│   ├── cosmetics.csv
│   ├── ingredientsList.csv
│   ├── skincare_products_clean.csv
│   └── README.md             # Dataset download instructions
├── notebooks/
│   ├── build_modules.ipynb   # Writes all .py files to Drive
│   └── demo.ipynb            # Runs the pipeline with tests and evaluation
└── README.md
```
## Setup
1. Clone this repo and open `notebooks/build_modules.ipynb` in Google Colab
2. Mount your Google Drive and run all cells to write the `.py` files
3. Add your Gemini API key to Colab Secrets as `GEMINI_API_KEY`
4. Open `notebooks/demo.ipynb` and run all cells from top to bottom

## Datasets
The following datasets are required and should be placed in your Google Drive at  
`/content/drive/MyDrive/skincare_project/data/`:

| Dataset | Source |
|---|---|
| Skincare Products Clean | [Kaggle](https://www.kaggle.com/datasets/eward96/skincare-products-clean-dataset) |
| Cosmetics (Sephora) | [Kaggle](https://www.kaggle.com/datasets/kingabzpro/cosmetics-datasets) |
| INCI Ingredient List | [Kaggle](https://www.kaggle.com/datasets/amaboh/skin-care-product-ingredients-inci-list) |
| Skin Disease Classification | [Kaggle](https://www.kaggle.com/datasets/trainingdatapro/skin-defects-acne-redness-and-bags-under-the-eyes) |

Image datasets are not included in this repository due to size constraints.

## Evaluation Results (80 test cases) (NEED UPDATE)
| Metric | Score |
|---|---|
| Conflict detection accuracy | 59% |
| Allergy detection accuracy | 99% |
| Fitzpatrick fairness — allergy accuracy | 100% |

Note: Conflict recall on true conflict cases is 100%. The 59% overall accuracy 
reflects a deliberate precision/recall tradeoff to eliminate false positives.

## Requirements
- Google Colab (recommended)
- Gemini API key with billing enabled (Paid 1 tier)
- `google-genai`, `chromadb`, `pandas` (installed via setup cell in demo.ipynb)
