# Multimodal Retrieval-Augmented AI System for Personalized Skincare Recommendations

**DS-UA 301: Advanced Topics in Data Science — Group 12**  
Trinity Chan · Jennifer Ran · Savanna Thomas


## Overview
A domain-specialized multi-agent pipeline that generates personalized, conflict-free 
skincare routine recommendations. The system accepts both text descriptions and optional facial images as input, retrieves real products from a vector database of 2,418 products, checks for ingredient conflicts and allergens, and builds structured AM/PM routines.

## Pipeline
1. **Skin Profile Agent** — Parses user text and optional facial image into a structured profile using Gemini 2.5 Flash vision
2. **Product Retrieval Agent** — Retrieves candidate products from ChromaDB using embedding similarity, filtered by skin type
3. **Budget Agent** — Extracts budget constraints and filters products by price
4. **Conflict Checker Agent** — Detects ingredient conflicts via hardcoded rules + RAG, flags allergens using synonym mapping
5. **Routine Builder Agent** — Assembles final AM/PM routine using Gemini 2.5 Flash

## Repository Structure
```
├── agents/
│   ├── __init__.py
│   ├── skin_profile.py                  # Agent 1: multimodal input → structured profile
│   ├── retrieval.py                     # Agent 2: ChromaDB semantic search
│   ├── budget_agent.py                  # Agent 3: budget extraction and filtering
│   ├── conflict_checker.py              # Agent 4: conflict detection + allergen flagging
│   └── routine_builder.py               # Agent 5: routine assembly
├── pipeline.py                          # End-to-end pipeline chaining all agents
├── data_loader.py                       # Data cleaning, merging, ChromaDB indexing
├── data/
│   ├── cosmetics.csv                    
│   ├── ingredientsList.csv
│   ├── skincare_products_clean.csv
│   └── README.md                        # Dataset download instructions
├── notebooks/
│   ├── deprecated_build_modules.ipynb   # Legacy notebook, no longer used (deprecated)
│   └── demo.ipynb                       # Runs pipeline with tests and evaluation
└── README.md
```
## Setup
1. Open `notebooks/demo.ipynb` in Google Colab
2. At the top of the notebook, the repo will be cloned automatically using GitHub
3. Add your Gemini API key to Colab Secrets as `GEMINI_API_KEY`
4. Run all cells from top to bottom to:
   - install dependencies
   - load data and build the vector database
   - run pipeline and evaluation
  
Note:
- The project now uses GitHub as the source of truth for all `.py` files.
- Do not run `notebooks/deprecated_build_modules.ipynb` (it is deprecated and only kept for reference).
- To make changes, edit `.py` files directly in GitHub and re-run `notebooks/demo.ipynb`, which will pull latest version.

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

## Evaluation Results (100 test cases)
| Metric | Score |
|---|---|
| Conflict detection accuracy | 58–67% (varies across runs) |
| Allergy detection accuracy | 98–99% |
| Budget omission accuracy | 80–85% (n=20 budget cases) |
| Fitzpatrick fairness — conflict accuracy | 50–70% |
| Fitzpatrick fairness — allergy accuracy | 100% |

**Baseline vs. Pipeline:** Pipeline average 3.83/5 vs. raw Gemini 3.50/5 across 6 comparison cases. Pipeline outperformed on 3/6 cases, with most significant advantage on allergy-flagging cases where the pipeline identifies specific allergen-containing products from the real database — a capability raw Gemini cannot replicate.

**Note on conflict accuracy:** Conflict recall on true conflict cases is 100%. The overall accuracy reflects a deliberate precision/recall tradeoff to eliminate false positives from the RAG layer. Remaining variance across runs is due to retrieval non-determinism — different products are retrieved each run, changing which ingredient pairs the conflict checker sees.

## Requirements
- Google Colab (recommended)
- GitHub (to access project repository)
- Gemini API key with billing enabled (Paid 1 tier)
- `google-genai`, `chromadb`, `pandas` (installed via setup cell in demo.ipynb)

## Interactive Demo
Public Streamlit app:
