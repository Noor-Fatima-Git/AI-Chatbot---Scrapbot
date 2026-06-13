# AI-Chatbot---Scrapbot

# ScrapBot 🤖

AI-powered conversational chatbot built as part of the **ScrapIt** platform. ScrapBot lets users search for information across multiple domains using natural language in English and Roman Urdu.

---

## Domains Supported

| Domain | What it does |
|---|---|
| 🍕 Food | Find restaurants, cuisines, delivery options |
| 💼 Jobs | Search career opportunities by title and city |
| ✈️ Travel | Explore destinations, hotels, and booking links |
| 🚗 Automobiles | Browse cars by type, brand, and budget |
| 🛍️ E-Commerce | Shop products with budget filtering |
| 🏠 Real Estate | Find properties to rent or buy |

---

## Tech Stack

- **Backend** — FastAPI (Python)
- **Real-time Chat** — WebSocket
- **Database** — PostgreSQL
- **Intent Classification** — Logistic Regression + TF-IDF (scikit-learn)
- **Entity Extraction** — spaCy + custom regex
- **LLM Enrichment** — Groq API (LLaMA 3.3 70B Versatile)
- **Recommendations** — XGBoost (6 domain models)
- **Semantic Search** — Sentence Transformers + FAISS
- **Frontend** — HTML, CSS, JavaScript (WebSocket client)

---

## Project Structure

```
├── chatbot.py              # Main orchestration logic
├── web_app.py              # FastAPI app + WebSocket + UI
├── train_intent_model.py   # Train intent classifier
├── train_xgb_model.py      # Train XGBoost recommendation models
├── intents.csv             # Training data for intent classifier
├── nlp/
│   ├── intent_classifier.py
│   ├── entity_extractor.py
│   ├── groq_enricher.py
│   ├── recommender.py
│   └── embeddings.py
├── domains/
│   ├── food_domain.py
│   ├── jobs_domain.py
│   ├── travel_domain.py
│   ├── automobiles_domain.py
│   ├── products_domain.py
│   └── real_estate_domain.py
├── context/
│   └── context_manager.py
├── db/
│   ├── database.py
│   └── crud.py
├── rag/
│   └── rag_engine.py
├── data/
│   ├── food.json
│   ├── jobs.json
│   ├── automobiles.json
│   ├── products.json
│   ├── trips.json
│   └── real_estate.json
└── utils/
    ├── helpers.py
    └── logger.py
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/scrapbot.git
cd scrapbot
```

### 2. Create virtual environment
```bash
python -m venv env
env\Scripts\activate      # Windows
source env/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Set up environment variables
Create a `.env` file in the root:
```
GROQ_API_KEY=your_groq_api_key
POSTGRES_URL_SYNC=postgresql://username:password@localhost/scrapbot_db
```

### 5. Set up PostgreSQL
```bash
# Create database
createdb scrapbot_db

# Run migrations
python data/migrate_to_postgres.py
```

### 6. Train models
```bash
python train_intent_model.py
python train_xgb_model.py
```

### 7. Run the server
```bash
uvicorn web_app:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

---

## Example Queries

```
mujhe lahore mein sasti biryani chahiye
software engineer job islamabad
honda car under 50 lacs
cheap mobile under 40000
rent flat in islamabad
travel to hunza
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key for LLaMA enrichment |
| `POSTGRES_URL_SYNC` | PostgreSQL connection string |

---

## Integration with ScrapIt

ScrapBot is designed to connect to ScrapIt's live scraped database. Once the scrapers populate the shared PostgreSQL instance with real listings, ScrapBot will automatically serve those results with direct source URLs.

---

## Team

Built as a Final Year Project (FYP) — ScrapBot module developed by **Noor Fatima**, integrated into the larger **ScrapIt** platform.

---

## License

This project is part of an academic Final Year Project.
