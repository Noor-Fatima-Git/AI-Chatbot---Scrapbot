"""
Migrate all JSON data to PostgreSQL domain_items table.
Run: py data/migrate_to_postgres.py
"""
import asyncio, json, hashlib, uuid, sqlite3
from pathlib import Path
from dotenv import load_dotenv
import sys

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from db.database import engine

DATA_DIR = Path(__file__).resolve().parent

def mid(d, v):
    return hashlib.md5(f"{d}:{v}".encode()).hexdigest()

async def insert(conn, rows, label):
    ok = 0
    for r in rows:
        try:
            await conn.execute(text("""
                INSERT INTO domain_items (id, domain, item_id, title, url, metadata_json, scraped_at)
                VALUES (:id, :domain, :item_id, :title, '', cast(:meta AS jsonb), NOW())
                ON CONFLICT (item_id) DO NOTHING
            """), {"id": str(uuid.uuid4()), "domain": r["domain"],
                   "item_id": r["item_id"], "title": r["title"],
                   "meta": json.dumps(r["meta"])})
            ok += 1
        except Exception:
            pass
    print(f"  {label}: {ok} items")
    return ok

async def migrate():
    rows_all = {"food": [], "jobs": [], "automobiles": [], "products": [], "travel": []}

    # FOOD
    data = json.loads((DATA_DIR / "food.json").read_text())
    for food_type, cities in data.get("food_places", {}).items():
        for city, places in (cities.items() if isinstance(cities, dict) else []):
            for item in (places if isinstance(places, list) else []):
                if isinstance(item, dict):
                    n = item.get("name", "?")
                    rows_all["food"].append({
                        "domain": "food", "item_id": mid("food", n+city+food_type),
                        "title": n, "meta": {**item, "city": city, "food_type": food_type}
                    })

    # JOBS - structure: jobs_db → job_title → [list of companies]
    data = json.loads((DATA_DIR / "jobs.json").read_text())
    for job_title, companies in data.get("jobs_db", {}).items():
        for item in (companies if isinstance(companies, list) else []):
            if isinstance(item, dict):
                company = item.get("company", "")
                city = item.get("city", "")
                rows_all["jobs"].append({
                    "domain": "jobs",
                    "item_id": mid("jobs", job_title + company + city),
                    "title": f"{job_title.title()} at {company}",
                    "meta": {**item, "job_title": job_title}
                })

    # AUTOMOBILES - structure: cars → car_type → models → [list]
    data = json.loads((DATA_DIR / "automobiles.json").read_text())
    for car_type, info in data.get("cars", {}).items():
        for item in (info.get("models", []) if isinstance(info, dict) else []):
            if isinstance(item, dict):
                n = item.get("name", "?")
                rows_all["automobiles"].append({
                    "domain": "automobiles",
                    "item_id": mid("autos", n + car_type),
                    "title": n,
                    "meta": {**item, "car_type": car_type,
                             "fuel": info.get("fuel", []),
                             "price_range": info.get("price", "")}
                })

    # PRODUCTS - structure: products → category → [list]
    data = json.loads((DATA_DIR / "products.json").read_text())
    for category, items in data.get("products", {}).items():
        for item in (items if isinstance(items, list) else []):
            if isinstance(item, dict):
                n = item.get("name", "?")
                rows_all["products"].append({
                    "domain": "products",
                    "item_id": mid("products", n + category),
                    "title": n,
                    "meta": {**item, "category_type": category}
                })

    # TRAVEL - structure: trips_places → place → {about, hotels, activities}
    data = json.loads((DATA_DIR / "trips.json").read_text())
    for place_name, info in data.get("trips_places", {}).items():
        if isinstance(info, dict):
            rows_all["travel"].append({
                "domain": "travel",
                "item_id": mid("travel", place_name),
                "title": place_name.title(),
                "meta": {
                    "place_name": place_name,
                    "about": info.get("about", ""),
                    "best_time": info.get("best_time", ""),
                    "hotels": info.get("hotels", []),
                    "activities": info.get("activities", [])
                }
            })

    # FOOD SQLite
    food_db = DATA_DIR / "food.db"
    if food_db.exists():
        try:
            con = sqlite3.connect(str(food_db))
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            for table in tables:
                try:
                    cur.execute(f"SELECT * FROM {table} LIMIT 200")
                    for r in cur.fetchall():
                        item = dict(r)
                        n = item.get("name", "?")
                        rows_all["food"].append({
                            "domain": "food",
                            "item_id": mid("food_db", n + str(item.get("id","")) + str(item.get("area",""))),
                            "title": n,
                            "meta": item
                        })
                except Exception:
                    pass
            con.close()
        except Exception as e:
            print(f"  food.db error: {e}")

    async with engine.begin() as conn:
        total = 0
        for domain, rows in rows_all.items():
            total += await insert(conn, rows, domain)

        # REAL ESTATE
        re_path = DATA_DIR / "real_estate.json"
        if re_path.exists():
            data = json.loads(re_path.read_text(encoding="utf-8"))
            rows = []
            for city, listings in data.get("real_estate", {}).items():
                if isinstance(listings, list):
                    for item in listings:
                        if isinstance(item, dict):
                            title = item.get("title") or "Unknown"
                            rows.append({
                                "domain": "real_estate",
                                "item_id": mid("real_estate", title + city),
                                "title": title,
                                "url": "",
                                "meta": {**item, "city": city},
                            })
            total += await insert(conn, rows, "real_estate")

        print(f"Total: {total} items migrated to local PostgreSQL!")

if __name__ == "__main__":
    asyncio.run(migrate())