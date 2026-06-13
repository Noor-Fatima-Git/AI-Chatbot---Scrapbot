"""
Build food.db from food.json (Master Plan Phase 2).
Run from project root: python -m data.build_food_db
"""

import json
import os
import re
import sqlite3


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_price_min(price_str):
    if not price_str or not isinstance(price_str, str):
        return 999999
    numbers = re.findall(r"[\d.]+", price_str)
    if not numbers:
        return 999999
    try:
        return float(numbers[0])
    except (ValueError, TypeError):
        return 999999


def main():
    root = get_project_root()
    json_path = os.path.join(root, "data", "food.json")
    db_path = os.path.join(root, "data", "food.db")
    if not os.path.exists(json_path):
        print(f"Not found: {json_path}")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS restaurants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            city TEXT NOT NULL,
            food_type TEXT NOT NULL,
            area TEXT,
            distance TEXT,
            price_range TEXT,
            price_min REAL,
            rating REAL
        )
    """)
    cur.execute("DELETE FROM restaurants")
    food_places = data.get("food_places", {})
    for food_type, cities in food_places.items():
        for city, places in cities.items():
            for p in places:
                if isinstance(p, dict):
                    price_min = parse_price_min(p.get("price_range", ""))
                    rating = p.get("rating")
                    if rating is not None:
                        try:
                            rating = float(rating)
                        except (ValueError, TypeError):
                            rating = None
                    cur.execute(
                        "INSERT INTO restaurants (name, city, food_type, area, distance, price_range, price_min, rating) VALUES (?,?,?,?,?,?,?,?)",
                        (
                            p.get("name", ""),
                            city,
                            food_type,
                            p.get("area", ""),
                            p.get("distance", ""),
                            p.get("price_range", ""),
                            price_min,
                            rating,
                        ),
                    )
    conn.commit()
    count = cur.execute("SELECT COUNT(*) FROM restaurants").fetchone()[0]
    conn.close()
    print(f"Built {db_path} with {count} restaurants.")


if __name__ == "__main__":
    main()
