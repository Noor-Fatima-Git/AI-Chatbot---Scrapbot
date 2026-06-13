import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from db.database import engine, Base
from db import models

async def init():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        print("✅ Users table created!")
        print("✅ UserInteractions table created!")
        print("✅ Recommendations table created!")
        print("✅ DomainItems table created!")
        print("✅ Database ready on PostgreSQL!")

if __name__ == "__main__":
    asyncio.run(init())