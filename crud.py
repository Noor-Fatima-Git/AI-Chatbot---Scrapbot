"""
Database CRUD operations for ScrapBot.
"""
import json
from datetime import datetime
from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert
from db.models import UserInteraction, Recommendation, DomainItem





async def log_interaction(
    db: AsyncSession,
    domain: str,
    query: str,
    entities: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
):
    """Save every chat message to user_interactions table."""
    stmt = insert(UserInteraction).values(
        user_id=None,
        domain=domain or "unknown",
        query=query,
        entities_json=entities or {},
        timestamp=datetime.utcnow(),
    )
    await db.execute(stmt)
    await db.commit()


async def save_recommendation(
    db: AsyncSession,
    domain: str,
    item_id: str,
    item_data: Dict[str, Any],
    score: float = 0.0,
):
    """Save a recommendation to recommendations table."""
    stmt = insert(Recommendation).values(
        domain=domain,
        item_id=item_id,
        item_data_json=item_data,
        score=score,
        created_at=datetime.utcnow(),
    )
    await db.execute(stmt)
    await db.commit()


async def upsert_domain_item(
    db: AsyncSession,
    domain: str,
    item_id: str,
    title: str,
    url: str,
    metadata: Dict[str, Any],
):
    """Save scraped item to domain_items table."""
    stmt = insert(DomainItem).values(
        domain=domain,
        item_id=item_id,
        title=title,
        url=url,
        metadata_json=metadata,
        scraped_at=datetime.utcnow(),
    ).on_conflict_do_update(
        index_elements=["item_id"],
        set_=dict(
            title=title,
            url=url,
            metadata_json=metadata,
            scraped_at=datetime.utcnow(),
        )
    )
    await db.execute(stmt)
    await db.commit()