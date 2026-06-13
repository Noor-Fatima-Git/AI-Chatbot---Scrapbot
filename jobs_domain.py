"""
Jobs Domain Handler for Scrapbot.
Upgraded with clickable URLs and 3-step recommendation flow.
"""

import json
import os
from typing import Tuple, Optional, Dict, Any
from domains.base_domain import BaseDomain
from nlp.entity_extractor import entity_extractor
from utils.logger import logger
from utils.helpers import get_project_root, fuzzy_match_choice


class JobsDomain(BaseDomain):

    def __init__(self, context_manager):
        super().__init__("jobs", context_manager)
        self.jobs_data = self._load_jobs_data()

    def _load_jobs_data(self) -> dict:
        data_path = os.path.join(get_project_root(), "data", "jobs.json")
        try:
            if os.path.exists(data_path):
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.logger.info("Jobs data loaded from JSON file")
                    return data
        except Exception as e:
            logger.log_error(e, "JobsDomain._load_jobs_data")
        return {"jobs_db": {}}

    def can_handle(self, intent: Optional[str], user_input: str) -> bool:
        text = user_input.lower()
        if intent == "job":
            return True
        if self.is_active():
            return True
        job_keywords = [
            "job", "jobs", "vacancy", "hiring", "career",
            "position", "employment", "apply", "salary",
            "recruitment", "openings", "work", "nokri", "naukri"
        ]
        return any(keyword in text for keyword in job_keywords)

    def _format_job(self, title: str, r: dict, show_link: bool = False) -> str:
        """Format a single job listing."""
        company = r.get("company", "N/A")
        city = str(r.get("city", "N/A")).title()
        sal = r.get("salary")
        salary_str = f"PKR {sal:,}" if isinstance(sal, (int, float)) else str(sal or "N/A")
        url = r.get("url", "")

        line = f"💼 {title} at {company} in {city} ({salary_str})"
        if show_link and url:
            line += f"\n   🔗 {url}"
        return line

    def handle(self, user_input: str, entities: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[str], list]:
        text = user_input.lower()

        # Check if user is asking for best/top/link
        asking_best = any(x in text for x in ["best", "top", "which one", "recommend", "link", "apply", "url"])

        if entities is None:
            entities = entity_extractor.extract_entities(text, domain="jobs")

        # Update context
        if entities.get("job_title"):
            self.context_manager.set_entity("job_title", entities["job_title"])
        city = entities.get("job_city") or entities.get("city")
        if city:
            self.context_manager.set_entity("job_city", city)

        job_title = self.context_manager.get_entity("job_title") or entities.get("job_title")
        job_city = self.context_manager.get_entity("job_city") or entities.get("city")

        if job_title:
            jobs_db = self.jobs_data.get("jobs_db", {})
            keys = list(jobs_db.keys())

            # Fuzzy match job title
            resolved = job_title.lower() if job_title.lower() in keys else (
                fuzzy_match_choice(job_title, keys, cutoff=0.45) or job_title.lower()
            )
            results = jobs_db.get(resolved, [])

            # Filter by city
            if job_city:
                city_results = [r for r in results if isinstance(r, dict) and r.get("city", "").lower() == job_city.lower()]
                if city_results:
                    results = city_results

            results = results[:8]

            if results:
                display_title = resolved.title()
                fuzzy_hint = f"Showing results for {display_title}\n" if resolved.lower() != job_title.lower() else ""

                # Step 3 — user wants best → show top 1 with link
                if asking_best:
                    top = results[0]
                    url = top.get("url", "")
                    company = top.get("company", "N/A")
                    city_name = str(top.get("city", "")).title()
                    sal = top.get("salary")
                    salary_str = f"PKR {sal:,}" if isinstance(sal, (int, float)) else str(sal or "")
                    response = f"🏆 Best match for {display_title}:\n\n"
                    response += f"💼 {display_title} at {company}\n"
                    response += f"📍 {city_name}\n"
                    response += f"💰 {salary_str}\n"
                    if url:
                        response += f"🔗 Apply here: {url}"
                    self.context_manager.reset()
                    return response, None, []

                # Step 1 & 2 — show listings
                formatted = []
                for r in results:
                    if isinstance(r, dict):
                        formatted.append(self._format_job(display_title, r, show_link=True))
                    else:
                        formatted.append(f"💼 {display_title} - {r}")

                reason = f"job_title={resolved}, city={job_city or 'any'}, count={len(results)}"
                logger.logger.info(f"GROUNDING: {reason}")
                self.context_manager.reset()
                return fuzzy_hint + "\n".join(formatted), None, []
            else:
                self.context_manager.reset()
                return f"Sorry, no {job_title} jobs found. Try a different title or city.", None, []

        # Fallback — ask what job
        available = list(self.jobs_data.get("jobs_db", {}).keys())
        jobs_list = ", ".join(j.title() for j in available[:6])
        suggestions = [j.title() for j in available[:6]]
        return f"What job are you looking for?\n{jobs_list}?", "jobs", suggestions