import json
import logging
import os
from typing import List, Optional, Union

import requests
from qwen_agent.tools.base import BaseTool, register_tool

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


# --------------- SFT-aligned response formatter ---------------

def _parse_scholar_response(query: str, results: dict) -> str:
    if "organic" not in results:
        return f"No results found for '{query}'. Try with a more general query."

    entries = []
    for idx, page in enumerate(results["organic"], start=1):
        link_info = "no available link"
        if "pdfUrl" in page:
            link_info = "pdfUrl: " + page["pdfUrl"]
        elif "link" in page:
            link_info = "link: " + page["link"]

        parts = [f"{idx}. [{page['title']}]({link_info})"]

        pub_info = page.get("publicationInfo", "")
        if isinstance(pub_info, dict):
            pub_info = pub_info.get("summary", str(pub_info))
        if pub_info:
            parts.append(f"Publication Info: {pub_info}")

        if "year" in page:
            parts.append(f"Date published: {page['year']}")

        cited_by = page.get("citedBy", None)
        if cited_by is not None:
            if isinstance(cited_by, dict):
                cited_by = cited_by.get("total", cited_by.get("count", ""))
            parts.append(f"Cited by: {cited_by}")

        if "snippet" in page:
            parts.append(f"Snippet: {page['snippet']}")

        entry = "\n".join(parts)
        entry = entry.replace("Your browser can't play this video.", "")
        entries.append(entry)

    head = f"A Google Scholar search for '{query}' found {len(entries)} results:\n\n## Scholar Results"
    if entries:
        return head + "\n" + "\n\n".join(entries)
    return head


# --------------- Tool class ---------------

@register_tool("google_scholar", allow_overwrite=True)
class Scholar(BaseTool):
    name = "google_scholar"
    description = "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "The search query.",
                },
                "minItems": 1,
                "description": "The list of search queries for Google Scholar.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        search_mode = os.getenv('SEARCH_MODE', 'multi')
        if search_mode == 'single':
            self.description = "Leverage Google Scholar to retrieve relevant information from academic publications."
            self.parameters = {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for Google Scholar.",
                    },
                },
                "required": ["query"],
            }
        super().__init__(cfg)

    @staticmethod
    def _get_locale_params(query: str) -> dict:
        if any('一' <= c <= '鿿' for c in query):
            return {"location": "China", "gl": "cn", "hl": "zh-cn"}
        return {"location": "United States", "gl": "us", "hl": "en"}

    def google_scholar_search(self, query: str, timeout: int = 10) -> str:
        serper_ak = os.environ.get("SERPER_API_KEY")
        if not serper_ak:
            return "Google Scholar failed: SERPER_API_KEY not set."

        locale = self._get_locale_params(query)
        params = {"q": query, **locale}
        headers = {
            "X-API-KEY": serper_ak,
            "Content-Type": "application/json",
        }

        for attempt in range(5):
            try:
                resp = requests.post(
                    "https://google.serper.dev/scholar",
                    headers=headers,
                    data=json.dumps(params),
                    timeout=timeout,
                    verify=False,
                )
                resp.raise_for_status()
                return _parse_scholar_response(query, resp.json())
            except Exception as e:
                logger.debug(f"Scholar attempt {attempt + 1}/5 failed: {e}")
                if attempt == 4:
                    return "Google Scholar Timeout, return None, Please try again later."

        return "Google Scholar Timeout, return None, Please try again later."

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except Exception:
            return "[google_scholar] Invalid request format: Input must be a JSON object containing 'query' field"

        if isinstance(query, str):
            return self.google_scholar_search(query)

        assert isinstance(query, List)
        responses = [self.google_scholar_search(q) for q in query]
        return "\n=======\n".join(
            r if isinstance(r, str) else str(r) for r in responses
        )
