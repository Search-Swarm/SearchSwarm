import json
import logging
import os
from typing import List, Optional, Union

import requests
from qwen_agent.tools.base import BaseTool, register_tool

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class ToolCallFormatError(Exception):
    """Raised when the model produces a tool call whose arguments cannot be parsed."""
    pass

SEARCH_NUM_RESULTS = int(os.getenv('SEARCH_NUM_RESULTS', 10))


# --------------- SFT-aligned response formatter ---------------

def _parse_search_response(query: str, results: dict) -> str:
    if "organic" not in results:
        return f"No results found for '{query}'. Try with a more general query."

    entries = []
    for idx, page in enumerate(results["organic"], start=1):
        header_parts = [f"{idx}. [{page['title']}]({page['link']})"]
        if "date" in page:
            header_parts.append(f"Date published: {page['date']}")
        if "source" in page:
            header_parts.append(f"Source: {page['source']}")
        header = "\n".join(header_parts)

        snippet = page.get("snippet", "")
        entry = f"{header}\n\n{snippet}" if snippet else header
        entry = entry.replace("Your browser can't play this video.", "")
        entries.append(entry)

    head = f"A Google search for '{query}' found {len(entries)} results:\n\n## Web Results"
    if entries:
        return head + "\n" + "\n\n".join(entries)
    return head


# --------------- Tool class ---------------

@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = f"Performs batched web searches: supply an array 'query'; the tool retrieves the top {SEARCH_NUM_RESULTS} results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings. Include multiple complementary search queries in a single call.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        search_mode = os.getenv('SEARCH_MODE', 'multi')
        if search_mode == 'single':
            self.description = "Perform a Google web search then returns a string of the top search results."
            self.parameters = {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
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

    def google_search_with_serp(self, query: str, num: int = SEARCH_NUM_RESULTS, timeout: int = 10) -> str:
        locale = self._get_locale_params(query)

        serper_ak = os.environ.get("SERPER_API_KEY")
        if not serper_ak:
            return "Google search failed: SERPER_API_KEY not set."

        params = {"q": query, "num": num, **locale}
        headers = {
            "X-API-KEY": serper_ak,
            "Content-Type": "application/json",
        }

        for attempt in range(5):
            try:
                resp = requests.post(
                    "https://google.serper.dev/search",
                    headers=headers,
                    data=json.dumps(params),
                    timeout=timeout,
                    verify=False,
                )
                resp.raise_for_status()
                result_json = resp.json()

                return _parse_search_response(query, result_json)
            except Exception as e:
                logger.debug(f"Search attempt {attempt + 1}/5 failed: {e}")
                if attempt == 4:
                    return "Google search Timeout, return None, Please try again later."

        return "Google search Timeout, return None, Please try again later."

    def search_with_serp(self, query: str) -> str:
        return self.google_search_with_serp(query)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except Exception:
            raise ToolCallFormatError(
                "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
            )

        if isinstance(query, str):
            return self.search_with_serp(query)

        assert isinstance(query, List)
        responses = [self.search_with_serp(q) for q in query]
        return "\n=======\n".join(responses)
