import json
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Optional
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from prompt import EXTRACTOR_PROMPT
from openai import OpenAI
import random
from urllib.parse import urlparse, unquote
import time
from transformers import AutoTokenizer
import tiktoken

VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))
IS_EVIDENCE = os.getenv("IS_EVIDENCE", "1") == "1"


@staticmethod
def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""


@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    # The `description` tells the agent the functionality of this tool.
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {
                    "type": "string"
                    },
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
        },
        "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s)."
        }
        },
        "required": ["url", "goal"]
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self._summary_client = OpenAI(
            base_url=os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/") + "/v1",
            api_key=os.environ.get("API_KEY", ""),
            timeout=120.0,
        )

    # The `call` method is the main function of the tool.
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        start_time = time.time()

        # Create log folder if it doesn't exist
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)

        if isinstance(url, str):
            try:
                response = self.readpage_jina(url, goal)
            except Exception as e:
                response = f"Error fetching {url}: {str(e)}"
        else:
            response = []
            assert isinstance(url, List)
            start_time = time.time()
            for u in url:
                if time.time() - start_time > 900:
                    cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    if IS_EVIDENCE:
                        cur_response += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    cur_response += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                else:
                    try:
                        cur_response = self.readpage_jina(u, goal)
                    except Exception as e:
                        cur_response = f"Error fetching {u}: {str(e)}"
                response.append(cur_response)
            response = "\n=======\n".join(response)

        print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip()

    def call_server_requests(self, msgs, max_retries=2):
        model_name = os.environ.get("SUMMARY_MODEL_NAME", "")

        for attempt in range(max_retries):
            try:
                response = self._summary_client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                )

                content = response.choices[0].message.content.strip()

                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string
                        left = content.find('{')
                        right = content.rfind('}')
                        if left != -1 and right != -1 and left <= right:
                            content = content[left:right+1]
                    return content

            except Exception:
                if attempt == (max_retries - 1):
                    return ""
                continue


    def jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina service.

        Args:
            url: The URL to read

        Returns:
            str: The webpage content or error message
        """
        max_retries = 5
        timeout = 50

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers={
                        "Authorization": f"Bearer {os.environ.get('JINA_API_KEY', '')}",
                        "Accept": "text/markdown",
                    },
                    timeout=timeout,
                )
                response.raise_for_status()
                return response.text
            except Exception as e:
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."
                time.sleep(0.5 * (2 ** attempt))

        return "[visit] Failed to read page."

    def html_readpage_jina(self, url: str) -> str:
        max_attempts = 10
        for attempt in range(max_attempts):
            content = self.jina_readpage(url)
            print("jina")
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                return content
            if attempt < max_attempts - 1:
                time.sleep(min(2 ** attempt, 30))
        return "[visit] Failed to read page."

    def _try_extract(self, summary_page_func, content, goal, max_retries):
        """Call extractor model and try to parse the result as JSON.

        Returns:
            dict if successful, None otherwise.
        """
        messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(
            webpage_content=content, goal=goal
        )}]
        raw = summary_page_func(messages, max_retries=max_retries)

        if not raw or len(raw) < 10:
            return None

        if isinstance(raw, str):
            raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _build_error_output(self, url, goal):
        """Build output string for when extraction fails."""
        output = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
        if IS_EVIDENCE:
            output += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
        output += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
        return output

    def readpage_jina(self, url: str, goal: str) -> str:
        """
        Fetch webpage content via jina, then extract structured information
        using the extractor model.

        Retry strategy for extraction:
          Phase 1 - exponential backoff with same content (5 retries: 4/8/16/32/64s)
          Phase 2 - progressive content truncation (3 retries, 70% each round)

        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page

        Returns:
            str: Formatted extraction result or error message
        """
        summary_page_func = self.call_server_requests
        max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 1))

        content = self.html_readpage_jina(url)

        if not content or content.startswith("[visit] Failed to read page.") or content == "[visit] Empty content." or content.startswith("[document_parser]"):
            return self._build_error_output(url, goal)

        content = truncate_to_tokens(content, max_tokens=95000)

        # Phase 1: exponential backoff retries with same content
        backoff_delays = [4, 8, 16, 32, 64]
        raw_dict = self._try_extract(summary_page_func, content, goal, max_retries)
        for i, delay in enumerate(backoff_delays):
            if raw_dict is not None:
                break
            print(f"[visit] Extractor attempt {i + 1} failed for url[{url}], retrying in {delay}s")
            time.sleep(delay)
            raw_dict = self._try_extract(summary_page_func, content, goal, max_retries)

        # Phase 2: progressive content truncation retries
        if raw_dict is None:
            content_for_extract = content
            for i in range(3):
                content_for_extract = content_for_extract[:int(0.7 * len(content_for_extract))]
                print(f"[visit] Truncation retry {i + 1}/3 for url[{url}], content length: {len(content_for_extract)}")
                raw_dict = self._try_extract(summary_page_func, content_for_extract, goal, max_retries)
                if raw_dict is not None:
                    break

        if raw_dict is None:
            return self._build_error_output(url, goal)

        output = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
        if IS_EVIDENCE:
            output += "Evidence in page: \n" + str(raw_dict["evidence"]) + "\n\n"
        output += "Summary: \n" + str(raw_dict["summary"]) + "\n\n"
        return output
