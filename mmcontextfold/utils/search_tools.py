"""
search_tools.py

Unified search tool interface for MM-ContextFold pipeline.
Wraps text search, reverse image search (Google Lens), and web page reading.
"""

import os
import json
import requests
import time
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Optional

from configs.api_keys import (
    SERPER_API_KEY,
    JINA_API_KEY,
)


def format_tool_result(tool_name: str, query: str, goal: str, success: bool, count: int = 0, results: str = "") -> str:
    status = "Success" if success else "Failed"

    if not success:
        return f"""## {tool_name} Results

**Query:** {query}
**Goal:** {goal}
**Status:** {status}

No results found. Try to fix the current query or use a different tool."""

    return f"""## {tool_name} Results

**Query:** {query}
**Goal:** {goal}
**Status:** {status}
**Count:** {count} results

### Results:
{results}"""


def t2t_search_serper(query, round_id):
    query_text = query[0].get("query", "")
    goal = query[0].get("goal", "")
    search_data = []

    def _fail_result():
        search_data.append({
            "sub_query_id": f"r{round_id}_t2t",
            "function": "text_search",
            "query": query_text,
            "goal": goal,
            "search_results": [],
        })
        fail_msg = format_tool_result("text_search", query_text, goal, success=False)
        return search_data, fail_msg

    max_retries = 5
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query_text, "location": "United States"})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    for i in range(max_retries):
        try:
            response = requests.request("POST", url, headers=headers, data=payload, timeout=60)
            response.raise_for_status()
            break
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            print(f"Text search attempt {i+1}/{max_retries} failed: {e}")
            if i == max_retries - 1:
                return _fail_result()
            time.sleep(3)
            continue

    try:
        results = response.json()
    except json.JSONDecodeError as e:
        print(f"JSON parse failed: {e}")
        return _fail_result()

    if "organic" not in results or not results["organic"]:
        return _fail_result()

    web_snippets = []
    for idx, page in enumerate(results["organic"]):
        snippet = page.get('snippet', '')
        snippet_text = f"\n  {snippet}" if snippet else ""
        line = f"{idx}. [{page['title']}]({page['link']}){snippet_text}"
        web_snippets.append(line)

    results_text = "\n\n".join(web_snippets)
    search_results = format_tool_result("text_search", query_text, goal, success=True, count=len(web_snippets), results=results_text)

    search_data.append({
        "sub_query_id": f"r{round_id}_t2t",
        "function": "text_search",
        "query": query_text,
        "goal": goal,
        "search_results": results["organic"],
    })

    return search_data, search_results


def i2i_search_serper(query, round_id, source_type, img_idx, img_global_id, save_dir):
    img_path = query[0].get("img_path", "")
    goal = query[0].get("goal", "")

    search_data = []
    local_paths = []

    def _fail_result():
        search_data.append({
            "sub_query_id": f"r{round_id}_i2i",
            "function": "lens_search",
            "query": img_path,
            "goal": goal,
            "search_results": [],
        })
        fail_msg = format_tool_result("lens_search", f"<image#{img_idx}>", goal, success=False)
        return search_data, fail_msg, local_paths

    img_name = str(Path(img_path).name)

    # Dataset-specific image URL mapping
    dataset_url_map = {
        "livevqa": f'https://your-storage-endpoint/livevqa/imgs/{img_name}',
        "mmsearch": f'https://your-storage-endpoint/mmsearch/imgs/{img_name}',
        "mmsearch_plus": f'https://your-storage-endpoint/mmsearch_plus/{img_name}',
        "simplevqa": f'https://your-storage-endpoint/simplevqa/imgs/{img_name}',
        "vdr_bench": f'https://your-storage-endpoint/vdr_bench/imgs/{img_name}',
        "world_vqa": f'https://your-storage-endpoint/world_vqa/imgs/{img_name}',
        "browsecomp_vl": f'https://your-storage-endpoint/browsecomp_vl/imgs/{img_name}',
    }

    query_img_url = dataset_url_map.get(source_type)
    if source_type == "new_load":
        query_img_url = img_path
    if not query_img_url:
        return _fail_result()

    url = "https://google.serper.dev/lens"
    payload = json.dumps({"url": query_img_url, "num": 100, "location": "United States"})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.request("POST", url, headers=headers, data=payload, timeout=100)
            response.raise_for_status()
            break
        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
            print(f"Lens search attempt {i+1}/{max_retries} failed: {e}")
            if i == max_retries - 1:
                return _fail_result()
            continue

    try:
        results = response.json()
    except json.JSONDecodeError:
        return _fail_result()

    if "organic" not in results or not results["organic"]:
        return _fail_result()

    img_metadata = results.get("organic", [])
    local_img_data, local_paths = img_download(img_metadata, round_id, img_global_id, save_dir, max_imgs=10)

    search_data.append({
        "sub_query_id": f"r{round_id}_i2i",
        "function": "lens_search",
        "query": img_path,
        "goal": goal,
        "search_results": local_img_data,
    })

    img_snippets = []
    for idx, img_data in enumerate(local_img_data):
        title = img_data.get('title', 'No title')
        link = img_data.get('link', '')
        snippet = f"- <image#{img_data['img_global_id']}> [{title}], ORIGINAL PAGE SOURCE({link})"
        img_snippets.append(snippet)

    results_text = "\n".join(img_snippets)
    search_results = format_tool_result("lens_search", f"<image#{img_idx}>", goal, success=True, count=len(img_snippets), results=results_text)

    return search_data, search_results, local_paths


def img_download(img_metadata, round_id, img_global_id, save_dir="downloads", max_imgs=10):
    img_data = []
    local_img_paths = []
    downloaded_count = 0
    range_max = max_imgs * 3
    img_metadata = img_metadata[:range_max]

    for idx, data in enumerate(img_metadata):
        if downloaded_count >= max_imgs:
            break

        img_thumbnailUrl = data.get("thumbnailUrl")
        if not img_thumbnailUrl:
            continue
        try:
            response = requests.get(img_thumbnailUrl, timeout=10)
            response.raise_for_status()
            img_bytes = BytesIO(response.content)
            img = Image.open(img_bytes)
            img_path = os.path.join(save_dir, f"round_{round_id}_img_{idx}_global_{img_global_id}.png")
            img.save(img_path, format='PNG')
            local_img_paths.append(img_path)

            meta_data = {
                "img_id": f"round_{round_id}_img_{idx}_global_{img_global_id}",
                "img_global_id": img_global_id,
                "img_url": data.get("imageUrl"),
                "img_path": img_path,
                "title": data.get("title"),
                "thumbnailUrl": data.get("thumbnailUrl"),
                "source": data.get("source"),
                "link": data.get("link"),
            }

            img_data.append(meta_data)
            img_global_id += 1
            downloaded_count += 1

        except Exception as e:
            continue

    return img_data, local_img_paths


def call_visit_jina(url, goal):
    max_retries = 3
    timeout = 50

    for attempt in range(max_retries):
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
            "X-Md-Link-Style": "discarded",
            "X-Retain-imgs": "none",
            "X-Return-Format": "markdown"
        }

        try:
            response = requests.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                timeout=timeout
            )
            if response.status_code == 200:
                webpage_content = response.text.strip()
                if not webpage_content:
                    return None

                blocked_indicators = [
                    "Verify you are human", "Cloudflare", "Just a moment...",
                    "Ray ID:", "403 Forbidden", "Access Denied",
                    "Target URL returned error"
                ]
                if any(indicator in webpage_content for indicator in blocked_indicators):
                    return None

                return webpage_content
            else:
                raise ValueError(f"jina readpage error: {response.status_code}")
        except Exception as e:
            time.sleep(0.5)
            if attempt == max_retries - 1:
                return None

    return None


def call_visit_serper(url, goal):
    max_retries = 10
    timeout = 50
    api_url = "https://scrape.serper.dev"

    for attempt in range(max_retries):
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        payload = json.dumps({"url": url, "includeMarkdown": True})

        try:
            response = requests.post(api_url, headers=headers, data=payload, timeout=timeout)

            if response.status_code == 200:
                webpage_content = response.text.strip()
                if not webpage_content:
                    return None

                blocked_indicators = [
                    "Verify you are human", "Cloudflare", "Just a moment...",
                    "Ray ID:", "403 Forbidden", "Access Denied"
                ]
                if any(indicator in webpage_content for indicator in blocked_indicators):
                    return None

                return webpage_content
            else:
                raise ValueError("serper scrape error")

        except Exception as e:
            time.sleep(0.5)
            if attempt == max_retries - 1:
                return None

    return None


class SearchTools:
    """
    Unified search interface for MM-ContextFold pipeline.

    - text_search(query) -> str
    - visit_page(url) -> str
    - lens_search(image_path) -> str
    - lens_search_with_images(image_path) -> (items, paths)
    """

    def __init__(self, output_dir: str, dataset_name: str = ""):
        self.output_dir = output_dir
        self.img_save_dir = os.path.join(output_dir, "imgs")
        self.dataset_name = dataset_name
        self._round = 0
        self._img_global_id = 0
        os.makedirs(self.img_save_dir, exist_ok=True)

    def text_search(self, query: str) -> str:
        self._round += 1
        _search_data, formatted_text = t2t_search_serper(
            [{"query": query, "goal": ""}], self._round
        )
        return formatted_text

    def visit_page(self, url: str) -> str:
        content = call_visit_jina(url, "")
        if content is None:
            content = call_visit_serper(url, "")
        return content or "Failed to read page content."

    def lens_search(self, image_path: str) -> str:
        self._round += 1
        _search_data, formatted_text, _local_paths = i2i_search_serper(
            [{"img_path": image_path, "goal": ""}],
            self._round,
            source_type=self.dataset_name,
            img_idx=0,
            img_global_id=self._img_global_id,
            save_dir=self.img_save_dir,
        )
        self._img_global_id += len(_local_paths)
        return formatted_text

    def lens_search_with_images(self, image_path: str) -> tuple:
        self._round += 1
        search_data, _formatted_text, local_paths = i2i_search_serper(
            [{"img_path": image_path, "goal": ""}],
            self._round,
            source_type=self.dataset_name,
            img_idx=0,
            img_global_id=self._img_global_id,
            save_dir=self.img_save_dir,
        )
        self._img_global_id += len(local_paths)
        items = []
        if search_data and search_data[0].get("search_results"):
            for r in search_data[0]["search_results"]:
                items.append({
                    "img_global_id": r.get("img_global_id", 0),
                    "title": r.get("title", ""),
                    "link": r.get("link", ""),
                })
        return items, local_paths
