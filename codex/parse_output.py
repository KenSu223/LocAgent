"""Parse JSONL output from `codex exec --json`."""

import json
import re
from typing import List, Optional


def parse_codex_jsonl(output: str) -> Optional[str]:
    """Extract the final assistant message text from codex JSONL output.

    Each line is a JSON object with an "event" field. We look for
    "item.completed" events with type "agent_message" and return the
    last one.
    """
    last_message = None
    for line in output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("event") != "item.completed":
            continue

        item = event.get("item", {})
        # The item has a "type" field and content varies by type
        if item.get("type") == "agent_message":
            # agent_message content is in "content"
            content = item.get("content", "")
            if isinstance(content, list):
                # content can be a list of content blocks
                texts = []
                for block in content:
                    if isinstance(block, dict):
                        texts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        texts.append(block)
                content = "\n".join(texts)
            if content:
                last_message = content

    return last_message


def extract_files(text: str) -> List[str]:
    """Extract file paths from the agent's response.

    Tries JSON parsing first, then falls back to regex extraction.
    """
    if not text:
        return []

    # Try to find a JSON object with files_to_modify
    json_match = re.search(r'\{[^{}]*"files_to_modify"\s*:\s*\[([^\]]*)\][^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            files = obj.get("files_to_modify", [])
            if isinstance(files, list) and files:
                return [f.strip() for f in files if isinstance(f, str) and f.strip()]
        except json.JSONDecodeError:
            pass

    # Try parsing the entire text as JSON
    try:
        obj = json.loads(text.strip())
        files = obj.get("files_to_modify", [])
        if isinstance(files, list) and files:
            return [f.strip() for f in files if isinstance(f, str) and f.strip()]
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: extract file paths that look like source files
    # Match paths like src/foo/bar.py, lib/thing.js, etc.
    paths = re.findall(r'(?:^|\s|`|"|\')([a-zA-Z0-9_][a-zA-Z0-9_/\-]*\.[a-zA-Z]{1,4})(?:\s|`|"|\'|$|,)', text)
    # Filter to likely source files
    seen = set()
    result = []
    for p in paths:
        p = p.strip()
        if p not in seen and '/' in p:
            seen.add(p)
            result.append(p)
    return result[:5]


def parse_codex_output(raw_output: str) -> List[str]:
    """Full pipeline: parse JSONL, then extract files from the last message."""
    message = parse_codex_jsonl(raw_output)
    if message:
        return extract_files(message)

    # If JSONL parsing yielded nothing, try the raw output directly
    return extract_files(raw_output)
