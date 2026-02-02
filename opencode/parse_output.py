"""Parse output from `opencode run --format json`."""

import json
import re
from typing import List, Optional


def parse_opencode_json(output: str) -> Optional[str]:
    """Extract the final assistant message text from opencode JSON output.

    When using --format json, opencode emits JSON event lines.
    We look for text content in assistant messages.
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

        # opencode json format: look for assistant text events
        etype = event.get("type", "")

        # opencode events have a "part" sub-object with the actual content
        part = event.get("part", {})

        if etype in ("text", "text-delta"):
            # Text can be in part.text or at the top level
            text = part.get("text", "") or event.get("text", "")
            if text:
                if last_message is None:
                    last_message = text
                else:
                    last_message += text

    return last_message


def extract_files(text: str) -> List[str]:
    """Extract file paths from the agent's response.

    Tries JSON parsing first, then falls back to regex extraction.
    """
    if not text:
        return []

    # Try to find a JSON object with files_to_modify
    json_match = re.search(
        r'\{[^{}]*"files_to_modify"\s*:\s*\[([^\]]*)\][^{}]*\}',
        text, re.DOTALL,
    )
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
    paths = re.findall(
        r'(?:^|\s|`|"|\')([a-zA-Z0-9_][a-zA-Z0-9_/\-]*\.[a-zA-Z]{1,4})(?:\s|`|"|\'|$|,)',
        text,
    )
    seen = set()
    result = []
    for p in paths:
        p = p.strip()
        if p not in seen and '/' in p:
            seen.add(p)
            result.append(p)
    return result[:5]


def parse_opencode_output(raw_output: str) -> List[str]:
    """Full pipeline: parse JSON output, then extract files from the last message.

    Falls back to treating the entire output as text if JSON parsing fails.
    """
    message = parse_opencode_json(raw_output)
    if message:
        files = extract_files(message)
        if files:
            return files

    # If JSON parsing yielded nothing useful, try the raw output directly
    return extract_files(raw_output)
