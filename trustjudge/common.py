from collections import defaultdict
import math
import os
import json
import glob
import re
from typing import Optional, Union, List, Dict, Iterable, Any
from vllm import LLM


OPENAI_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-instruct",
    "gpt-4",
    "gpt-4o",
    "gpt-4-all",
    "gpt-4-32k",
    "gpt-4o-all",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4o-mini",
    "gpt-4-32k-0314",
    "gpt-4-32k-0613",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-preview",
    "gpt-4-turbo-browsing",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-turbo-2024-04-09",
    "claude-3-5-haiku-20241022",
    "gemini-1.5-flash",
    "glm-4-air",
    "deepseek-ai/DeepSeek-R1",
    "DeepSeek-R1"
    ]

API_MAX_RETRY = 10
API_RETRY_SLEEP = 2


def find_jsonl_files(path: str) -> list:
    """
    Retrieve all JSONL files from the given path.
    """
    if path.endswith(".jsonl"):
        return [path]
    else:
        return glob.glob(os.path.join(path, "*.jsonl"))


def read_jsonl(file_path: str) -> Union[List[Dict], List]:
    """
    Read data from JSONL files.
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            first_char = file.read(1)
            file.seek(0)
            if first_char == '[':
                data = json.load(file)
            else:
                for line in file:
                    data.append(json.loads(line.strip()))
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from file {file_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file {file_path}: {e}")
    return data


def write_jsonl(data: list, file_path: str, append: bool = False):
    """
    Write data to a JSONL file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    mode = 'a' if append else 'w'

    with open(file_path, mode, encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')


def extract_middle_dict(logprobs: List[Dict], allowed_first_keys: Iterable[Any] = None) -> Dict:
    """
    From a sequence of token logprob dictionaries, find the dictionary between matching "[" and "]"
    """
    def clear_key(key: Any) -> Any:
        if isinstance(key, str):
            return key.replace("\u0120", "").replace("Ġ", "").replace("\u2581", "").strip()
        return key

    original_logprobs: List[Dict[Any, float]] = logprobs 

    processed_logprobs: List[Dict[Any, float]] = []
    for entry in original_logprobs:
        acc: Dict[Any, float] = defaultdict(float)
        for k, v in entry.items():
            ck = clear_key(k)
            acc[ck] += v
        processed_logprobs.append(dict(acc))

    logprobs = processed_logprobs

    for left_index, d in enumerate(logprobs):
        if not d:
            continue
        if next(iter(d)) == "[":
            search_limit = min(left_index + 6, len(logprobs))
            for right_index in range(left_index + 1, search_limit):
                if any(isinstance(key, str) and key.strip() == ']' for key in logprobs[right_index].keys()):
                    in_between = logprobs[left_index + 1:right_index]
                    if len(in_between) != 1:
                        continue
                    middle_dict = in_between[0]
                    if allowed_first_keys is not None:
                        if not middle_dict:
                            continue
                        orig_first_key = next(iter(middle_dict))
                        first_key_stripped = orig_first_key.strip() if isinstance(orig_first_key, str) else orig_first_key
                        if first_key_stripped in allowed_first_keys:
                            return middle_dict
                        else:
                            try:
                                key_numeric = float(first_key_stripped)
                            except (ValueError, TypeError):
                                key_numeric = None
                            if key_numeric is not None and key_numeric in allowed_first_keys:
                                return middle_dict
                            else:
                                continue
                    else:
                        return middle_dict
    raise ValueError("No valid square bracket pairs were found or the middle dictionary does not meet the requirements.")


def clean_probs_dict(probabilities: Dict[str, float]) -> Dict[str, float]:
    """
    Trim whitespace from keys and merge duplicate keys by summing their probability values.
    """
    cleaned = {}
    for key, value in probabilities.items():
        trimmed_key = key.strip()
        cleaned[trimmed_key] = cleaned.get(trimmed_key, 0) + value
    return cleaned


def calculate_perplexity(cumulative_logprob: float, token_count: int) -> float:
    """
    Compute perplexity from a cumulative log-probability and token count.
    """
    if token_count == 0:
        raise ValueError("Token count must be greater than zero.")
    avg_logprob = cumulative_logprob / token_count
    try:
        return math.exp(-avg_logprob)
    except OverflowError:
        return float('inf')


def prompt_list_without_system(messages):
    """
    Validate there is exactly one system and one user message, then combine their contents into a single user message for models that don’t support a separate system role.
    """
    system_content = None
    user_content = None

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            system_content = msg.get("content", "")
        elif role == "user":
            user_content = msg.get("content", "")

    if system_content is None or user_content is None:
        raise ValueError("Input must contain one 'system' and one 'user' message.")

    combined_content = f"{system_content}\n\n{user_content}"
    return [{"role": "user", "content": combined_content}]