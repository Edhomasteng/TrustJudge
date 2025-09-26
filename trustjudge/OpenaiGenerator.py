import argparse
import json
import math
import requests
import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from common import API_MAX_RETRY, API_RETRY_SLEEP


class OpenaiGenerator:
    def __init__(self, model: str, api_key: str, api_base: str = None, temperature: float = 1.0, 
                 max_tokens: int = 2048, n: int = 1, top_logprobs: int = 1):
        if not api_key:
            raise ValueError("OpenAI API key is required.")

        self.temperature = 10e-7 if temperature == 0 else temperature
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.n = n
        self.top_logprobs = top_logprobs


    def generate(self, prompt: list) -> tuple:
        """
        Submit a single chat-completions request with logprobs, retrying on transient errors; return a standardized record with text, per-token logprobs/probs, and cumulative logprob.
        """
        for _ in range(API_MAX_RETRY):
            try:
                messages = prompt

                headers = {
                    'Accept': 'application/json',
                    'Authorization': f'Bearer {self.api_key}', 
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }

                payload = json.dumps({
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "n": self.n,
                    "logprobs": True,
                    "top_logprobs": self.top_logprobs
                })

                response = requests.post(self.api_base, headers=headers, data=payload, timeout=10)
                response = response.json()

                probs_pairs, logprobs_pair = self._calculate_probs_and_logprobs(
                    response["choices"][0]["logprobs"]["content"]
                )

                output = {
                    "prompt": messages,
                    "prompt_token_ids": None,
                    "encoder_prompt": None, 
                    "encoder_prompt_token_ids": None, 
                    "prompt_logprobs": None, 
                    "cumulative_logprob": self._calculate_cumulative_logprob(response),
                    "output_text": response["choices"][0]["message"]["content"],
                    "output_token_ids": None,
                    "original_output_logprobs": response["choices"][0]["logprobs"]["content"],
                    "output_logprobs": logprobs_pair,
                    "output_probs": probs_pairs
                }

                if 'logprobs' in response["choices"][0]:
                    return output
                else:
                    print("Logprobs not found in response, retrying...")
                    time.sleep(API_RETRY_SLEEP)
                    continue
                
            except requests.exceptions.Timeout as e:
                print(f"Timeout occurred: {e}, retrying...")
                time.sleep(API_RETRY_SLEEP)
            except openai.OpenAIError as e:
                print(f"OpenAIError: {e}")
                time.sleep(API_RETRY_SLEEP)
            except requests.exceptions.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
                time.sleep(API_RETRY_SLEEP)
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                time.sleep(API_RETRY_SLEEP)
            except KeyError as e:
                print(response)
                print(f"KeyError: {e}, retrying...") 
                time.sleep(API_RETRY_SLEEP)
            except TypeError as e:
                print(f"TypeError: {e}, retrying...")
                time.sleep(API_RETRY_SLEEP)
        raise RuntimeError("Max retries exceeded, request failed.")


    def batch_generate(self, prompt_list: list, openai_batch_size: int = 4) -> list:
        """
        Run generate concurrently over a list of prompts using a thread pool.
        """
        results = []
        with ThreadPoolExecutor(max_workers=openai_batch_size) as executor:
            future_to_index = {executor.submit(self.generate, prompt): i for i, prompt in enumerate(prompt_list)}
            for future in tqdm(as_completed(future_to_index), total=len(prompt_list)):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results.append((index, result))
                except Exception as e:
                    print(f"Request failed for input {index}: {e}")
                    results.append((index, None))
        results.sort(key=lambda x: x[0])
        return [result[1] for result in results]


    def _calculate_probs_and_logprobs(self, logprobs):
        """
        Convert OpenAI per-token logprobs into parallel lists of {token→prob} and {token→logprob} maps for each output position.
        """
        probs_pairs = [
            {item['token']: math.exp(item['logprob']) for item in dict(entry)['top_logprobs']} 
            for entry in logprobs
        ]
        logprobs_pair = [
            {item['token']: item['logprob'] for item in dict(entry)['top_logprobs']} 
            for entry in logprobs
        ]
        return probs_pairs, logprobs_pair


    def _calculate_cumulative_logprob(self, response):
        """
        Sum per-token logprobs across the completion to produce a scalar cumulative log-likelihood.
        """
        content_entries = response["choices"][0]["logprobs"]["content"]
        cumulative_logprob = 0.0
        for entry in content_entries:
            cumulative_logprob += entry["logprob"]
        return cumulative_logprob
    

    def _get_token_rank(self, single_logprob):
        """
        Determine the rank of the emitted token within the provided top_logprobs list for that position.
        """
        current_token = single_logprob["token"]
        current_logprob = single_logprob["logprob"]
        top_logprobs = single_logprob["top_logprobs"]
            
        rank = None
        for idx, candidate in enumerate(top_logprobs, start=1):
            if candidate["token"] == current_token and candidate["logprob"] == current_logprob:
                rank = idx
                break
        return rank
    