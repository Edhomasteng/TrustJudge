import argparse
import random
import os, sys
from pathlib import Path
from vllm import LLM, SamplingParams
import math
import time
from common import prompt_list_without_system


class VllmGenerator:
    def __init__(self, model, temperature = 1.0, max_tokens=1024, n: int = 1, template_chat=None, top_logprobs: int = 1, tensor_parallel_size: int = 1):

        self.model = model
        self.llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True)
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, logprobs=top_logprobs, n=n)
        self.tokenizer = self.llm.get_tokenizer()
        self.template_chat = template_chat


    def batch_generate(self, prompt_list: list):
        """
        Generate completions for a list of prompts with vLLM.
        """
        responses = self.llm.generate(prompt_list, self.sampling_params)
        output = []
        for response in responses:
            random.seed(time.time())
            i = random.choice(range(len(response.outputs)))
            output.append({
                "prompt": response.prompt,
                "prompt_token_ids": response.prompt_token_ids,
                "encoder_prompt": response.encoder_prompt, 
                "encoder_prompt_token_ids": response.encoder_prompt_token_ids, 
                "prompt_logprobs": response.prompt_logprobs, 
                "cumulative_logprob": response.outputs[i].cumulative_logprob,
                "output_text": response.outputs[i].text,
                "output_token_ids": response.outputs[i].token_ids,
                "original_output_logprobs": [{
                    token_id: {
                        "logprob":logprob.logprob,
                        "rank": logprob.rank,
                        "decoded_token": logprob.decoded_token
                    } 
                    for token_id, logprob in logprob_dict.items()} for logprob_dict in response.outputs[i].logprobs],
                "output_logprobs": [{logprob.decoded_token: logprob.logprob for token_id, logprob in logprob_dict.items()} for logprob_dict in response.outputs[i].logprobs],
                "output_probs": self._convert_logprobs_to_probs(response.outputs[i].logprobs)
            })
        return output


    def _convert_logprobs_to_probs(self, logprobs):
        """
        Transform vLLM per-token logprob dictionaries into probability dictionaries
        """
        return [{logprob.decoded_token: math.exp(logprob.logprob) for token_id, logprob in logprob_dict.items()} for logprob_dict in logprobs]
    
