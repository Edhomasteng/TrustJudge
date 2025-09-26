import re
import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Union, Optional
from OpenaiGenerator import OpenaiGenerator
from VllmGenerator import VllmGenerator
from common import OPENAI_MODELS, read_jsonl, write_jsonl, prompt_list_without_system

class SingleAnswerGrade:
    def __init__(self,
                 model: str,
                 prompt: str,
                 openai_api_key: str = None,
                 openai_api_base: str = None,
                 temperature: float = 1.0,
                 max_tokens: int = 1024,
                 top_logprobs: int = 1,
                 tensor_parallel_size: int = 1):
        self.model = model
        self.model_type = "openai" if model in OPENAI_MODELS else "vllm"
        self.prompt = prompt
        if self.model_type == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key is required.")
            self.generator = OpenaiGenerator(
                model=model,
                api_key=openai_api_key,
                api_base=openai_api_base,
                temperature=temperature,
                max_tokens=max_tokens,
                top_logprobs=top_logprobs
            )
        else:
            self.generator = VllmGenerator(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_logprobs=top_logprobs,
                tensor_parallel_size=tensor_parallel_size
            )


    def _build_grade_prompt(self, question: str, response: str) -> List[Dict[str, str]]:
        """Construct a two-message chat prompt (system and user) that embeds the question and the candidate response for scalar quality scoring.
        """
        return [
            {
                "role": "system",
                "content": self.prompt
            },
            {
                "role": "user",
                "content": f"Question: {question}\nResponse: {response}"
            }
        ]

    def judge(self, data: List[Dict], openai_batch_size: int = 4, save_steps: int = None, output_file: str = None) -> List[Dict]:
        """
        For each candidate response, build grading prompts, batch-evaluate with the chosen model.
        """
        if save_steps is not None and save_steps > 0 and not output_file:
            raise ValueError("When using save_steps, the output_file parameter must be provided.")
        if save_steps is None or save_steps <= 0:
            all_prompts = []
            mapping = []

            for q_idx, item in enumerate(data):
                question_text = item["question"]
                for a_idx, ans in enumerate(item["answers"]):
                    for cand_idx, candidate_text in enumerate(ans["text"]):
                        messages = self._build_grade_prompt(question_text, candidate_text)
                        if self.model_type == "openai":
                            prompt = messages
                        elif "gemma" in self.model.lower():
                            prompt = self.generator.tokenizer.apply_chat_template(
                                prompt_list_without_system(messages), 
                                tokenize=False, 
                                add_generation_prompt=True
                            )
                        else:
                            prompt = self.generator.tokenizer.apply_chat_template(
                                messages, 
                                tokenize=False, 
                                add_generation_prompt=True
                            )
                        all_prompts.append(prompt)
                        mapping.append((q_idx, a_idx, cand_idx))
            
            if self.model_type == "openai":
                generated_outputs = self.generator.batch_generate(all_prompts, openai_batch_size)
            else:
                generated_outputs = self.generator.batch_generate(all_prompts)

            output_data = []
            local_index_mapping = {}
            for q_idx, item in enumerate(data):
                new_item = {
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "candidate_answers": [],
                    "judgements": []
                }
                local_global_index = 0
                for a_idx, ans in enumerate(item["answers"]):
                    for cand_idx, candidate_text in enumerate(ans["text"]):
                        new_item["candidate_answers"].append({
                            "global_index": local_global_index,
                            "local_index": cand_idx,
                            "text": candidate_text
                        })
                        new_item["judgements"].append({
                            "candidate_global_index": local_global_index,
                            "judgement": None
                        })
                        local_index_mapping[(q_idx, a_idx, cand_idx)] = local_global_index
                        local_global_index += 1
                output_data.append(new_item)
            
            for prompt_idx, (q_idx, a_idx, cand_idx) in enumerate(mapping):
                judgement = generated_outputs[prompt_idx]
                gi = local_index_mapping[(q_idx, a_idx, cand_idx)]
                output_data[q_idx]["judgements"][gi]["judgement"] = judgement
            
            return output_data

        final_output = []
        for block_start in range(0, len(data), save_steps):
            block_questions = data[block_start: block_start + save_steps]
            all_prompts_block = []
            mapping_block = []
            output_data_block = []
            local_index_mapping_block = {}

            for local_q_idx, item in enumerate(block_questions):
                new_item = {
                    "question_id": item["question_id"],
                    "question": item["question"],
                    "candidate_answers": [],
                    "judgements": []
                }
                local_global_index = 0
                for a_idx, ans in enumerate(item["answers"]):
                    for cand_idx, candidate_text in enumerate(ans["text"]):
                        messages = self._build_grade_prompt(item["question"], candidate_text)
                        if self.model_type == "openai":
                            prompt = messages
                        elif "gemma" in self.model.lower():
                            prompt = self.generator.tokenizer.apply_chat_template(
                                prompt_list_without_system(messages), 
                                tokenize=False, 
                                add_generation_prompt=True
                            )
                        else:
                            prompt = self.generator.tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True
                            )
                        all_prompts_block.append(prompt)
                        mapping_block.append((local_q_idx, a_idx, cand_idx))
                        
                        new_item["candidate_answers"].append({
                            "global_index": local_global_index,
                            "local_index": cand_idx,
                            "text": candidate_text
                        })
                        new_item["judgements"].append({
                            "candidate_global_index": local_global_index,
                            "judgement": None
                        })
                        local_index_mapping_block[(local_q_idx, a_idx, cand_idx)] = local_global_index
                        local_global_index += 1
                output_data_block.append(new_item)

            if self.model_type == "openai":
                generated_outputs_block = self.generator.batch_generate(all_prompts_block, openai_batch_size)
            else:
                generated_outputs_block = self.generator.batch_generate(all_prompts_block)

            for prompt_idx, (local_q_idx, a_idx, cand_idx) in enumerate(mapping_block):
                judgement = generated_outputs_block[prompt_idx]
                gi = local_index_mapping_block[(local_q_idx, a_idx, cand_idx)]
                output_data_block[local_q_idx]["judgements"][gi]["judgement"] = judgement
            write_jsonl(output_data_block, output_file)
            final_output.extend(output_data_block)
        return final_output


def main():
    parser = argparse.ArgumentParser(description="Generate Singe-Score") 
    parser.add_argument( "--model", type=str, default="gpt-4o", help="Model name or local path (e.g., 'gpt-4o' or a checkpoint directory)." ) 
    parser.add_argument( "--openai-api-key", type=str, default="sk-xxxxxxxxxx", help="API key for the OpenAI-compatible endpoint." ) 
    parser.add_argument( "--openai-api-base", type=str, default="https://api.openai.com/v1/chat/completions", help="Base URL of the OpenAI-compatible Chat Completions endpoint." ) 
    parser.add_argument( "--openai-batch-size", type=int, default=5, help="Number of parallel requests when using the OpenAI API (ignored by vLLM/local backends)." ) 
    parser.add_argument( "--tensor-parallel-size", type=int, default=2, help="Number of GPUs to use for tensor parallelism (local model only)." ) 
    parser.add_argument( '--gpu-ids', type=str, default="2,3", help="Comma-separated CUDA device IDs to make visible, e.g., '0,1,2,3'." ) 
    parser.add_argument( "--temperature", type=float, default=1.0, help="Sampling temperature; higher values produce more random outputs." ) 
    parser.add_argument( "--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate for each sample." ) 
    parser.add_argument( "--top-logprobs", type=int, default=1, help="Return the top-N token logprobs per step (if supported by the backend)." ) 
    parser.add_argument( "--input-file", type=str, help="Path to the input JSONL file containing candidate answers." ) 
    parser.add_argument( "--output-file", type=str, help="Path to the output JSONL file where graded results will be saved." ) 
    parser.add_argument( "--save-steps", type=int, default=100, help="Checkpoint frequency: write partial results every N items." ) 
    parser.add_argument( "--prompt-file", type=str, help="Path to the JSONL file containing prompt templates." ) 
    parser.add_argument( "--prompt-label", type=str, default="single_answer_grade_5_points", help="Key/name of the prompt to use from the prompt file." ) 
    parser.add_argument( "--test-number", type=int, default=None, help="Optional: limit processing to the first N samples for quick testing." )
    args = parser.parse_args()
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    input_data = read_jsonl(args.input_file)[:args.test_number]
    prompt = read_jsonl(args.prompt_file)[0][args.prompt_label]
    grader = SingleAnswerGrade(
        model=args.model,
        prompt=prompt,
        openai_api_key=args.openai_api_key,
        openai_api_base=args.openai_api_base,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_logprobs=args.top_logprobs,
        tensor_parallel_size=args.tensor_parallel_size
    )
    graded_data = grader.judge(input_data, openai_batch_size=args.openai_batch_size, save_steps=args.save_steps, output_file=args.output_file)
    write_jsonl(graded_data, args.output_file)
    

if __name__ == "__main__":
    main()
