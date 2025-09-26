import argparse
import os
from typing import List, Dict, Optional
from OpenaiGenerator import OpenaiGenerator
from VllmGenerator import VllmGenerator
from common import OPENAI_MODELS, read_jsonl, write_jsonl, prompt_list_without_system


class PairwiseComparison:
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


    def _build_pairwise_prompt(self, question: str, answer_a: str, answer_b: str) -> List[Dict[str, str]]:
        """
        Construct a two-message chat prompt (system + user) that embeds the question and two candidate answers in a fixed template for order-aware pairwise judging.
        """
        system_msg = {
            "role": "system",
            "content": self.prompt
        }
        user_msg = {
            "role": "user",
            "content": (
                f"[User Question]\n{question}\n\n"
                f"[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
                f"[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
            )
        }
        return [system_msg, user_msg]
    

    def judge_pairwise(self, data: List[Dict], openai_batch_size: int = 4, save_steps: int = None, output_file: str = None) -> List[Dict]:
        """
        For each question, enumerate all unordered answer pairs, generate both presentation orders, batch the model calls, attach the two judgments to each pair, and stream results to disk in blocks while preserving input order.
        """
        if save_steps is not None and save_steps > 0 and not output_file:
            raise ValueError("When using save_steps, the output_file parameter must be provided.")
        final_output = []  
        current_block_data = []  
        current_block_prompts = []  
        current_block_mapping = []
        for global_q_idx, item in enumerate(data):
            question_text = item["question"]
            candidate_answers = []
            for ans in item["answers"]:
                texts = ans.get("text", [])
                for local_idx, txt in enumerate(texts):
                    candidate_answers.append({
                        "local_index": local_idx,
                        "text": txt
                    })
            for i, cand in enumerate(candidate_answers):
                candidate_answers[i] = dict(global_index=i, **cand)
            current_q_idx = len(current_block_data)
            comparisons = []
            for i in range(len(candidate_answers)):
                for j in range(i+1, len(candidate_answers)):
                    if not candidate_answers[i]["text"] or not candidate_answers[j]["text"]:
                        continue
                    messages1 = self._build_pairwise_prompt(question_text, candidate_answers[i]["text"], candidate_answers[j]["text"])
                    if self.model_type == "openai":
                        prompt1 = messages1
                    elif "gemma" in self.model.lower():
                        prompt1 = self.generator.tokenizer.apply_chat_template(
                            prompt_list_without_system(messages1),
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        prompt1 = self.generator.tokenizer.apply_chat_template(
                            messages1,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    current_block_prompts.append(prompt1)
                    current_block_mapping.append((current_q_idx, candidate_answers[i]["global_index"], candidate_answers[j]["global_index"], 1))
                    
                    messages2 = self._build_pairwise_prompt(question_text, candidate_answers[j]["text"], candidate_answers[i]["text"])
                    if self.model_type == "openai":
                        prompt2 = messages2
                    elif "gemma" in self.model.lower():
                        prompt2 = self.generator.tokenizer.apply_chat_template(
                            prompt_list_without_system(messages2),
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    else:
                        prompt2 = self.generator.tokenizer.apply_chat_template(
                            messages2,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                    current_block_prompts.append(prompt2)
                    current_block_mapping.append((current_q_idx, candidate_answers[i]["global_index"], candidate_answers[j]["global_index"], 2))
                    
                    comparisons.append({
                        "candidate_global_index1": candidate_answers[i]["global_index"],
                        "candidate_global_index2": candidate_answers[j]["global_index"],
                        "judgement_order1": None,
                        "judgement_order2": None
                    })
            
            question_output = {
                "question_id": item["question_id"],
                "question": question_text,
                "candidate_answers": candidate_answers,
                "judgements": comparisons
            }
            current_block_data.append(question_output)
            
            if save_steps and len(current_block_data) >= save_steps:
                if self.model_type == "openai":
                    generated_outputs = self.generator.batch_generate(current_block_prompts, openai_batch_size)
                else:
                    generated_outputs = self.generator.batch_generate(current_block_prompts)
                
                for idx, (q_idx_in_block, cand_i, cand_j, order) in enumerate(current_block_mapping):
                    comparison_result = generated_outputs[idx]
                    for comp in current_block_data[q_idx_in_block]["judgements"]:
                        if comp["candidate_global_index1"] == cand_i and comp["candidate_global_index2"] == cand_j:
                            if order == 1:
                                comp["judgement_order1"] = comparison_result
                            else:
                                comp["judgement_order2"] = comparison_result
                            break
                
                write_jsonl(current_block_data, output_file)
                final_output.extend(current_block_data)
                current_block_data = []
                current_block_prompts = []
                current_block_mapping = []  
        if current_block_data:
            if current_block_prompts:
                if self.model_type == "openai":
                    generated_outputs = self.generator.batch_generate(current_block_prompts, openai_batch_size)
                else:
                    generated_outputs = self.generator.batch_generate(current_block_prompts)
                for idx, (q_idx_in_block, cand_i, cand_j, order) in enumerate(current_block_mapping):
                    comparison_result = generated_outputs[idx]
                    for comp in current_block_data[q_idx_in_block]["judgements"]:
                        if comp["candidate_global_index1"] == cand_i and comp["candidate_global_index2"] == cand_j:
                            if order == 1:
                                comp["judgement_order1"] = comparison_result
                            else:
                                comp["judgement_order2"] = comparison_result
                            break
            write_jsonl(current_block_data, output_file)
            final_output.extend(current_block_data)
        return final_output

def main():
    parser = argparse.ArgumentParser(description="Generate Pairwise Comparison.") 
    parser.add_argument( "--model", type=str, default="gpt-4o", help="Model name or local path (e.g., 'gpt-4o' or a checkpoint directory)." ) 
    parser.add_argument( "--openai-api-key", type=str, default="sk-xxxxxxxxx", help="API key for the OpenAI-compatible endpoint." ) 
    parser.add_argument( "--openai-api-base", type=str, default="https://api.openai.com/v1/chat/completions", help="Base URL of the OpenAI-compatible Chat Completions endpoint." ) 
    parser.add_argument( "--openai-batch-size", type=int, default=5, help="Number of parallel requests when using the OpenAI API (ignored by vLLM/local backends)." ) 
    parser.add_argument( "--tensor-parallel-size", type=int, default=2, help="Number of GPUs to use for tensor parallelism (local model only)." ) 
    parser.add_argument( "--gpu-ids", type=str, default="2,3", help="Comma-separated CUDA device IDs to make visible, e.g., '0,1,2,3'." ) 
    parser.add_argument( "--temperature", type=float, default=1.0, help="Sampling temperature; higher values produce more random outputs." ) 
    parser.add_argument( "--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate for each sample." ) 
    parser.add_argument( "--top-logprobs", type=int, default=1, help="Return the top-N token logprobs per step (if supported by the backend)." ) 
    parser.add_argument( "--input-file", type=str, help="Path to the input JSONL file containing candidate answers." ) 
    parser.add_argument( "--output-file", type=str, help="Path to the output JSONL file where graded results will be saved." ) 
    parser.add_argument( "--save-steps", type=int, default=100000, help="Checkpoint frequency: write partial results every N items." ) 
    parser.add_argument( "--prompt-file", type=str, help="Path to the JSONL file containing prompt templates." ) 
    parser.add_argument( "--prompt-label", type=str, default="pairwise_comparison", help="Key/name of the prompt to use from the prompt file." ) 
    parser.add_argument( "--test-number", type=int, default=None, help="Optional: limit processing to the first N samples for quick testing." ) 
    args = parser.parse_args()
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    input_data = read_jsonl(args.input_file)[:args.test_number]
    prompt = read_jsonl(args.prompt_file)[0][args.prompt_label]
    grader = PairwiseComparison(
        model=args.model,
        prompt=prompt,
        openai_api_key=args.openai_api_key,
        openai_api_base=args.openai_api_base,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_logprobs=args.top_logprobs,
        tensor_parallel_size=args.tensor_parallel_size
    )
    graded_data = grader.judge_pairwise(input_data, openai_batch_size=args.openai_batch_size, save_steps=args.save_steps, output_file=args.output_file)
    write_jsonl(graded_data, args.output_file)


if __name__ == "__main__":
    main()
