import os
import json
from pathlib import Path
import re
import itertools
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Any, Optional, Set
from common import clean_probs_dict, extract_middle_dict
from common import read_jsonl, write_jsonl
from common import calculate_perplexity
from tqdm import tqdm

def evaluate_triple(op1: str, op2: str, op3: str) -> dict:
    """
    Validate a triple of pairwise relations (>, =, <) for transitivity.
    """
    operator_map = {"larger": ">", "equal": "=", "smaller": "<"}
    try:
        key = (operator_map[op1], operator_map[op2], operator_map[op3])
    except KeyError as e:
        raise ValueError(
            f"Invalid operator: {e.args[0]}. Use only 'larger', 'equal', or 'smaller'."
        )
    
    valid_combinations = {
        ('>', '>', '>'): {"result": True, "type": None},
        ('>', '>', '='): {"result": False, "type": "equal"},
        ('>', '>', '<'): {"result": False, "type": "circle"},
        ('>', '=', '>'): {"result": True, "type": None},
        ('>', '=', '='): {"result": False, "type": "equal"},
        ('>', '=', '<'): {"result": False, "type": "circle"},
        ('>', '<', '>'): {"result": True, "type": None},
        ('>', '<', '='): {"result": True, "type": None},
        ('>', '<', '<'): {"result": True, "type": None},
        ('=', '>', '>'): {"result": True, "type": None},
        ('=', '>', '='): {"result": False, "type": "equal"},
        ('=', '>', '<'): {"result": False, "type": "circle"},
        ('=', '=', '>'): {"result": False, "type": "equal"},
        ('=', '=', '='): {"result": True, "type": None},
        ('=', '=', '<'): {"result": False, "type": "equal"},
        ('=', '<', '>'): {"result": False, "type": "circle"},
        ('=', '<', '='): {"result": False, "type": "equal"},
        ('=', '<', '<'): {"result": True, "type": None},
        ('<', '>', '>'): {"result": True, "type": None},
        ('<', '>', '='): {"result": True, "type": None},
        ('<', '>', '<'): {"result": True, "type": None},
        ('<', '=', '>'): {"result": False, "type": "circle"},
        ('<', '=', '='): {"result": False, "type": "equal"},
        ('<', '=', '<'): {"result": True, "type": None},
        ('<', '<', '>'): {"result": False, "type": "circle"},
        ('<', '<', '='): {"result": False, "type": "equal"},
        ('<', '<', '<'): {"result": True, "type": None}
    }
    
    return valid_combinations.get(key, None)


class NonTransitivityCalculator:
    def __init__(self, question_with_category: Dict[int, str], pairwise_comparison_file: str, result_file: str, length: int, tolerance_gap: float, workers: int, model: str, method: str):
        self.question_with_category = question_with_category
        self.pairwise_comparison_file = pairwise_comparison_file
        self.result_file = result_file
        self.length = length
        self.tolerance_gap = tolerance_gap
        self.workers = workers
        self.model = model
        self.method = method
        self.pairwise_comparison_data = self.set_pairwise_comparison_data()
        self.logged_pairs: Set[Tuple[Any,int,int]] = set()
        self.logged_pairs: Set[Tuple[Any, int, int]] = set()
        log_path = os.path.join(os.path.dirname(self.result_file), f"non_transitivity_{self.method}.jsonl")
        if os.path.exists(log_path):
            try:
                existing = read_jsonl(log_path)
                for rec in existing:
                    m = rec.get("model")                
                    qid = rec.get("question_id")
                    i1 = rec.get("idx1")
                    i2 = rec.get("idx2")
                    if m is not None and qid is not None and i1 is not None and i2 is not None:
                        self.logged_pairs.add((m, qid, i1, i2))
            except Exception:
                pass


    def set_pairwise_comparison_data(self) -> dict:
        """
        Load or build a cache mapping (question_id, idx1, idx2) to bidirectional pairwise judgements plus candidate list.
        """
        cache_dir = Path("./data/judgements_visible/pairwise_comparison")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / os.path.basename(self.pairwise_comparison_file)
        if cache_file.exists():
            result = {}
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    question_id = record["question_id"]
                    candidate_global_index1 = record["candidate_global_index1"]
                    candidate_global_index2 = record["candidate_global_index2"]
                    judgement_order1 = record["judgement_order1"]
                    judgement_order2 = record["judgement_order2"]
                    candidate_list = record["candidate_list"]
                    result[(question_id, candidate_global_index1, candidate_global_index2)] = {
                        "judgement_order1": judgement_order1,
                        "judgement_order2": judgement_order2,
                        "candidate_list": candidate_list
                    }
            return result
        data = read_jsonl(self.pairwise_comparison_file)
        result = {}
        for item in tqdm(data, desc="reading pairwise comparison data"):
            question_id = item.get("question_id")
            for judgement in item.get("judgements", []):
                candidate_global_index1 = judgement.get("candidate_global_index1")
                candidate_global_index2 = judgement.get("candidate_global_index2")
                candidate_list = [candidate["global_index"] for candidate in item["candidate_answers"]]
                candidate_list.sort()
                allowed_numbers = set(list("ABC"))
                try:
                    letter1 = extract_middle_dict(judgement["judgement_order1"].get("output_probs"), allowed_first_keys=allowed_numbers)
                    letter2 = extract_middle_dict(judgement["judgement_order2"].get("output_probs"), allowed_first_keys=allowed_numbers)
                except Exception as e:
                    continue
                result[(question_id, candidate_global_index1, candidate_global_index2)] = {
                    "judgement_order1": {
                        "output_probs": letter1, 
                        "cumulative_logprob": judgement["judgement_order1"].get("cumulative_logprob"),
                        "original_output_logprobs_length": len(judgement["judgement_order1"]["original_output_logprobs"])
                        },
                    "judgement_order2": {
                        "output_probs": letter2, 
                        "cumulative_logprob": judgement["judgement_order2"].get("cumulative_logprob"),
                        "original_output_logprobs_length": len(judgement["judgement_order2"]["original_output_logprobs"])
                        },
                    "candidate_list": candidate_list
                }

        with open(cache_file, "w", encoding="utf-8") as f:
            for (question_id, candidate_global_index1, candidate_global_index2), value in result.items():
                f.write(json.dumps({
                    "question_id": question_id,
                    "candidate_global_index1": candidate_global_index1,
                    "candidate_global_index2": candidate_global_index2,
                    "judgement_order1": value["judgement_order1"],
                    "judgement_order2": value["judgement_order2"],
                    "candidate_list": value["candidate_list"]
                }, ensure_ascii=False) + "\n")

        return result


    @staticmethod
    def transform_letter(probabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Swap labels A and B in a probability dict.
        """
        cleaned = clean_probs_dict(probabilities)
        transformed = defaultdict(float)
        for key, value in cleaned.items():
            if key == "A":
                new_key = "B"
            elif key == "B":
                new_key = "A"
            else:
                new_key = key
            transformed[new_key] += value

        return dict(transformed)
    

    @staticmethod
    def merge_probs(
        dict1: Dict[Any, float],
        dict2: Dict[Any, float],
        allow_letters: Dict = {"A", "B", "C"}
    ) -> Dict[Any, float]:
        """
        Aggregate two letter-probability maps by summing values for labels A, B, and C.
        """
        allowed = set(allow_letters)
        merged = defaultdict(float)
        for d in (dict1, dict2):
            for key, value in d.items():
                if key in allowed:
                    merged[key] += value
        return dict(merged)


    def get_pairwise_result(self, question_id: Any, idx1: int, idx2: int, pairwise_comparison_data: dict) -> Optional[str]:
        """
        Derive a single relation among {larger, smaller, equal} for a candidate pair using the selected method (probability aggregation, perplexity comparison with tolerance, or raw label agreement), with optional one-time logging.
        """
        sorted_key = (question_id, min(idx1, idx2), max(idx1, idx2))
        key = sorted_key
        try:
            judgement = pairwise_comparison_data[key]
            letter1 = judgement["judgement_order1"]["output_probs"]
            letter2 = self.transform_letter(judgement["judgement_order2"]["output_probs"])
        except Exception as e:
            return None
        
        if self.method == "likelihood":
            merged = self.merge_probs(letter1, letter2)
            if sorted_key not in self.logged_pairs:
                log_path = os.path.join(os.path.dirname(self.result_file), f"non_transitivity_{self.method}.jsonl")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                write_jsonl([{
                    "model": self.model,
                    "question_id": question_id,
                    "idx1": sorted_key[1],
                    "idx2": sorted_key[2],
                    "merged": merged
                }], log_path, append=True)
                self.logged_pairs.add(sorted_key)
            
            first, second = sorted(merged.values(), reverse=True)[:2]

            choice = max({'A','B','C'}.intersection(merged), key=merged.get, default=None)
            return "equal" if (abs(first - second) / 2)  < self.tolerance_gap else {"A": "larger", "B": "smaller", "C": "equal"}.get(choice, "unknown")
        
        elif self.method == "ppl":
            ppl_judgement_order1 = calculate_perplexity(judgement["judgement_order1"]["cumulative_logprob"], judgement["judgement_order1"]["original_output_logprobs_length"])
            ppl_judgement_order2 = calculate_perplexity(judgement["judgement_order2"]["cumulative_logprob"], judgement["judgement_order2"]["original_output_logprobs_length"])

            if sorted_key not in self.logged_pairs:
                log_path = os.path.join(os.path.dirname(self.result_file), f"non_transitivity_{self.method}.jsonl")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                write_jsonl([{
                    "model": self.model,
                    "question_id": question_id,
                    "idx1": sorted_key[1],
                    "idx2": sorted_key[2],
                    "ppl_judgement_order1": ppl_judgement_order1,
                    "ppl_judgement_order2": ppl_judgement_order2
                }], log_path, append=True)
                self.logged_pairs.add(sorted_key)
            if  ppl_judgement_order1 < ppl_judgement_order2 and abs(ppl_judgement_order1 - ppl_judgement_order2) > self.tolerance_gap:
                return {"A": "larger", "B": "smaller", "C": "equal"}.get(next(iter(letter1)))
            elif ppl_judgement_order1 > ppl_judgement_order2 and abs(ppl_judgement_order1 - ppl_judgement_order2) > self.tolerance_gap:
                return {"A": "larger", "B": "smaller", "C": "equal"}.get(next(iter(letter2)))
            else:
                return "equal"
        else:
            if sorted_key not in self.logged_pairs:
                log_path = os.path.join(os.path.dirname(self.result_file), f"non_transitivity_{self.method}.jsonl")
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                write_jsonl([{
                    "model": self.model,
                    "question_id": question_id,
                    "idx1": sorted_key[1],
                    "idx2": sorted_key[2],
                    "letter1": next(iter(letter1)),
                    "letter2": next(iter(letter2)),
                    "force_tie": not (next(iter(letter1)) == next(iter(letter2)))
                }], log_path, append=True)
                self.logged_pairs.add(sorted_key)
            if next(iter(letter1)) == next(iter(letter2)):
                return {"A": "larger", "B": "smaller", "C": "equal"}.get(next(iter(letter1)))
            else:
                return "equal"


    def process_question(self, question_id: Any, candidate_list: List[int], pairwise_comparison_data: dict) -> Tuple[int, int, Dict[str, int]]:
        """
        For a given question, enumerate all k-sized candidate subsets, evaluate all induced triples for transitivity, and accumulate counts of non-transitive cases and error types.
        """
        non_transitive_count = 0
        total_groups = 0
        error_type_counts: Dict[str, int] = defaultdict(int)

        for group in itertools.combinations(candidate_list, self.length):
            group_valid = True
            group_transitive = True
            group_sorted = sorted(group)
            group_error_types: Set[str] = set()

            for triple in itertools.combinations(group_sorted, 3):
                i, j, k = triple
                op1 = self.get_pairwise_result(question_id, i, j, pairwise_comparison_data)
                op2 = self.get_pairwise_result(question_id, j, k, pairwise_comparison_data)
                op3 = self.get_pairwise_result(question_id, i, k, pairwise_comparison_data)
                if op1 is None or op2 is None or op3 is None:
                    group_valid = False
                    break
                result = evaluate_triple(op1, op2, op3)
                if not result["result"] and result.get("type"):
                    group_error_types.add(result["type"])
            if group_valid:
                total_groups += 1
                if group_error_types:
                    non_transitive_count += 1
                    for et in group_error_types:
                        error_type_counts[et] += 1
        question_category = self.question_with_category.get(question_id, "No Category")
        return non_transitive_count, total_groups, error_type_counts, question_category


    def calculate_non_transitivity_proportion(self) -> dict:
        """
        Run per-question analyses in parallel and compute overall and per-category non-transitivity proportions and error-type distributions.
        """
        total_non_transitive = 0
        total_groups = 0
        total_error_type_counts: Dict[str, int] = defaultdict(int)
        category_group_counts: Dict[str, int] = defaultdict(int)
        category_non_trans_counts: Dict[str, int] = defaultdict(int)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            results = list(tqdm(
                executor.map(
                    lambda item: self.process_question(item[0], item[1], self.pairwise_comparison_data),
                    {qid: v["candidate_list"] for (qid, _, _), v in self.pairwise_comparison_data.items()}.items()
                ),
                total=len({qid: v["candidate_list"] for (qid, _, _), v in self.pairwise_comparison_data.items()}),
                desc="Processing questions"
            ))

        for non_trans, groups, err_counts, question_category in results:
            total_non_transitive += non_trans
            total_groups += groups
            for et, cnt in err_counts.items():
                total_error_type_counts[et] += cnt
            category_group_counts[question_category] += groups
            category_non_trans_counts[question_category] += non_trans

        if total_groups == 0:
            raise ValueError("No valid candidate answer combinations were generated.")

        error_type_proportions = {
            et: total_error_type_counts[et] / total_groups
            for et in total_error_type_counts
        }

        category_non_transitive_proportion = {}
        for cat, grp_cnt in category_group_counts.items():
            non_trans_cnt = category_non_trans_counts[cat]
            category_non_transitive_proportion[cat] = {
                "category_total": grp_cnt,
                "non_trans_count": non_trans_cnt,
                "non_transitive_proportion": non_trans_cnt / grp_cnt if grp_cnt > 0 else 0.0
            }

        result_dict = {
            "total": total_groups,
            "non_transitive_count": total_non_transitive,
            "non_transitive_proportion": total_non_transitive / total_groups,
            "error_type_counts": dict(total_error_type_counts),
            "error_type_proportions": error_type_proportions,
            "category_non_transitive_proportion": category_non_transitive_proportion
        }

        return result_dict
    

    def store_results(self, non_transitivity_result: dict):
        """
        Upsert results keyed by (model, length, method, tolerance) into a JSONL file.
        """
        try:
            existing_results = read_jsonl(self.result_file)
            if not isinstance(existing_results, list):
                existing_results = []
        except Exception:
            existing_results = []
        
        record_found = False
        for record in existing_results:
            if record.get("model") == self.model and record.get("length") == self.length and record.get("method") == self.method and record.get("tolerance") == self.tolerance_gap:
                record["non_transitivity_proportion"] = non_transitivity_result
                record_found = True
                break

        if not record_found:
            new_record = {
                "model": self.model,
                "length": self.length,
                "method": self.method,
                "tolerance": self.tolerance_gap,
                "non_transitivity_proportion": non_transitivity_result
            }
            existing_results.append(new_record)

        existing_results.sort(key=lambda x: (x.get("model"), x.get("length")))
        write_jsonl(existing_results, self.result_file, append=False)
        return existing_results


def main():
    parser = argparse.ArgumentParser(description="Calculate Pairwise Transitivity Inconsistency.")
    parser.add_argument("--question-file", type=str, default="./data/answers/filtered_selected_answers_with_category.jsonl", help="Contains the category information of questions")
    parser.add_argument("--pairwise-comparison-file", type=str, required=True, help="Path to the JSONL or JSON file containing Pairwise Comparison data")
    parser.add_argument("--result-file", type=str, required=True, help="Path to the JSONL file to store the final statistical results")
    parser.add_argument("--workers", type=int, default=1, help="Number of threads for multithreading")
    parser.add_argument("--length", type=int, required=True, help="Number of candidate answers used for the transitivity check (e.g., 5)")
    parser.add_argument("--tolerance-gap", type=float, default=0.0, help="Tolerance gap for score differences")
    parser.add_argument("--model", type=str, required=True, help="Model name, used to label the statistical results")
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "probs", "ppl"])
    args = parser.parse_args()

    question_info = read_jsonl(args.question_file)
    question_with_category = {item['question_id']: item['category'] for item in question_info}

    calculator = NonTransitivityCalculator(
        question_with_category=question_with_category,
        pairwise_comparison_file=args.pairwise_comparison_file,
        result_file=args.result_file,
        length=args.length,
        tolerance_gap=args.tolerance_gap,
        workers=args.workers,
        model=args.model,
        method=args.method
    )

    non_transitivity_result = calculator.calculate_non_transitivity_proportion()
    stored_results = calculator.store_results(non_transitivity_result)
    print("Non-transitivity proportion Calculation Result:")
    print(non_transitivity_result)
    print("Stored Results:")
    print(stored_results)

if __name__ == "__main__":
    main()
