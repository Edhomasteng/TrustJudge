#!/usr/bin/env python
from collections import defaultdict
import os
import json
import math
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Union, List, Dict, Iterable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import read_jsonl, write_jsonl, extract_middle_dict


class ConflictCalculator:
    def __init__(self, question_with_category: Dict[int, str], single_answer_grade_file: str, single_answer_grade_scale: int, pairwise_comparison_file: str, result_file: str, model: str, tolerance_gap: float, workers: int = 1, method: str = "baseline"):
        self.question_with_category = question_with_category
        self.single_answer_grade_file = single_answer_grade_file
        self.single_answer_grade_scale = single_answer_grade_scale
        self.pairwise_comparison_file = pairwise_comparison_file
        self.result_file = result_file
        self.workers = workers
        self.model = model
        self.tolerance_gap = tolerance_gap
        self.method = method
        self.single_answer_data = self.set_single_answer_grade_data()
        self.pairwise_comparison_data = self.set_pairwise_comparison_data()


    def set_single_answer_grade_data(self) -> dict:
        """
        Load or build a cache mapping (question_id, candidate_global_index) to judgement-probabilities for single-score.
        """
        cache_dir = Path("./data/judgements_cache/single_answer_grade")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / os.path.basename(self.single_answer_grade_file)

        # 1) Cache hit: load visible JSONL into dict
        if cache_file.exists():
            result = {}
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    question_id = record["question_id"]
                    candidate_global_index = record["candidate_global_index"]
                    judgement = record["judgement"]
                    result[(question_id, candidate_global_index)] = judgement
            return result
        
        data = read_jsonl(self.single_answer_grade_file)
        result = {}
        for item in tqdm(data, desc="reading single answer grade data"):
            question_id = item.get("question_id")
            for judgement in item.get("judgements", []):
                candidate_global_index = judgement.get("candidate_global_index")
                try:
                    allowed_numbers = set(range(1, self.single_answer_grade_scale + 1))
                    result[(question_id, candidate_global_index)] = extract_middle_dict(judgement["judgement"]["output_probs"], allowed_first_keys=allowed_numbers)
                except Exception as e:
                    continue

        with open(cache_file, "w", encoding="utf-8") as f:
            for (question_id, candidate_global_index), value in result.items():
                f.write(json.dumps({
                    "question_id": question_id,
                    "candidate_global_index": candidate_global_index,
                    "judgement": value
                }, ensure_ascii=False) + "\n")

        return result


    def set_pairwise_comparison_data(self) -> dict:
        """
        Load or build a cache mapping (question_id, idx1, idx2) to bidirectional pairwise comparison judgement data (probs, logprobs length, candidate list).
        """
        cache_dir = Path("./data/judgements_cache/pairwise_comparison")
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
                    "candidate_list": candidate_list
                }, ensure_ascii=False) + "\n")

        return result


    def compute_score(self, result: Dict) -> float:
        """
        Convert a judgement-probabilities dict into a scalar score using the selected method.
        """
        allowed_numbers = set(range(1, self.single_answer_grade_scale + 1))
        if self.method == "softmax_weighted":
            return self.compute_softmax_weighted_score(result, allowed_numbers)
        if self.method == "geval":
            return self.compute_weighted_score(result, allowed_numbers)
        else:
            return next(iter(result)).strip()


    @staticmethod
    def compute_weighted_score(score_dict: Dict[Any, float], allowed_scores: Iterable[Any] = None) -> float:
        """
        Compute a probability-weighted average of numeric scores directly from provided probabilities.
        """
        if allowed_scores is not None:
            try:
                allowed_set = {float(x) for x in allowed_scores}
            except Exception as e:
                raise ValueError("The elements in allowed_scores must be convertible to the float type.") from e
        else:
            allowed_set = None

        valid_items = []
        for k, prob in score_dict.items():
            try:
                score = float(str(k).strip())
            except Exception:
                continue
            if allowed_set is None or score in allowed_set:
                valid_items.append((score, prob))
        
        if not valid_items:
            return 0.0

        weighted_sum = sum(score * weight for score, weight in valid_items)
        return weighted_sum
        
    @staticmethod
    def compute_softmax_weighted_score(
        score_dict: Dict[Any, float],
        allowed_scores: Iterable[Any] = None,
        T: float = 0.2,
        eps: float = 1e-8
    ) -> float:
        """
        Convert probabilities to logits, apply softmax to get weights, and return the weighted score.
        """
        if allowed_scores is not None:
            try:
                allowed_set = {float(x) for x in allowed_scores}
            except Exception as e:
                raise ValueError("The elements in allowed_scores must be convertible to the float type") from e
        else:
            allowed_set = None

        valid_items: List[Tuple[float, float]] = []
        for k, p in score_dict.items():
            try:
                score = float(str(k).strip())
            except Exception:
                continue
            if allowed_set is None or score in allowed_set:
                if p <= 0 or p >= 1:
                    continue
                valid_items.append((score, p))
        if not valid_items:
            return 0.0
        logits = [math.log((p + eps) / (1 - p + eps)) for _, p in valid_items]
        exp_logits = [math.exp(l / T) for l in logits]
        sum_exp = sum(exp_logits)
        weights = [x / sum_exp for x in exp_logits]
        weighted_sum = sum(score * w for (score, _), w in zip(valid_items, weights))
        return weighted_sum


    @staticmethod
    def transform_letter(letter: str) -> str:
        """
        Swap A and B.
        """
        if letter == "A":
            return "B"
        elif letter == "B":
            return "A"
        elif letter == "C":
            return "C"
        else:
            raise ValueError(f"Invalid letter value: {letter}")

    
    def calculate_conflict_proportion(self) -> dict:
        """
        Compare pairwise outcomes with expected ordering from single-answer scores (with tolerance) and aggregate conflict rates per category and overall.
        """
        def process_item(key, pairwise_item):
            question_id, idx1, idx2 = key

            try:
                letter1 = next(iter( pairwise_item["judgement_order1"]["output_probs"])).strip()
                letter2 = self.transform_letter(next(iter( pairwise_item["judgement_order2"]["output_probs"])).strip())
            except Exception as e:
                return None

            if letter1 == "A" and letter2 == "A":
                pairwise_result = "larger"
            elif letter1 == "B" and letter2 == "B":
                pairwise_result = "smaller"
            else:
                pairwise_result = "equal"


            try:
                probs1 = self.single_answer_data[(question_id, idx1)]
                probs2 = self.single_answer_data[(question_id, idx2)]
                score1 = float(self.compute_score(probs1))
                score2 = float(self.compute_score(probs2))
            except Exception as e:
                return None

            diff = abs(score1 - score2)
            if diff < self.tolerance_gap:
                expected = {"equal"}
            else:
                if score1 > score2:
                    expected = {"larger"}
                elif score1 < score2:
                    expected = {"smaller"}
                else:
                    expected = {"equal"}

            is_conflict = pairwise_result not in expected
            category = self.question_with_category.get(question_id, "No Category")
            return is_conflict, category
        
        category_total = defaultdict(int)
        category_conflicts = defaultdict(int)

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(process_item, key, item)
                for key, item in self.pairwise_comparison_data.items()
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
                result = future.result()
                if result is None:
                    continue
                conflict_flag, category = result
                category_total[category] += 1
                if conflict_flag:
                    category_conflicts[category] += 1

        total = sum(category_total.values())
        conflict_count = sum(category_conflicts.values())
        conflict_proportion = conflict_count / total if total > 0 else 0.0

        category_conflict_proportion = {
            cat: {
                "category_total": category_total[cat],
                "conflict_count": category_conflicts[cat],
                "conflict_proportion": (category_conflicts[cat] / category_total[cat]) if category_total[cat] else 0.0
            }
            for cat in category_total
        }

        return {
            "total": total,
            "conflict_count": conflict_count,
            "conflict_proportion": conflict_proportion,
            "category_conflict_proportion": category_conflict_proportion
        }

   
    def store_results(self, conflict_result: dict):
        """
        Merge the new conflict statistics into the results JSONL file.
        """
        try:
            existing_results = read_jsonl(self.result_file)
            if not isinstance(existing_results, list):
                existing_results = []
        except Exception:
            existing_results = []
        
        record_found = False
        for record in existing_results:
            if record.get("model") == self.model and record.get("tolerance") == self.tolerance_gap and record.get("method") == self.method:
                record["conflict_proportion"] = conflict_result
                record_found = True
                break
        
        if not record_found:
            new_record = {
                "model": self.model,
                "tolerance": self.tolerance_gap,
                "method": self.method,
                "conflict_proportion": conflict_result
            }
            existing_results.append(new_record)

        existing_results.sort(key=lambda x: (x.get("model"), x.get("tolerance"), x.get("method")))
        write_jsonl(existing_results, self.result_file, append=False)
        return existing_results


def main():
    parser = argparse.ArgumentParser(description="Calculate Score-Comparison Inconsistency.")
    parser.add_argument("--question-file", type=str, default="./data/answers/filtered_selected_answers_with_category.jsonl", help="The question file should contains the category information of each question.")
    parser.add_argument("--single-answer-grade-file", type=str, required=True, help="Path to the single answer grade JSONL/JSON file")
    parser.add_argument("--single-answer-grade-scale", type=int, default=100, help="The grading scale of single answer grade")
    parser.add_argument("--pairwise-comparison-file", type=str, required=True, help="Path to the pairwise comparison JSONL/JSON file")
    parser.add_argument("--result-file", type=str, required=True, help="Path to the output result file (JSONL)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads for parallel processing")
    parser.add_argument("--tolerance-gap", type=float, default=0.0, help="Tolerance gap for score differences")
    parser.add_argument("--model", type=str, required=True, help="Model name for the result record")
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "softmax_weighted", "geval"])
    args = parser.parse_args()

    question_info = read_jsonl(args.question_file)
    question_with_category = {item['question_id']: item['category'] for item in question_info}

    calculator = ConflictCalculator(
        question_with_category=question_with_category,
        single_answer_grade_file=args.single_answer_grade_file,
        single_answer_grade_scale=args.single_answer_grade_scale,
        pairwise_comparison_file=args.pairwise_comparison_file,
        result_file=args.result_file,
        model=args.model,
        tolerance_gap=args.tolerance_gap,
        workers=args.workers,
        method=args.method
    )

    conflict_result = calculator.calculate_conflict_proportion()
    stored_results = calculator.store_results(conflict_result)
    print("Conflict proportion Calculation Result:")
    print(conflict_result)
    print("Stored Results:")
    print(stored_results)

if __name__ == "__main__":
    main()
