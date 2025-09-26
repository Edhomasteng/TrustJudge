
### 1. Data Demo Format
> What this is: JSONL examples containing questions and multiple candidate answers. All later steps read from these files and write judgments/metrics into `data/judgements/*` and `data/results/*`.



```json
{
    "question_id": ..., 
    "category": ..., 
    "question": ...,
    "answers": [
    {"text": ["...", "...", ...]},
    {"text": ["...", "...", ...]},
    ....
    ]
}
```
Recommended: Build your dataset with the same schema as the demo (keys and types below). The evaluation scripts assume this structure.

---

### 2. Generate Single-Score

**What this does:** Invokes the judge LLM to assign a **single score per answer** using the prompt label you choose.

* Use `single_answer_grade_5_points` for a **1–5** discrete score.
* Use `single_answer_grade_100_points` to compute **distribution-sensitive/softmax-weighted expectations** (denser label space preserves more information).
* With `--top-logprobs`, the script saves per-label (or token) probabilities/logprobs for later aggregation.

```sh
models=(
    "model paths"
)
single_answer_grade_prompt_labels=(
    "single_answer_grade_5_points"
    "single_answer_grade_100_points"
)
for model in "${models[@]}"; do
    model_name=$(basename "${model}")
    for label in "${single_answer_grade_prompt_labels[@]}"; do
        python SingleAnswerGrade.py \
            --model ${model} \
            --input-file ./data/answers/filtered_selected_answers.jsonl \
            --output-file ./data/judgements/single_answer_grade/${model_name}_${label}.jsonl \
            --prompt-file ./prompts/single_answer_grade.jsonl \
            --prompt-label ${label} \
            --temperature 1.0 \
            --max-tokens 2048 \
            --top-logprobs 20 \
            --tensor-parallel-size  8 \
            --gpu-ids 0,1,2,3,4,5,6,7 \
            --openai-batch-size 1 \
            # --openai-api-key  sk-xxxxxxxxxxxxxx \
            # --openai-api-base https://api.openai.com/v1/chat/completions \
            # --test-number 1
    done
done
```

**Expected output schema (per question):** The file records the question, candidate answers with their **single scores**, and the judge’s raw/probabilistic outputs, which are later used by the inconsistency calculators.

```json
{
    "question_id": ...,
    "question": "...",
    "candidate_answers": [
        {
            "global_index": ...,
            "score": ...,
            "local_index": ...,
            "text": "..."
        },
        ...
    ],
    "judgements": [
        {
            "candidate_global_index": ...,
            "judgement": {
                "prompt": "...",
                "prompt_token_ids": [...],
                "encoder_prompt": null,
                "encoder_prompt_token_ids": null,
                "prompt_logprobs": null,
                "cumulative_logprob": ...,
                "output_text": "...",
                "output_token_ids": [...],
                "original_output_logprobs": [...],
                "output_logprobs": [...],
                "output_probs": [...]
            }
        },
        ...
    ]
}
```

---

### 3. Generate Pairwise Comparison

**What this does:** Produces **pairwise preferences** between answers. For each pair it queries the judge twice (order 1→2 and 2→1) to reduce position bias, saving **win/tie/lose** decisions and probabilities/logprobs for each order. These outputs feed into likelihood-aggregation or PPL-based method in the inconsistency metrics.

```sh
models=(
    "model paths"
)
pairwise_comparisonprompt_labels=(
    "pairwise_comparison"
)
for model in "${models[@]}"; do
    model_name=$(basename "${model}")
    for label in "${pairwise_comparisonprompt_labels[@]}"; do
        python PairwiseComparison.py \
            --model ${model} \
            --input-file ./data/answers/filtered_selected_answers.jsonl \
            --output-file ./data/judgements/pairwise_comparison/${model_name}_${label}.jsonl \
            --prompt-file ./prompts/pairwise_comparison.jsonl \
            --prompt-label ${label} \
            --temperature 1.0 \
            --max-tokens 2048 \
            --top-logprobs 20 \
            --tensor-parallel-size 8 \
            --gpu-ids 0,1,2,3,4,5,6,7 \
            --openai-batch-size 1 \
            # --openai-api-key  sk-xxxxxxxxxxxxxx \
            # --openai-api-base https://api.openai.com/v1/chat/completions \
            # --test-number 1
    done
done
```

**Expected output schema (per question):** Two judgments (`judgement_order1`, `judgement_order2`) correspond to the two presentation orders. The likelihood aggregation sums their probabilities; ppl-based method uses the average perplexity.

```json
{
    "question_id": ...,
    "question": "...",
    "candidate_answers": [
        {
            "global_index": ...,
            "score": ...,
            "local_index": ...,
            "text": "..."
        },
        ...
    ],
    "judgements": [
        {
            "candidate_global_index1": ...,
            "candidate_global_index2": ...,
            "judgement_order1": {
                "prompt": "...",
                "prompt_token_ids": [...],
                "encoder_prompt": null,
                "encoder_prompt_token_ids": null,
                "prompt_logprobs": null,
                "cumulative_logprob": ...,
                "output_text": "...",
                "output_token_ids": [...],
                "original_output_logprobs": [...],
                "output_logprobs": [...],
                "output_probs": [...]
            },
            "judgement_order2": {
                "prompt": "...",
                "prompt_token_ids": [...],
                "encoder_prompt": null,
                "encoder_prompt_token_ids": null,
                "prompt_logprobs": null,
                "cumulative_logprob": ...,
                "output_text": "...",
                "output_token_ids": [...],
                "original_output_logprobs": [...],
                "output_logprobs": [...],
                "output_probs": [...]
            }
        },
        ...
    ]
}
```

---

### 4. Calculate Score-Comparison Inconsistency

**What this does:** Computes the **conflict ratio** between single-answer scores and pairwise preferences. You can choose how the single scores are interpreted before comparison:

* `baseline`: compare **discrete** 1–5 scores directly against pairwise outcomes.
* `geval`: compute an **weighted score** from the label probabilities (no softmax normalization).
* `softmax_weighted`: apply **softmax** over the probabilities and compute **weighted score**.
  Use `--tolerance-gap` to treat tiny score differences as **ties**.

```sh
models=(
    "model paths"
)
ConflictCalculator_methods=("baseline" "geval" "softmax_weighted")
tolerance_gap=(0)
for model in "${models[@]}"; do
    model_name=$(basename "${model}")
    for method in "${ConflictCalculator_methods[@]}"; do
      if [[ "$method" == "softmax_weighted" ]]; then
        single_answer_grade_scale=100
      else
        single_answer_grade_scale=5
      fi
      python ConflictCalculator.py \
        --single-answer-grade-file "./data/judgements/single_answer_grade/${model_name}_single_answer_grade_${single_answer_grade_scale}_points.jsonl" \
        --single-answer-grade-scale ${single_answer_grade_scale} \
        --pairwise-comparison-file "./data/judgements/pairwise_comparison/${model_name}_pairwise_comparison.jsonl" \
        --result-file "./data/results/score_comparison_inconsistency.jsonl" \
        --workers 10 \
        --tolerance-gap ${tolerance_gap} \
        --model ${model_name} \
        --method ${method}
  done
done
```

---

### 5. Calculate Pairwise Transitivity Inconsistency

**What this does:** Measures **non-transitivity** within k-item subsets (`--length` ∈ {3,4,5}). It detects **cycles** (A > B > C > A)  **equivalence inconsistencies**(A = B = C ≠ A).
Aggregation options:

* `baseline`: use two orders; disagreements become ties; check transitivity directly.
* `ppl`: compare orderings via **perplexity** and pick the lower-PPL side (less uncertain) when undecided/tied,.
* `likelihood`: **sum** win/tie/lose probabilities from both orders, then **argmax**.
  `--tolerance-gap` again treats very small margins of ppl difference or sumer probability differece as ties.

```sh
models=(
    "model paths"
)
NonTransitivityCalculator_methods=("baseline" "ppl" "likelihood")
length=(3 4 5)
  for method in "${NonTransitivityCalculator_methods[@]}"; do
    for len in "${length[@]}"; do
      for model in "${models[@]}"; do
        model_name=$(basename "${model}")
        tolerance_gap=(0)
        python NonTransitivityCalculator.py \
          --pairwise-comparison-file "./data/judgements/pairwise_comparison/${model_name}_pairwise_comparison.jsonl" \
          --result-file "./data/results/pairwise_transitivity_inconsistency.jsonl" \
          --workers 10 \
          --length "${len}" \
          --model "${model_name}" \
          --method "${method}" \
          --tolerance-gap "${tolerance_gap}"
    done
  done
done
```