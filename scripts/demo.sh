cd ../trustjudge

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