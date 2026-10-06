#!/usr/bin/env bash
set -euo pipefail

# Intentionally requires explicit judge model choices/credentials. This script
# never pretends a contemporary API is the retired original hosted judge.
out='runs/targets/v1/native_scorers'
echo "Generated request files:"
echo "  $out/sycophancy_gpt4_requests.jsonl"
echo "  $out/do_not_answer_gpt4_requests.jsonl"
echo "  $out/truthfulqa_gemini_requests.jsonl"
echo
echo 'Run a declared modern OpenAI judge over the *released source templates*, then apply the released parsers:'
echo "persona-audit run-openai-judge --requests $out/sycophancy_gpt4_requests.jsonl --model YOUR_MODEL --workers 8 --output $out/sycophancy_judge_responses.jsonl"
echo "persona-audit apply-judge-responses --family sycophancy --requests $out/sycophancy_gpt4_requests.jsonl --responses $out/sycophancy_judge_responses.jsonl --output $out/sycophancy_scores.jsonl"
echo "persona-audit run-openai-judge --requests $out/do_not_answer_gpt4_requests.jsonl --model YOUR_MODEL --workers 8 --output $out/do_not_answer_judge_responses.jsonl"
echo "persona-audit apply-judge-responses --family do_not_answer --requests $out/do_not_answer_gpt4_requests.jsonl --responses $out/do_not_answer_judge_responses.jsonl --output $out/do_not_answer_scores.jsonl"
echo
echo 'For the pinned TruthfulQA repository Gemini scorer (requires GEMINI_API_KEY):'
echo "persona-audit score-truthfulqa-gemini --requests $out/truthfulqa_gemini_requests.jsonl --source-root data/snapshots/target_sources_v1 --model YOUR_GEMINI_MODEL --output $out/truthfulqa_scores.jsonl"
