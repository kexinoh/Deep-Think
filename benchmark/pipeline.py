"""Benchmark pipeline for evaluating Deep Thinker on a set of problems.

This script loads a dataset of problems and uses the main framework to attempt
solving each one. Results are printed to stdout with a simple solved/not solved
metric based on string matching of the expected answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

from config import AppConfig
from llm_service import LLMService
from main import SentenceLevelMathTask
from mcts import DeepThinker


async def _run_single_problem(problem: str, expected: str, config: AppConfig) -> Dict[str, str]:
    """Run Deep Thinker on a single problem and return the result summary."""

    llm_service = LLMService(api_key=config.api_key, base_url=config.api_base_url)
    task = SentenceLevelMathTask(
        problem=problem,
        num_actions=config.task_num_actions,
        max_steps=config.task_max_steps,
        llm_service=llm_service,
        config=config,
    )
    thinker = DeepThinker(task=task, config=config)

    final_state = await thinker.think()
    formatted = await task.format_result(final_state)
    solved = expected.lower() in formatted.lower()

    return {
        "problem": problem,
        "expected": expected,
        "result": formatted,
        "solved": str(solved),
    }


async def run_benchmark(dataset: List[Dict[str, str]], config: AppConfig) -> List[Dict[str, str]]:
    """Run the benchmark across all problems in *dataset*."""

    results = []
    for item in dataset:
        problem = item["problem"]
        answer = item.get("answer", "")
        result = await _run_single_problem(problem, answer, config)
        results.append(result)
    return results


def main() -> None:
    """Command line entry point for the benchmark pipeline."""

    parser = argparse.ArgumentParser(description="Run Deep Thinker benchmark")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path(__file__).with_name("sample_problems.json")),
        help="Path to JSON file containing problems",
    )
    args = parser.parse_args()

    config = AppConfig()
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    results = asyncio.run(run_benchmark(dataset, config))

    solved_count = sum(1 for r in results if r["solved"] == "True")
    print(f"Solved {solved_count}/{len(results)} problems")
    for r in results:
        print("-" * 80)
        print(f"Problem: {r['problem']}")
        print(r["result"])
        print(f"Expected answer: {r['expected']}")
        print(f"Solved: {r['solved']}")


if __name__ == "__main__":
    main()
