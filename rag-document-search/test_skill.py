#!/usr/bin/env python3
"""Test the RAG search skill with sample queries."""

import json
import sys
from scripts.semantic_search import SemanticSearcher, format_results

# Add to allow running from any directory
sys.path.insert(0, '.')

def run_tests():
    """Run test evals."""

    with open('evals/evals.json', 'r') as f:
        evals_data = json.load(f)

    searcher = SemanticSearcher('references/')

    results = {}

    for eval_case in evals_data['evals']:
        eval_id = eval_case['id']
        prompt = eval_case['prompt']

        print(f"\n{'='*70}")
        print(f"EVAL {eval_id}: {prompt[:50]}...")
        print(f"{'='*70}")

        # Run search
        search_results = searcher.search(prompt, top_k=3)

        # Format output
        formatted = format_results(search_results, prompt)
        print(formatted)

        results[f"eval_{eval_id}"] = {
            "prompt": prompt,
            "num_results": len(search_results),
            "results": search_results
        }

    return results

if __name__ == "__main__":
    results = run_tests()

    # Save results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("Test complete. Results saved to test_results.json")
    print(f"{'='*70}")
