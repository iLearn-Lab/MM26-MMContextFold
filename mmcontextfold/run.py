#!/usr/bin/env python3
"""
run.py — MM-ContextFold Pipeline Runner

Usage:
    python run.py --method mmcontextfold --index 0
    python run.py --method mmcontextfold --start 0 --end 3
    python run.py --method mmcontextfold --all
    python run.py --list-methods
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from configs import load_config, get_dataset_config, get_model_config
from pipelines import get_pipeline, list_pipelines
from pipelines.base_pipeline import PipelineConfig


MODEL_ALIASES = {
    'gemini-2.5-flash': 'gemini-2.5-flash',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='MM-ContextFold Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --method mmcontextfold --index 0
  python run.py --method mmcontextfold --start 0 --end 3
  python run.py --list-methods
        """
    )

    parser.add_argument('--method', type=str, default='mmcontextfold',
                        help='Pipeline method (default: mmcontextfold)')
    parser.add_argument('--dataset', type=str, default='mmsearch_plus',
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash',
                        help='Model name')

    parser.add_argument('--index', type=int, default=None,
                        help='Single sample index')
    parser.add_argument('--start', type=int, default=None,
                        help='Start index (inclusive)')
    parser.add_argument('--end', type=int, default=None,
                        help='End index (exclusive)')
    parser.add_argument('--all', action='store_true',
                        help='Run all samples')

    parser.add_argument('--max-rounds', type=int, default=15,
                        help='Maximum rounds')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory')
    parser.add_argument('--list-methods', action='store_true',
                        help='List available methods')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_methods:
        print("Available methods:")
        for method in list_pipelines():
            print(f"  - {method}")
        return

    yaml_config = load_config("base")
    dataset_config = get_dataset_config(yaml_config, args.dataset)
    model_name = MODEL_ALIASES.get(args.model, args.model)
    model_config = get_model_config(yaml_config, model_name)

    if args.method == "mmcontextfold":
        from pipelines.pipeline_mmcontextfold import MMContextFoldConfig
        config = MMContextFoldConfig(
            dataset=args.dataset,
            data_file=os.path.join(BASE_DIR, dataset_config['data_file']),
            img_base_dir=os.path.join(BASE_DIR, dataset_config['img_base_dir']),
            model=args.model,
            model_full_name=model_config['full_name'],
            start_idx=args.start if args.start is not None else 0,
            end_idx=args.end,
            single_idx=args.index,
            output_dir=args.output_dir or "",
            max_rounds=args.max_rounds,
        )
    else:
        config = PipelineConfig(
            dataset=args.dataset,
            data_file=os.path.join(BASE_DIR, dataset_config['data_file']),
            img_base_dir=os.path.join(BASE_DIR, dataset_config['img_base_dir']),
            model=args.model,
            model_full_name=model_config['full_name'],
            start_idx=args.start if args.start is not None else 0,
            end_idx=args.end,
            single_idx=args.index,
            output_dir=args.output_dir or "",
            max_rounds=args.max_rounds,
        )

    if args.all:
        config.start_idx = 0
        config.end_idx = None
        config.single_idx = None

    pipeline_class = get_pipeline(args.method)

    print(f"{'='*60}")
    print(f"MM-ContextFold Pipeline Runner")
    print(f"{'='*60}")
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model} -> {model_config['full_name']}")
    print(f"Max Rounds: {args.max_rounds}")
    print(f"{'='*60}")

    pipeline = pipeline_class(config)
    results = pipeline.run()

    return results


if __name__ == "__main__":
    main()
