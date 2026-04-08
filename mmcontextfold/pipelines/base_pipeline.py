"""
base_pipeline.py

Abstract base class for all pipelines.
"""

import os
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    dataset: str = "mmsearch_plus"
    data_file: str = ""
    img_base_dir: str = ""
    model: str = "gemini-2.5-flash"
    model_full_name: str = "google/gemini-2.5-flash"
    start_idx: int = 0
    end_idx: Optional[int] = None
    single_idx: Optional[int] = None
    output_dir: str = ""
    max_rounds: int = 15


class SampleLogger:
    def __init__(self, log_dir: str, sample_id: str, method_name: str = "Pipeline"):
        self.log_dir = log_dir
        self.sample_id = sample_id
        self.method_name = method_name
        self.log_file = os.path.join(log_dir, f"log_{sample_id}.txt")

        os.makedirs(log_dir, exist_ok=True)

        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"Sample ID: {sample_id}\n")
            f.write(f"Method: {method_name}\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")

    def log(self, message: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(f"[{timestamp}] {message}")

    def close(self, status: str = "completed"):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Status: {status}\n")
            f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n")


_current_logger: Optional[SampleLogger] = None


def log(message: str):
    global _current_logger
    if _current_logger:
        _current_logger.log(message)
    else:
        print(message)


def set_logger(logger: SampleLogger):
    global _current_logger
    _current_logger = logger


def get_logger() -> Optional[SampleLogger]:
    return _current_logger


class BasePipeline(ABC):

    method_name: str = "base"

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results: List[Dict] = []

        if not self.config.output_dir:
            self.config.output_dir = self._default_output_dir()

        os.makedirs(self.config.output_dir, exist_ok=True)
        self.data = self._load_data()

    def _default_output_dir(self) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(
            base_dir,
            "outputs",
            self.config.dataset,
            f"{self.config.model}_{self.method_name}_{timestamp}"
        )

    def _load_data(self) -> List[Dict]:
        data = []
        with open(self.config.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _get_indices(self) -> List[int]:
        if self.config.single_idx is not None:
            return [self.config.single_idx]
        start = self.config.start_idx
        end = self.config.end_idx if self.config.end_idx is not None else len(self.data)
        return list(range(max(0, start), min(len(self.data), end)))

    def _prepare_question(self, item: Dict) -> str:
        if item.get('options'):
            return f"{item['query']}\n\nOptions:\n" + "\n".join([f"- {opt}" for opt in item['options']])
        return item['query']

    def _prepare_images(self, item: Dict) -> List[str]:
        img_paths = []
        for img_name in item.get('query_images', []):
            img_path = os.path.join(self.config.img_base_dir, img_name)
            if os.path.exists(img_path):
                img_paths.append(img_path)
        return img_paths

    @abstractmethod
    def process_single(self, item: Dict, index: int) -> Dict:
        pass

    def run(self) -> List[Dict]:
        indices = self._get_indices()

        print(f"{'='*60}")
        print(f"Starting {self.method_name} Pipeline")
        print(f"Dataset: {self.config.dataset}")
        print(f"Model: {self.config.model}")
        print(f"Samples: {len(indices)}")
        print(f"Output: {self.config.output_dir}")
        print(f"{'='*60}")

        for i, idx in enumerate(indices):
            print(f"\n{'─'*40}")
            print(f"Progress: {i+1}/{len(indices)} (index={idx})")

            item = self.data[idx]
            result = self.process_single(item, idx)
            self.results.append(result)
            self._save_results()

        success = sum(1 for r in self.results if r.get('status') == 'success')

        print(f"\n{'='*60}")
        print(f"Completed: {success}/{len(self.results)} success")
        print(f"Results saved to: {self.config.output_dir}")
        print(f"{'='*60}")

        return self.results

    def _save_results(self):
        results_file = os.path.join(self.config.output_dir, "results.jsonl")
        with open(results_file, 'w', encoding='utf-8') as f:
            for r in self.results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        config_file = os.path.join(self.config.output_dir, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump({
                "method": self.method_name,
                "dataset": self.config.dataset,
                "model": self.config.model,
                "model_full_name": self.config.model_full_name,
                "max_rounds": self.config.max_rounds,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
