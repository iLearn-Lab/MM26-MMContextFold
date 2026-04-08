# Datasets

This directory contains the evaluation benchmarks used in the MM-ContextFold paper.

## Supported Datasets

| Dataset | Description | Samples |
|---------|-------------|---------|
| MMSearch+ | Multimodal search questions requiring web retrieval | - |
| MMSearch | Multimodal search benchmark | - |
| World-VQA | World knowledge visual question answering | - |
| VDR-Bench | Visual deep reasoning benchmark | - |
| SimpleVQA | Simple visual question answering | - |
| LiveVQA | Live/temporal visual question answering | - |
| BrowseComp-VL | Visual browsing comprehension | - |

## Data Format

Each dataset uses a unified JSONL format (`data_unified.jsonl`):

```json
{
  "id": "sample_001",
  "query": "What is shown in the image?",
  "query_images": ["image_001.png"],
  "answer": "ground truth answer",
  "options": ["A", "B", "C", "D"]
}
```

## Directory Structure

```
dataset/
├── mmsearch_plus/
│   ├── data_unified.jsonl
│   └── imgs/
├── world_vqa/
│   ├── data_unified.jsonl
│   └── imgs/
├── vdr_bench/
│   ├── data_unified.jsonl
│   └── imgs/
└── ...
```

## Obtaining the Data

Due to licensing restrictions, the actual data files are not included in this repository.
Please refer to the original dataset repositories to obtain the data:

- **MMSearch+**: [link to be added]
- **World-VQA**: [link to be added]  
- **VDR-Bench**: [link to be added]
- **SimpleVQA**: [link to be added]
- **LiveVQA**: [link to be added]
- **BrowseComp-VL**: [link to be added]

After downloading, place the data files in the corresponding directories following the structure above.
