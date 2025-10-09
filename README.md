# 🇬🇪 Georgian Attractions Dataset Creator

A modular Python pipeline for building, unifying, and publishing the **Georgian Tourist Attractions Dataset** to [Hugging Face](https://huggingface.co/datasets/AIAnastasia/Georgian-attractions).

---

## Overview

This repository contains a clean and reproducible pipeline for:
- merging bilingual (English & Russian) attraction datasets,  
- attaching corresponding images,  
- exporting unified data in multiple formats (CSV, JSONL, Metadata),  
- preparing data for Hugging Face Hub upload.

---

## Project Structure

```plaintext
examples/          # input examples (CSV + demo images)
output_example/    # generated sample output (CSV, JSONL, metadata)
src/               # dataset creation pipeline
| Step | Module             | Description                             |
| ---- | ------------------ | --------------------------------------- |
| 1    | `config.py`        | global paths & Hugging Face settings    |
| 2    | `loader.py`        | reads EN/RU CSV files                   |
| 3    | `merger.py`        | merges and normalizes bilingual data    |
| 4    | `image_handler.py` | attaches and copies images              |
| 5    | `exporter.py`      | saves dataset to CSV, JSONL, metadata   |
| 6    | `uploader.py`      | (optional) push dataset to Hugging Face |
| 7    | `main.py`          | orchestrates the full build             |
```

--- 
```
| File                                  | Description              |
| ------------------------------------- | ------------------------ |
| `examples/example_en.csv`             | English dataset sample   |
| `examples/example_ru.csv`             | Russian dataset sample   |
| `output_example/dataset_sample.jsonl` | unified bilingual output |
| `output_example/metadata_sample.json` | example metadata         |
```

---

### Install dependencies:
```pip install -r requirements.txt```

### Run full pipeline:
```python src/main.py```

--- 
### License
Released under CC BY-NC 4.0 (Attribution–NonCommercial).
For commercial or research collaboration, please contact AIAnastasia.
---
## Hugging Face Dataset

🔗 AIAnastasia/Georgian-attractions





