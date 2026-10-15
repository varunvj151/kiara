# Implementation Plan: Step 22 — Dataset → Tokenizer → Training Tensor Pipeline

Establishes the authoritative data preprocessing, sequence packing/chunking, batch collation, and PyTorch `DataLoader` pipeline for the Kiara foundation model using the frozen **32,768 Byte-Level BPE Tokenizer**.

## Proposed Architecture

```
Corpus v2.0 / Raw Text
        │
        ▼
   DataLoader / Dataset Streamer
        │
        ▼
   KiaraTokenizer (Frozen 32K Byte-BPE)
        │
        ├── EOS delimiter insertion
        ├── input_ids
        └── attention_mask
        │
        ▼
   PackedDataset / ChunkedDataset
   (Packs continuous token streams into fixed seq_len blocks)
        │
        ▼
   DataCollatorForCausalLM
   (Handles [B, T] tensor conversion, padding masks, and label alignment)
        │
        ▼
   PyTorch DataLoader (Deterministic, Seeded, Worker-Ready)
        │
        ▼
   Training Batch Contract:
   {
       "input_ids":      torch.Tensor[B, T] (dtype=torch.long),
       "attention_mask": torch.Tensor[B, T] (dtype=torch.long),
       "labels":         torch.Tensor[B, T] (dtype=torch.long, with -100 for ignored/padded tokens),
   }
```

## Proposed Changes

### `kiara/data/`

#### [NEW] [`special_constants.py`](file:///c:/Users/varun/OneDrive/Documents/flutter1/kiara/kiara/data/constants.py)
- Defines `IGNORE_INDEX = -100` (PyTorch CrossEntropyLoss default ignore index).

#### [NEW] [`dataset.py`](file:///c:/Users/varun/OneDrive/Documents/flutter1/kiara/kiara/data/dataset.py)
- **`TokenizedDataset`**: In-memory / memory-efficient list of tokenized documents.
- **`PackedDataset`**: Concatenates tokenized documents separated by `[EOS]` and chunks them into uniform blocks of size `seq_len` without wasted padding.
- **`PaddedDataset`**: Handles variable-length documents with sequence-level padding.
- Checkpoint/resume state support (`state_dict()`, `load_state_dict()`).

#### [NEW] [`collator.py`](file:///c:/Users/varun/OneDrive/Documents/flutter1/kiara/kiara/data/collator.py)
- **`DataCollatorForCausalLM`**: Collates batches into `input_ids`, `attention_mask`, and `labels`.
  - For packed sequences: `labels` matches `input_ids` (causal shift handled in model loss).
  - For padded sequences: pads to batch max / multiple of 8, sets `labels` at pad positions to `-100`.

#### [NEW] [`pipeline.py`](file:///c:/Users/varun/OneDrive/Documents/flutter1/kiara/kiara/data/pipeline.py)
- **`create_training_dataloader`** & **`create_eval_dataloader`**:
  - Deterministic PyTorch `DataLoader` factory with worker seeding.
  - Integration with Corpus v2.0 generator and evaluation splits (strictly disjoint).

#### [NEW] [`__init__.py`](file:///c:/Users/varun/OneDrive/Documents/flutter1/kiara/kiara/data/__init__.py)
- Public exports for `kiara.data`.

### Tests

#### [NEW] [`tests/test_data_pipeline.py`](file:///c:/Users/varun/OneDrive/Documents/flutter1/kiara/tests/test_data_pipeline.py)
- Test deterministic tokenization & sequence packing
- Test batch tensor shapes `[B, T]` and dtype `torch.long`
- Test pad masking in `labels` with `-100`
- Test DataLoader iteration, batch shuffling, and determinism with random seed
- Test state checkpointing & resumption
- Test evaluation data disjointness (zero leakage)
- Test zero UNK tokens across entire training pipeline

## Verification Plan

### Automated Tests
```powershell
python -m pytest tests/ -v
```
All unit and integration tests across `tests/test_data_pipeline.py`, `tests/test_tokenizer.py`, `tests/test_tokenizer_benchmark.py`, and `tests/test_package.py` must pass.
