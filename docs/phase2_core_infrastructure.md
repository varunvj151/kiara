# Phase 2 — Core Infrastructure

## Objective

Build the infrastructure required to train a frontier language model before implementing the model itself. This phase focuses on creating the supporting systems that make training reliable, reproducible, and scalable.

## Phase 2 Overview

Development Environment

↓

Configuration System

↓

Logging

↓

Experiment Tracking

↓

Dataset Pipeline

↓

Tokenizer

↓

Vocabulary

↓

Training Dataset

↓

Model Development

## Step 15 — Development Environment

The project should be set up for reproducibility from the beginning.

### Deliverables

- Python version
- dependency management
- project structure
- virtual environment
- code formatting
- linting
- testing
- pre-commit hooks

### Recommended Stack

- Python 3.12+
- uv or Poetry
- Git
- pytest
- Ruff
- Black
- MyPy
- pre-commit

## Step 16 — Configuration System

Large models require a flexible configuration system. Instead of changing Python code for each experiment, the system should load settings from configuration files.

### Suggested Configuration Files

- configs/base.yaml
- configs/tiny.yaml
- configs/small.yaml
- configs/base_model.yaml
- configs/training.yaml
- configs/tokenizer.yaml
- configs/inference.yaml

## Step 17 — Logging System

Training runs can last hours or days, so logging must be designed early. Logs should include:

- loss
- learning rate
- memory usage
- throughput
- checkpoints
- validation scores
- errors
- runtime

## Step 18 — Experiment Tracking

Every experiment should be reproducible and comparable.

### Required Metadata

- model configuration
- dataset version
- Git commit hash
- hyperparameters
- evaluation metrics
- checkpoint paths

## Step 19 — Dataset Pipeline Design

The dataset pipeline should be modular and explicit.

Raw Data

↓

Language Detection

↓

Cleaning

↓

Deduplication

↓

Quality Filtering

↓

PII Removal

↓

Formatting

↓

Chunking

↓

Tokenization

↓

Binary Dataset

↓

Training

## Step 20 — Dataset Strategy

### Sources

- Web
- Books
- Code
- Wikipedia
- Academic papers
- Documentation
- Mathematics
- Conversations

### Quality Controls

- allowed languages
- minimum document length
- maximum document length
- duplicate thresholds
- toxicity filtering
- license compliance

## Step 21 — Tokenizer Research

Before implementation, the tokenizer design should be investigated carefully.

### Tokenization Approaches to Review

- character tokenization
- word tokenization
- BPE
- WordPiece
- SentencePiece
- Unigram

### Key Question

Why do nearly all modern LLMs use subword tokenization?

The answer should guide the final tokenizer choice.

## Step 22 — Build the Tokenizer

The tokenizer is the first real code component of Kiara.

### Goals

- train a tokenizer on a small corpus
- inspect vocabulary
- encode text
- decode text
- measure compression

## Step 23 — Dataset Builder

The dataset builder should transform raw documents into efficient training files.

### Responsibilities

- read raw text
- split into documents
- tokenize
- chunk into fixed-length sequences
- save in a binary format for fast loading

## Step 24 — Validation

Before building neural networks, the infrastructure should be validated.

### Validation Checks

- tokenizer correctness
- dataset integrity
- vocabulary statistics
- token frequency distribution
- sequence length distribution
- training sample generation

## Step 25 — Architecture Freeze

Once the tokenizer and data pipeline are complete, the project should freeze key design choices.

### Freeze Points

- tokenizer version
- vocabulary
- model specification

Any later change to these items would require retraining from scratch.
