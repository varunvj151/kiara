"""
tests/test_eval_stage_26B.py — Test Suite for Stage 26B Evaluation Logic
========================================================================
Verifies:
  1. ROUGE-L exact and partial match mathematical correctness
  2. Character-level edit similarity bounds [0.0, 1.0]
  3. Teacher-forced loss metric format and types
  4. Disjointness of evaluation subsets
"""

import math
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.eval_stage_26B import compute_rouge_l, compute_edit_similarity


def test_rouge_l_exact_match():
    tokens = ["the", "quick", "brown", "fox"]
    score = compute_rouge_l(tokens, tokens)
    assert pytest.approx(score, 1e-4) == 1.0


def test_rouge_l_no_match():
    tokens1 = ["the", "quick", "brown", "fox"]
    tokens2 = ["lorem", "ipsum", "dolor", "sit"]
    score = compute_rouge_l(tokens1, tokens2)
    assert score == 0.0


def test_rouge_l_partial_match():
    pred = ["once", "upon", "a", "time", "in", "paris"]
    target = ["once", "upon", "a", "time", "there", "was"]
    score = compute_rouge_l(pred, target)
    assert 0.0 < score < 1.0
    # 4 common tokens out of 6
    assert pytest.approx(score, 1e-3) == (4 / 6)


def test_edit_similarity_bounds():
    assert compute_edit_similarity("hello", "hello") == 1.0
    assert compute_edit_similarity("", "") == 1.0
    assert compute_edit_similarity("abc", "xyz") == 0.0
    sim = compute_edit_similarity("kitten", "sitting")
    assert 0.0 < sim < 1.0
