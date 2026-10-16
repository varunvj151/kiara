import json

with open("notebooks/kiara_colab_trainer.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Run Comprehensive 43-Item 6-Pillar Benchmark (Arithmetic, Python AST, Science, Constraints, Tool Dispatch)\n",
        "!python experiments/eval_comprehensive.py --checkpoint checkpoints/kiara_135m_instruct/checkpoint_final.pt --output experiments/benchmark_colab_post_sft.json\n",
        "!cat experiments/benchmark_colab_post_sft.json | grep -E 'accuracy|passed|total'\n"
    ]
}

# Only append if not already present
already_added = any("eval_comprehensive.py" in "".join(c.get("source", [])) for c in nb["cells"])
if not already_added:
    nb["cells"].append(new_cell)
    with open("notebooks/kiara_colab_trainer.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully added comprehensive evaluation cell to Colab notebook.")
else:
    print("Evaluation cell already present.")
