import json
from pathlib import Path

nb_path = Path("notebooks/kiara_colab_trainer.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

md_cell = {
    "cell_type": "markdown",
    "metadata": {"id": "Gold_Header"},
    "source": [
        "## 9. 🚀 Industrial Scale: Open-Source Gold Training (50,000+ Samples)\n",
        "\n",
        "Train Kiara on the **world's highest-grade open-source datasets** used by frontier labs:\n",
        "- **OpenHermes 2.5** (15,000 deep reasoning samples)\n",
        "- **Orca Math 200k** (12,000 step-by-step arithmetic & algebra samples)\n",
        "- **CodeAlpaca 20k** (10,000 algorithmic code implementations)\n",
        "- **UltraChat 200k** (10,000 natural multi-turn conversations)\n",
        "- Accelerated with **PyTorch 2.0 FlashAttention / SDPA**!"
    ]
}

ingest_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "Gold_Ingest"},
    "outputs": [],
    "source": [
        "# Install datasets and stream 50,000+ open-source gold samples\n",
        "!pip install -q datasets\n",
        "!python data/instructions/acquire_open_source_gold.py\n"
    ]
}

train_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "Gold_Train"},
    "outputs": [],
    "source": [
        "# Execute 3,000-step high-throughput training with FlashAttention\n",
        "!python experiments/finetune_sft.py --steps 3000 --batch-size 16 --lr 1.5e-4 --seq-len 256\n",
        "\n",
        "# Backup trained model to Google Drive\n",
        "!mkdir -p /content/drive/MyDrive/kiara_checkpoints/kiara_instruct_gold/\n",
        "!cp -r checkpoints/kiara_instruct/* /content/drive/MyDrive/kiara_checkpoints/kiara_instruct_gold/\n",
        "print(\"✅ Kiara Open-Source Gold model trained and safely backed up to Google Drive!\")\n"
    ]
}

test_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "Gold_Test"},
    "outputs": [],
    "source": [
        "# Test Gold Model on complex multi-domain reasoning\n",
        "!python scripts/chat.py --checkpoint checkpoints/kiara_instruct/checkpoint_final.pt --instruct --prompt \"Solve this step by step: A store sells apples for $2 each and oranges for $3 each. If Sarah buys 4 apples and 5 oranges, how much does she spend?\"\n",
        "!python scripts/chat.py --checkpoint checkpoints/kiara_instruct/checkpoint_final.pt --instruct --prompt \"Write a complete Python implementation of a LRU Cache using collections.OrderedDict.\"\n",
        "!python scripts/chat.py --checkpoint checkpoints/kiara_instruct/checkpoint_final.pt --instruct --prompt \"Explain quantum entanglement in simple terms for a high school student.\"\n"
    ]
}

nb["cells"].extend([md_cell, ingest_cell, train_cell, test_cell])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Added Industrial Open-Source Gold cells to kiara_colab_trainer.ipynb successfully!")
