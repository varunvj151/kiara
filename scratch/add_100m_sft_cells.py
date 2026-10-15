import json
from pathlib import Path

nb_path = Path("notebooks/kiara_colab_trainer.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

md_cell = {
    "cell_type": "markdown",
    "metadata": {"id": "SFT100M_Header"},
    "source": [
        "## 8. Supervised Fine-Tuning on Kiara-Small (100M Parameters)\n",
        "\n",
        "Fine-tuning the **99.5M parameter foundation model** (`Stage 26D.1`: 12 layers, 768 hidden dimension) on our curated instruction dataset for enhanced reasoning, concept differentiation, and coding."
    ]
}

code_train_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "SFT100M_Train"},
    "outputs": [],
    "source": [
        "# Execute Instruction Fine-Tuning on Kiara-Small (100M)\n",
        "!python experiments/finetune_sft.py --base-ckpt checkpoints/stage_26D1/checkpoint_final.pt --output-dir checkpoints/kiara_small_instruct --steps 800 --batch-size 8 --lr 8e-5\n",
        "\n",
        "# Backup 100M instruct checkpoint to Google Drive\n",
        "!mkdir -p /content/drive/MyDrive/kiara_checkpoints/kiara_small_instruct\n",
        "!cp -r checkpoints/kiara_small_instruct/* /content/drive/MyDrive/kiara_checkpoints/kiara_small_instruct/\n",
        "print(\"✅ Kiara-Small (100M) Instruct model trained and safely backed up to Google Drive!\")\n"
    ]
}

code_eval_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"id": "SFT100M_Test"},
    "outputs": [],
    "source": [
        "# Live Interactive Generation Tests with 100M Instruct Model\n",
        "!python scripts/chat.py --checkpoint checkpoints/kiara_small_instruct/checkpoint_final.pt --instruct --prompt \"What is the difference between a list and a tuple in Python?\"\n",
        "!python scripts/chat.py --checkpoint checkpoints/kiara_small_instruct/checkpoint_final.pt --instruct --prompt \"Write a Python function to check if a number is prime.\"\n",
        "!python scripts/chat.py --checkpoint checkpoints/kiara_small_instruct/checkpoint_final.pt --instruct --prompt \"Write a short poem about a robot exploring Mars.\"\n"
    ]
}

nb["cells"].extend([md_cell, code_train_cell, code_eval_cell])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Updated notebooks/kiara_colab_trainer.ipynb successfully!")
