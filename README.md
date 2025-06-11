# LoRA-fine-tuning

# UltraFeedback Training Suite

This repository contains modular, research-focused code for training and evaluating Large Language Models (LLMs) on the [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) dataset. It supports methods like **SFT**, **DPO**, and **ORPO**, with optional integration of **LoRA (PEFT)** adapters.

---

## 📌 About

This repo is designed for **experimentation with alignment objectives** on preference-based datasets. It enables quick prototyping and testing of various training methods on binary comparisons derived from human feedback.

It is not intended to be production-ready but rather a **research scaffold** for controlled experimentation.

---

## ✨ Features

- ✅ Support for SFT, DPO, ORPO training modes
- 🧠 Easy integration with HuggingFace models and tokenizers
- 🪶 PEFT/LoRA fine-tuning support via `peft`
- 📊 Logging predictions and basic evaluation (BLEU, ROUGE, BERTScore)
- 🔍 Lightweight and modular script structure

---

## 🧱 Technology Stack

- `transformers` (HuggingFace)
- `datasets`
- `peft`
- `evaluate`
- `torch` (MPS or CUDA)

---

## 📁 Project Structure

```bash
LoRA-fine.tuning/
├── results
├── src
│   ├── __init__.py
│   ├── data_loading.py
│   ├── evaluation.py
│   ├── model.py
│   ├── plotting_utils.py
│   ├── training.py
│   └── utils.py
├── testing_nb.ipynb
└── ultrafeedback_train_sft.json
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.10+
- PyTorch (with MPS or CUDA)
- Git + virtualenv or conda

### Install dependencies

```bash
pip install -r requirements.txt
```

🚀 Quickstart

Evaluate a model (SFT or fine-tuned)

```bash
python main.py
```

This will:

Load the model (google/gemma-2b or another HF ID)
Run predictions on ultrafeedback_binarized
Save metrics and predictions to outputs/
Modify settings in settings.py
default_model_id = "google/gemma-2b"
use_peft = True
peft_settings = {
"r": 8,
"lora_alpha": 16,
...
}
🧪 Training Modes (Coming/Available)

⚠️ SFT (train_sft.py)
⚠️ DPO (train_dpo.py) – experimental
⚠️ ORPO (train_orpo.py) – in progress
To train with LoRA, pass use_peft = True and customize your LoRA config in settings.py.
📊 Evaluation

Run:

```bash
python main.py
```

Generates:

```bash
outputs/predictions.txt
outputs/metrics.json with BLEU, ROUGE, and BERTScore
```

🧠 Dataset Info

We use:

HuggingFaceH4/ultrafeedback_binarized
Structure includes:

prompt, chosen, and rejected texts
Use-case: preference modeling, reward modeling, DPO training
🤝 Contributing

This is a research-oriented repo. Feel free to open issues or pull requests to add new training methods, logging utilities, or evaluation functions.

📄 License

MIT License. See LICENSE for more details.

📬 Contact
For questions, collaborations, or feedback, feel free to reach out:

📧 Email: fermaat.vl@gmail.com
🧑‍💻 GitHub: [@fermaat](https://github.com/fermaat)
🌐 [Website](https://fermaat.github.io)
