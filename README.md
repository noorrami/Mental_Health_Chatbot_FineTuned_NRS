# 🌿 Mental Health Support Chatbot – Fine-Tuned with Empathetic Dialogues

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.36+-orange.svg)](https://huggingface.co/docs/transformers/index)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-green.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A fine-tuned conversational agent that responds empathetically to users sharing emotional struggles, stress, or daily life events.**

---

## 📌 Overview

This repository contains a **Jupyter notebook** that fine-tunes a **`distilgpt2`** causal language model on the [**Empathetic Dialogues**](https://github.com/facebookresearch/EmpatheticDialogues) dataset.  
The goal is to build a **safe, supportive chatbot** capable of:

- Listening without judgment.
- Providing gentle, empathetic replies.
- Avoiding harmful or insensitive outputs via a post‑generation safety filter.

The fine‑tuned model is exposed through an interactive **Gradio chat interface** – ready to test in your browser.

---

## ✨ Key Features

- **Fine‑tuned on 8,000 conversation turns** (from 76k available) for fast experimentation.
- **Safe response filtering** – blocks phrases like “kill yourself”, “suicide”, etc.
- **Optimised training** – cosine LR scheduler, warmup steps, FP16 mixed precision.
- **Gradio UI** with example inputs and a clean, soft theme.
- **Full dataset training code is commented out** – easily enable it if you have more compute resources.

---

## 🧠 Model & Dataset

| Component       | Details                                                                 |
|----------------|-------------------------------------------------------------------------|
| **Base Model** | `distilgpt2` (lightweight, 82M parameters)                              |
| **Dataset**    | [Empathetic Dialogues](https://github.com/facebookresearch/EmpatheticDialogues) – 76,668 context‑utterance pairs with emotion labels |
| **Prompt Format** | `Context: ... \nUser: ... \nResponse:`                                |
| **Training Objective** | Causal Language Modeling (next‑token prediction)                     |

---

## 🚀 Quick Start

### 1. Clone the repository (or download the notebook)
```bash
git clone https://github.com/your-username/mental-health-chatbot.git
cd mental-health-chatbot
```

### 2. Install dependencies
```bash
pip install -q transformers accelerate gradio datasets scikit-learn wget torch
```

### 3. Open the notebook in Google Colab (recommended) or locally  
- The notebook automatically downloads the Empathetic Dialogues dataset.  
- It then fine‑tunes `distilgpt2` on a **subset of 8,000 samples** (about 10%).  
- Training on a T4 GPU takes ~30–40 minutes.

### 4. Launch the Gradio interface  
After training, run the final cell to get a public link (or local URL) and start chatting.

> **Note:** If you want to train on the **full dataset** (76k samples), uncomment the relevant lines in the "Data Preparation" cell. This will take several hours.

---

## 📊 Training Details (Subset)

| Hyperparameter      | Value          |
|---------------------|----------------|
| Training samples    | 7,200          |
| Evaluation samples  | 800            |
| Epochs              | 10             |
| Batch size (train)  | 4              |
| Learning rate       | 5e-5           |
| Warmup steps        | 500            |
| Scheduler           | Cosine         |
| Mixed precision     | FP16           |

**Validation loss** after 10 epochs: `2.626` (slight overfitting after epoch 2 – reducing epochs to 3–5 would improve generalisation).

---

## 🛡️ Safety Filter

The function `is_safe_response()` blocks responses containing any of the following (case‑insensitive):
```
["kill yourself", "suicide", "self-harm", "worthless", "end your life", "die"]
```
If detected, a fallback message is returned.

---

## 🖥️ Gradio Interface Example

```python
demo = gr.ChatInterface(
    fn=generate_response,
    title="🌿 Mental Health Support Chatbot (Improved)",
    description="I'm here to listen and offer gentle, empathetic support.",
    examples=[
        ["I feel so lonely today."],
        ["I'm really anxious about my job interview tomorrow."],
        ["I've been feeling sad for weeks."]
    ]
)
demo.launch(share=True)
```

---

## ⚠️ Limitations & Future Improvements

- **Small training subset** – the model may lack generalisation. Increase `total_samples` to 20k–30k for better performance.
- **Overfitting** – validation loss increases after epoch 2. Reduce epochs or increase `weight_decay`.
- **Context length** – currently 256 tokens. Increase to 512 if GPU memory allows.
- **Not a substitute for professional help** – always consult a mental health professional for serious issues.

---

## 📂 Repository Structure

```
.
├── Mental_Health_Chatbot_FineTuned_NRS.ipynb   # main notebook
├── README.md                                   # this file
└── (optional) finetuned-mental-health-bot-improved/   # saved model (not uploaded due to size)
```

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Empathetic Dialogues](https://github.com/facebookresearch/EmpatheticDialogues) by Facebook Research.
- Hugging Face `transformers` team for the amazing library.
- Gradio for making UI deployment effortless.
- Special thanks to DevelopersHub Corporation© for the fantastic training opportunity.

---

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

**Enjoy chatting with your empathetic bot! 🌿**


---
