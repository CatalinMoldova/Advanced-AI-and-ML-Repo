# LoRA Fine-Tuning of SmolLM-135M on Greentexts

Supervised fine-tuning of [`HuggingFaceTB/SmolLM-135M`](https://huggingface.co/HuggingFaceTB/SmolLM-135M) using Low-Rank Adaptation (LoRA), with a rank ablation and perplexity-based evaluation. Built as Homework 5 for **Advanced AI and Machine Learning** (Prof. Keith Ross, NYU).

**Author:** Catalin Botezat

---

## Overview

LoRA (Low-Rank Adaptation) freezes a pre-trained model's weights and only trains small low-rank matrices *A* and *B* injected into selected linear layers, such that the effective weight becomes *W ≈ W₀ + BA*. This makes fine-tuning a 135M-parameter LLM feasible on a free Colab T4.

This project:

1. Fine-tunes SmolLM-135M on the [`maxmyn/wholesome_greentext_110k`](https://huggingface.co/datasets/maxmyn/wholesome_greentext_110k) dataset (first 100 rows).
2. Evaluates the model using **perplexity** across training checkpoints.
3. Generates text before and after fine-tuning on a fixed set of prompts.
4. Runs an **ablation** on LoRA rank (*r* = 16 vs *r* = 4).
5. Logs training loss to **Weights & Biases**.

---

## Repository Contents

| File | Description |
|---|---|
| `LoRA_SFT_Assignment_final.ipynb` | Main Colab notebook — data loading, LoRA setup, SFT training, evaluation, ablation. |
| `HW5_LoRA_Report_Botezat.pdf` | Final PDF report with all results and discussion. |
| `HW5_LoRA_Report_Botezat.tex` | LaTeX source for the report. |
| `wandb_loss.png` | Training loss screenshot from Weights & Biases. |
| `README.md` | This file. |

---

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (tested on Colab T4 with fp16)
- A [Weights & Biases](https://wandb.ai) account (free tier is sufficient)

### Installation

```bash
pip install transformers peft datasets trl wandb accelerate
```

### Authentication

```python
import wandb
wandb.login()  # paste API key from https://wandb.ai/authorize
```

---

## How to Run

The notebook is designed to run end-to-end in Google Colab. Execute the cells in order:

1. **Install & import** dependencies.
2. **Authenticate** with W&B.
3. **Load** SmolLM-135M, tokenizer, and the greentext dataset (first 100 rows).
4. **Generate baseline outputs** from the un-tuned model for four fixed prompts.
5. **Configure LoRA** and wrap the model with `get_peft_model`.
6. **Train** with `SFTTrainer` for 500 steps; checkpoints saved every 100 steps to Google Drive.
7. **Evaluate** perplexity at each checkpoint.
8. **Generate post-training outputs** for the same prompts.
9. **Ablation** — repeat with *r* = 4 and compare.

Checkpoints are saved to `/content/drive/MyDrive/AI_ML_HW5_Checkpoints/` — adjust this path if you're not running in Colab.

---

## Configuration

Main training run:

| Hyperparameter | Value |
|---|---|
| Base model | `HuggingFaceTB/SmolLM-135M` |
| Target modules | `q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj` |
| LoRA rank *r* | 16 (main) / 4 (ablation) |
| LoRA α | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 2e-4 |
| Batch size | 4 |
| Max steps | 500 |
| Precision | fp16 |
| Inference sampling | *T* = 0.7, top-*p* = 0.9 |

---

## Results Summary

### Training loss

Loss drops from ~3.8 to ~0.2 across 500 steps, with a steep descent in the first 100 steps (format acquisition) and a plateau from ~300 onward.

### Perplexity on training set

| Step | *r* = 16 | *r* = 4 |
|---:|---:|---:|
| 0 | 29.96 | 29.95 |
| 100 | 4.68 | 4.37 |
| 200 | 1.49 | 2.55 |
| 300 | 1.25 | 1.44 |
| 400 | 1.19 | 1.29 |
| 500 | 1.28 | 1.29 |

Both configurations converge to essentially the same final perplexity. *r* = 16 fits the data faster; *r* = 4 is slower but reaches a comparable endpoint.

### Generation quality

After fine-tuning, the model consistently produces the greentext surface format (`>`-prefixed imperative clauses) across all prompts. Semantic coherence improves over the base model but remains shallow, and **repetition is the dominant failure mode** — a direct consequence of training on only 100 examples for ~20 effective epochs.

See the PDF report for full prompt-by-prompt before/after comparisons.

---

## Key Caveats

- **Evaluation is on the training set.** The notebook passes `eval_dataset=train_dataset`, so reported perplexities reflect fit, not generalization. A proper held-out split would give noticeably higher numbers.
- **Dataset size is tiny.** 100 examples × 500 steps ÷ batch 4 ≈ 20 passes per example. Low loss here is evidence of memorization, not language modeling quality.
- **Ablation is confounded.** α was fixed at 32 while *r* was halved, so the effective update magnitude per LoRA direction was not held constant. A cleaner ablation would scale α with *r* (e.g. *r* = 4 with α = 8).

---

## What I'd Do Differently

- Add a proper train/validation split and report held-out perplexity.
- Scale the dataset to 10³–10⁴ examples to move past the memorization regime.
- Sweep (*r*, α) pairs that keep α/*r* constant.
- Add quantitative generation metrics (distinct-*n* for repetition, BLEU/ROUGE for fluency) on a larger, disjoint prompt set.
- Try `repetition_penalty` at inference to mitigate the repetition without retraining (symptomatic fix, not a cure).

---

## References

- [Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT documentation](https://huggingface.co/docs/peft)
- [TRL `SFTTrainer` documentation](https://huggingface.co/docs/trl/sft_trainer)
- [SmolLM-135M model card](https://huggingface.co/HuggingFaceTB/SmolLM-135M)

---

## License

Academic coursework. Code and report are provided as-is for reference.
