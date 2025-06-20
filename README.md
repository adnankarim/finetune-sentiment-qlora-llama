﻿
# 🦙 QLoRA + Linear-Head Sentiment Classifier

This project fine-tunes a **4-bit quantized LLaMA 3 model** using **QLoRA** and a **linear classifier head** for sentiment analysis of tweets. It uses Unsloth's fast LLaMA loader, HuggingFace Transformers, and TRL's `SFTTrainer`.

---

##  Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Minimal `requirements.txt`:

```txt
transformers
datasets
scikit-learn
pandas
unsloth
trl
```

You also need:

* A CUDA-capable GPU
* PyTorch with CUDA (e.g., `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`)

---

##  Folder Structure

```
.
├── finetune_llama_sentiment_classifier.py     # Main script
├── data/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── llama-finetune.log                # Generated during training
└── llama3_lora/                      # Output model dir
```

---

##  Input Format

Each CSV must have either:

* A `tweet` or `text` column (string)
* And a label column:

  * `sentiment`: one of `"negative"`, `"neutral"`, `"positive"`
  * OR `target`: one of `0` (negative), `2` (neutral), `4` (positive)

---

##  Running the Script

### Basic usage

```bash
python finetune_llama_sentiment_classifier.py --load_4bit
```

This will:

* Load 4-bit `Meta-Llama-3.1-8B-Instruct` model
* Train for `0.1` epochs
* Save fine-tuned model + tokenizer to `llama3_lora/`
* Evaluate on `valid.csv` and `test.csv`

### Optional arguments

| Argument         | Default                              | Description                            |
| ---------------- | ------------------------------------ | -------------------------------------- |
| `--train_csv`    | `data/train.csv`                     | Training dataset                       |
| `--valid_csv`    | `data/valid.csv`                     | Validation dataset                     |
| `--test_csv`     | `data/test.csv`                      | Test dataset                           |
| `--model_name`   | `unsloth/Meta-Llama-3.1-8B-Instruct` | HF model to load                       |
| `--epochs`       | `0.1`                                | Fine-tuning epochs                     |
| `--batch`        | `8`                                  | Per-device batch size                  |
| `--grad_acc`     | `4`                                  | Gradient accumulation steps            |
| `--lr`           | `5e-5`                               | Learning rate                          |
| `--max_len`      | `512`                                | Max input length                       |
| `--lora_r`       | `16`                                 | LoRA rank                              |
| `--lora_alpha`   | `32`                                 | LoRA alpha                             |
| `--lora_dropout` | `0.05`                               | LoRA dropout                           |
| `--out_dir`      | `llama3_lora`                        | Output directory for model + tokenizer |
| `--load_4bit`    | `False` (flag)                       | Load the model in 4-bit QLoRA mode     |

---

##  Output

* Logs training and evaluation to `llama-finetune.log`
* Saves:

  * `llama3_lora/llama3_qora_classifier.pt`
  * `llama3_lora/tokenizer_config.json` and tokenizer files

Evaluation prints:

* Macro-F1 score
* Per-class precision/recall
* Confusion matrix

---

##  Notes

* This approach is ideal for **low-resource fine-tuning** with **small labeled datasets**.
* You can integrate this pipeline with a small inference script using `model.load_state_dict(...)` and tokenizer from saved folder.
