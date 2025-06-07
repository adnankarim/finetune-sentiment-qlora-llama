#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infer.py

Load your QLoRA‐finetuned Llama‐3 + linear head, then run inference on
valid.csv and test.csv.  Compute macro-F1/accuracy, and write `pred` to test.csv.
"""

import argparse
import pathlib
import logging
import sys
import json
import re
import unicodedata

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput

# ─────────────────────────────────────────────────────────────────────────────
# logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("predict_finetuned.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# text cleaning
# ─────────────────────────────────────────────────────────────────────────────
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")

def clean_text(txt: str) -> str:
    """
    Lowercase, normalize, replace URLs/mentions, squash whitespace.
    """
    t = unicodedata.normalize("NFKD", str(txt)).lower()
    t = URL_RE.sub("<URL>", t)
    t = MENTION_RE.sub("<USER>", t)
    return WS_RE.sub(" ", t).strip() or "<EMPTY>"

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ─────────────────────────────────────────────────────────────────────────────
# helper: load a CSV into a DataFrame, return DataFrame with columns ["text","label"]
# ─────────────────────────────────────────────────────────────────────────────
def load_dataframe(path: pathlib.Path) -> pd.DataFrame:
    """
    Reads a CSV, detects text column ("tweet" or "text"), and label column
    (either "sentiment" mapping to {"negative","neutral","positive"} or numeric
    "label"/"target").  Returns a DataFrame["text","label"] where:
      - text is cleaned by clean_text()
      - label is integer 0/1/2
    """
    df = pd.read_csv(path)
    # detect text column
    text_col = next(c for c in ("tweet", "text") if c in df.columns)
    # detect label column
    if "sentiment" in df.columns:
        df["label"] = df["sentiment"].str.lower().map(LABEL2ID)
    else:
        lbl_col = next(c for c in ("label", "target") if c in df.columns)
        # if target is {0,2,4}, map to {0,1,2}
        if df[lbl_col].isin([0,2,4]).all():
            df["label"] = df[lbl_col].replace({0:0, 2:1, 4:2}).astype(int)
        else:
            df["label"] = df[lbl_col].astype(int)
    df["text"] = df[text_col].astype(str).apply(clean_text)
    return df[["text", "label"]]


# ─────────────────────────────────────────────────────────────────────────────
# LlamaClassifier wrapper (same as in training)
# ─────────────────────────────────────────────────────────────────────────────
class LlamaClassifier(torch.nn.Module):
    """
    • Base = LoRA‐patched Llama.  
    • Pools the final hidden state at the last non‐pad token for each sequence.
    • Linear head → 3 logits.
    • If labels given, returns cross‐entropy loss; otherwise just logits.
    • generate(…) is forwarded to base.generate(...).
    """
    def __init__(self, base, hidden_size: int, n_classes: int = 3):
        super().__init__()
        self.base = base
        self.head = torch.nn.Linear(hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # run base model to get hidden states
        out = self.base.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_h = out.hidden_states[-1]                # (B, L, H)
        idx = attention_mask.sum(dim=1) - 1           # (B,)
        pooled = last_h[torch.arange(last_h.size(0)), idx]  # (B, H)
        logits = self.head(pooled)                    # (B, 3)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)

    # Unsloth requires these embedding helpers
    def get_input_embeddings(self):
        return self.base.get_input_embeddings()
    def set_input_embeddings(self, x):
        return self.base.set_input_embeddings(x)
    def get_output_embeddings(self):
        return self.base.get_output_embeddings()
    def set_output_embeddings(self, x):
        return self.base.set_output_embeddings(x)

    # forward generate → base.generate
    def generate(self, *args, **kwargs):
        return self.base.generate(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Given a list of raw texts, produce the "chat strings" exactly as finetuned, then
# batch‐tokenize and return tensors for inference.
# ─────────────────────────────────────────────────────────────────────────────
def texts_to_tensors(
    raw_texts: list[str],
    tok,
    device: torch.device,
    max_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    raw_texts: list of cleaned strings (no "Classify the tweet..." prefix yet).
    Returns input_ids, attention_mask for the entire list in chunks of <batch_size>.
    Here we do a one‐shot batch tokenization; if memory is tight, you can chunk.
    """
    # Build chat‐prompt for each input:
    chat_prompts = [
        tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "Classify the tweet sentiment (negative / neutral / positive):\n\n"
                               f"{txt}"
                },
                {"role": "assistant", "content": ""}  # generation prompt
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        for txt in raw_texts
    ]

    # Tokenize all at once (or you can chunk in bigger datasets)
    enc = tok(
        chat_prompts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    return input_ids, attention_mask


# ─────────────────────────────────────────────────────────────────────────────
# Run inference, return numpy array of predicted IDs (0/1/2)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(
    model: LlamaClassifier,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int
) -> np.ndarray:
    """
    Splits input_ids/attention_mask into batches, runs model, returns preds.
    """
    model.eval()
    n = input_ids.size(0)
    preds = []
    for i in range(0, n, batch_size):
        batch_ids = input_ids[i : i + batch_size]
        batch_mask = attention_mask[i : i + batch_size]
        outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
        batch_logits = outputs.logits  # (batch_size, 3)
        batch_preds = torch.argmax(batch_logits, dim=-1).cpu().numpy()
        preds.append(batch_preds)
    return np.concatenate(preds, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate on a DataFrame["text","label"], return (pred_ids, gold_ids, macro_f1, acc)
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_dataframe(
    df: pd.DataFrame,
    model: LlamaClassifier,
    tok,
    device: torch.device,
    batch_size: int,
    max_len: int,
    split_name: str
):
    """
    df: DataFrame with columns ["text","label"] (cleaned).
    Runs inference, computes and logs macro‐F1 + accuracy + classification report.
    Returns (pred_ids, gold_ids).
    """
    raw_texts = df["text"].tolist()
    gold = df["label"].to_numpy()

    # build input tensors
    input_ids, attention_mask = texts_to_tensors(raw_texts, tok, device, max_len)

    # run inference
    log.info(f"▶ Running inference on {split_name} (n={len(df)}) …")
    pred_ids = run_inference(model, input_ids, attention_mask, batch_size)

    # metrics
    macro_f1 = f1_score(gold, pred_ids, average="macro")
    acc = accuracy_score(gold, pred_ids)
    log.info(f"► {split_name} Macro-F1 = {macro_f1:.4f}, Acc = {acc:.4f}")

    if split_name.lower() == "test":
        log.info("\n" + classification_report(gold, pred_ids,
                                              target_names=["negative", "neutral", "positive"]))
        log.info("Confusion matrix:\n" + str(confusion_matrix(gold, pred_ids)))

    return pred_ids, gold, macro_f1, acc


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # 1) Load model + tokenizer + head
    log.info("▶ Loading fine-tuned model + tokenizer …")
    base, tok = FastLanguageModel.from_pretrained(
        args.model_name_or_path,
        max_seq_length=args.max_len,
        dtype=torch.float16,
        load_in_4bit=(not args.no_4bit),
    )
    tok = get_chat_template(tok, chat_template="llama-3.2")

    # wrap in LlamaClassifier and load head
    model = LlamaClassifier(base, base.model.config.hidden_size, n_classes=3).to(device)

    # load only the head weights
    head_path = pathlib.Path("llama3_finetuned1b/lora_head.pt")
    state_dict = torch.load(head_path, map_location="cpu")
    # if you saved the entire classifier state_dict, you may need to filter out base.*
    # but in our training we saved only `model.head.state_dict()`
    model.head.load_state_dict(state_dict)
    # cast head to FP16 (because base is in FP16)
    model.head = model.head.to(device).half()
    model.eval()

    # 2) Load and clean valid/test
    log.info("▶ Loading validation set …")
    valid_df = load_dataframe(pathlib.Path(args.valid_csv))
    log.info("▶ Loading test set …")
    test_df_raw = pd.read_csv(args.test_csv)  # keep raw for writing later
    test_df = load_dataframe(pathlib.Path(args.test_csv))

    # 3) Run inference + metrics on valid
    valid_preds, valid_gold, vf1, vacc = evaluate_dataframe(
        valid_df, model, tok, device, args.batch_size, args.max_len, split_name="Valid"
    )

    # 4) Run inference + metrics on test
    test_preds, test_gold, tf1, tacc = evaluate_dataframe(
        test_df, model, tok, device, args.batch_size, args.max_len, split_name="Test"
    )

    # 5) Write predictions into original test CSV
    log.info("▶ Writing `pred` column into test CSV …")
    test_df_raw["pred"] = test_preds.tolist()
    pd.DataFrame(test_df_raw).to_csv(args.test_csv, index=False)
    log.info(f"✓ Updated {args.test_csv} with predictions.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on valid/test using a fine-tuned Llama-3 QLoRA + linear head."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
         default= "unsloth/Meta-Llama-3.1-8B-Instruct",
        help=(
            "Directory where you saved your QLoRA-adapter model and lora_classifier.pt. "
            "This should contain adapter files plus head checkpoint."
        )
    )
    parser.add_argument(
        "--head_name",
        type=str,
        default="lora_classifier.pt",
        help="Name of the linear head checkpoint inside model directory (e.g. lora_classifier.pt)."
    )
    parser.add_argument(
        "--valid_csv",
        type=str,
        default="data/valid.csv",
        help="Path to your validation CSV."
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/test.csv",
        help="Path to your test CSV (will be overwritten with `pred` column)."
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference."
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        default=True,
        help="If set, do NOT load in 4-bit (loads FP16 instead)."
    )
    args = parser.parse_args()
    main(args)
