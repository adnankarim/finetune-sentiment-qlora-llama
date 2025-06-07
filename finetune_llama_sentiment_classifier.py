
# -------------------------------------------------------------
# QLoRA + Linear-Head Sentiment Classifier

#  * Works with 4-bit QLoRA, LoRA-adapters and a tiny linear head
# -------------------------------------------------------------
import argparse, logging, pathlib, re, string, sys, unicodedata

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import (
    DataCollatorForSeq2Seq,
    TrainingArguments,
    modeling_outputs as mo,
)

# ─────────────────── logging ───────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("llama-finetune.log", "w", "utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────── Unsloth / TRL ───────────────────
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer


URL, MENTION = re.compile(r"https?://\S+|www\.\S+"), re.compile(r"@\w+")
HASHTAG, MULTI = re.compile(r"#(\w+)"), re.compile(r"\s+")


def clean(txt: str) -> str:
    t = unicodedata.normalize("NFKD", str(txt)).lower()
    t = URL.sub("<URL>", t)
    t = MENTION.sub("<USER>", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = HASHTAG.sub(r"\1", t)
    t = MULTI.sub(" ", t).strip()
    return t or "<EMPTY>"


def csv_to_chat(
    path: pathlib.Path,
    tokenizer,
    split: str = "train",  # "train" | "valid" | "test"
) -> Dataset:
    """
    Returns a HF Dataset with fields:
      • chat_text   (string)  – prompt fed to the model
      • label_str   (int 0/1/2) – gold sentiment (still needed for loss/metrics)
    For *train* rows the assistant turn contains the gold label.
    For *valid* and *test* rows the assistant turn is empty → no leakage.
    """
    df = pd.read_csv(path)
    text_col = next(c for c in ("tweet", "text") if c in df.columns)

    # Map label to 0/1/2
    if "sentiment" in df.columns:
        df["label_str"] = df["sentiment"].str.lower().map(
            {"negative": 0, "neutral": 1, "positive": 2}
        )
    else:  # Kaggle format with 0/2/4
        df["label_str"] = df["target"].map({0: 0, 2: 1, 4: 2})

    df["tweet"] = df[text_col].astype(str).apply(clean)

    def build_conv(row):
        convo = messages = [
                {
                    "role": "user",
                    "content": (
                        "Classify this tweet’s sentiment "
                        "(positive / neutral / negative):\n\n"
                        f"{row.tweet}"
                    ),
                }
            ]
        print(row.tweet)
        if split == "train":
            print(row.label_str)
            messages.append({
                "role": "assistant",
                "content": ["negative", "neutral", "positive"][row.label_str],
            })
        return tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )

    df["chat_text"] = df.apply(build_conv, axis=1)
    return Dataset.from_pandas(df[["chat_text", "label_str"]], preserve_index=False)



class LastTokenLossCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors="pt")
        if "labels" in batch and batch["labels"] is not None:
            for i in range(len(batch["labels"])):
                mask = batch["labels"][i] != self.tokenizer.pad_token_id
                if mask.any():
                    last = mask.nonzero()[-1].item()
                    batch["labels"][i, :last] = self.tokenizer.pad_token_id
        return batch


# ════════════════════════════════════════════════════
#            WRAPPER  (LLM + tiny classifier)
# ════════════════════════════════════════════════════
class LlamaWithClassifier(torch.nn.Module):
    def __init__(self, base, hidden: int, n_classes: int = 3):
        super().__init__()
        self.base = base
        self.classifier = torch.nn.Linear(hidden, n_classes)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        label_str=None,
    ) -> mo.SequenceClassifierOutput:
        outs = self.base.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_h = outs.hidden_states[-1]  # (B,L,H)
        idx = attention_mask.sum(1) - 1  # last non-PAD
        idx = idx.view(-1, 1, 1).expand(-1, 1, last_h.size(2))
        pooled = last_h.gather(1, idx).squeeze(1)  # (B,H)
        logits = self.classifier(pooled)

        loss = None
        if label_str is not None:
            loss = torch.nn.functional.cross_entropy(logits, label_str)
        return mo.SequenceClassifierOutput(loss=loss, logits=logits)

    # forward embedding setters (needed by PEFT / Unsloth tricks)
    def get_input_embeddings(self):
        return self.base.get_input_embeddings()

    def set_input_embeddings(self, new_emb):
        self.base.set_input_embeddings(new_emb)

    def get_output_embeddings(self):
        return self.base.get_output_embeddings()

    def set_output_embeddings(self, new_emb):
        self.base.set_output_embeddings(new_emb)



def main(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("▶ Device: %s", device.upper())

    # Fast template model (just for formatting)
    _, tmp_tok = FastLanguageModel.from_pretrained(
        a.model_name,
        max_seq_length=a.max_len,
        dtype=torch.float16,
        load_in_4bit=a.load_4bit,
    )
    tmp_tok = get_chat_template(tmp_tok, "llama-3.2")

    log.info("▶ Loading CSVs …")
    train_ds = csv_to_chat(a.train_csv, tmp_tok, split="train")
    valid_ds = csv_to_chat(a.valid_csv, tmp_tok, split="valid")
    test_ds = csv_to_chat(a.test_csv, tmp_tok, split="test")

    # Real backbone
    log.info("▶ Loading backbone …")
    base, tok = FastLanguageModel.from_pretrained(
        a.model_name,
        max_seq_length=a.max_len,
        dtype=torch.float16,
        load_in_4bit=a.load_4bit,
    )
    tok = get_chat_template(tok, "llama-3.2")
    base = base.to(device)

    log.info("▶ Attaching LoRA adapters …")
    base = FastLanguageModel.get_peft_model(
        base,
        r=a.lora_r,
        lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = LlamaWithClassifier(
        base, hidden=base.model.config.hidden_size, n_classes=3
    ).to(device)
    model.config = base.model.config  # keeps dtype info for Trainer

    log.info("▶ Tokenising …")
    def tok_map(batch):
        enc = tok(
            batch["chat_text"],
            padding="max_length",
            truncation=True,
            max_length=a.max_len,
        )
        enc["label_str"] = batch["label_str"]
        return enc

    cols = ["chat_text", "label_str"]
    train_ds = train_ds.map(tok_map, batched=True, remove_columns=cols)
    valid_ds = valid_ds.map(tok_map, batched=True, remove_columns=cols)
    test_ds = test_ds.map(tok_map, batched=True, remove_columns=cols)
    

    for ds in (train_ds, valid_ds, test_ds):
        ds.set_format("torch")

    collator = LastTokenLossCollator(tokenizer=tok)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        max_seq_length=a.max_len,
        args=TrainingArguments(
            output_dir=a.out_dir,
            per_device_train_batch_size=a.batch,
            gradient_accumulation_steps=a.grad_acc,
            num_train_epochs=a.epochs,
            learning_rate=a.lr,
            logging_steps=20,
            save_strategy="no",
            fp16=True,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            warmup_steps=100,
            group_by_length=True,
            remove_unused_columns=False,
            report_to="none",
        ),
    )

    log.info("▶ Fine-tuning …")
    trainer.train()
    log.info("▶ Done fine-tuning.")

    (pathlib.Path(a.out_dir)).mkdir(parents=True, exist_ok=True)
    model_path = pathlib.Path(a.out_dir) / "llama3_qora_classifier.pt"
    torch.save(model.state_dict(), model_path)
    tok.save_pretrained(a.out_dir)

    # ───────── evaluation ─────────
    def evaluate(ds, name):
        model.eval()
        preds, labels = [], []
        loader = torch.utils.data.DataLoader(
            ds, batch_size=a.batch, collate_fn=collator
        )
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                gold = batch["label_str"].to(device)
                logit = model(ids, attn).logits
                preds.extend(torch.argmax(logit, -1).cpu().tolist())
                labels.extend(gold.cpu().tolist())
        f1 = f1_score(labels, preds, average="macro")
        log.info("► %s Macro-F1 %.4f", name, f1)
        if name == "Test":
            log.info("\n%s", classification_report(labels, preds, target_names=["neg", "neu", "pos"]))
            log.info("Confusion matrix:\n%s", confusion_matrix(labels, preds))

    log.info("▶ Evaluating …")
    evaluate(valid_ds, "Valid")
    evaluate(test_ds, "Test")
    log.info("✓ All done.")


# ════════════════════════════════════════════════════
#                         CLI
# ════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=pathlib.Path, default="data/train.csv")
    p.add_argument("--valid_csv", type=pathlib.Path, default="data/valid.csv")
    p.add_argument("--test_csv", type=pathlib.Path, default="data/test.csv")
    p.add_argument("--model_name", default="unsloth/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--epochs", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--out_dir", default="llama3_lora")
    p.add_argument("--load_4bit", action="store_true")
    main(p.parse_args())
