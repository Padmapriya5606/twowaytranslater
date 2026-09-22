"""
Fine-tunes the gloss -> fluent-sentence correction model on your
ISL gloss/English sentence pairs (ISLTranslate-style CSV).

Expected CSV columns: gloss_sequence, english_sentence
e.g.  "I GO SCHOOL YESTERDAY", "I went to school yesterday."

Usage:
    python src/training/train_transformer_nlp.py
    python src/training/train_transformer_nlp.py --model_name t5-small --epochs 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="t5-small")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_NLP)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=config.LR_NLP)
    parser.add_argument("--csv", default=str(config.ISL_SENTENCE_PAIRS_CSV))
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists() or csv_path.stat().st_size < 50:
        csltr_csv = config.DATA_RAW_DIR / "isl_sentences" / "isl_csltr_sentence_pairs.csv"
        if csltr_csv.exists() and csltr_csv.stat().st_size >= 50:
            print(f"[notice] Default CSV '{csv_path}' is missing or empty. Falling back to '{csltr_csv}'.")
            csv_path = csltr_csv

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Place your gloss/sentence pairs CSV at "
            f"data/raw/isl_sentences/isl_gloss_english_pairs.csv (see data/README.md)."
        )

    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
        Seq2SeqTrainer, Seq2SeqTrainingArguments,
    )

    df = pd.read_csv(csv_path).dropna()
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=42)

    if "gloss_sequence" in df.columns:
        df["input_text"] = "correct isl gloss: " + df["gloss_sequence"].astype(str)
    elif "video_id" in df.columns:
        df["input_text"] = "correct isl gloss: " + df["video_id"].astype(str)
    else:
        df["input_text"] = "correct isl gloss: " + df["english_sentence"].astype(str).str.upper()

    df["target_text"] = df["english_sentence"].astype(str)

    dataset = Dataset.from_pandas(df[["input_text", "target_text"]])
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input_text"], max_length=config.MAX_GLOSS_LEN, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            batch["target_text"], max_length=config.MAX_GLOSS_LEN, truncation=True, padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(preprocess, batched=True, remove_columns=["input_text", "target_text"])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    config.NLP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(config.NLP_MODEL_DIR),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        load_best_model_at_end=True,
        logging_steps=20,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()

    eval_preds = trainer.predict(tokenized["test"])
    preds = eval_preds.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = eval_preds.label_ids
    labels = [[l for l in label if l != -100] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1

    bleu_scores = []
    for pred, ref in zip(decoded_preds, decoded_labels):
        pred_tokens = pred.strip().lower().split()
        ref_tokens = [ref.strip().lower().split()]
        if pred_tokens and ref_tokens[0]:
            score = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
            bleu_scores.append(score)

    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    exact_matches = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p.strip().lower() == l.strip().lower())
    exact_accuracy = exact_matches / len(decoded_labels) if decoded_labels else 0.0

    print(f"\n" + "=" * 60)
    print(f"TRANSFORMER NLP MODEL TEST ACCURACY & BLEU SCORE")
    print("=" * 60)
    print(f"Exact Match Accuracy: {exact_accuracy * 100:.2f}% ({exact_matches}/{len(decoded_labels)})")
    print(f"BLEU-4 Translation Score: {avg_bleu * 100:.2f}%")

    trainer.save_model(str(config.NLP_MODEL_DIR))
    tokenizer.save_pretrained(str(config.NLP_MODEL_DIR))
    print(f"Saved fine-tuned NLP model -> {config.NLP_MODEL_DIR}")


if __name__ == "__main__":
    main()
