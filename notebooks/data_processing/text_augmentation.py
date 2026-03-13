import pandas as pd
import random
import re
import pandas as pd
import numpy as np
import torch
import random
import time
import itertools
from data_processing.text_embeddings import TextEmbedding
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling
)

from openpyxl import load_workbook

def augment_moa_descriptions(
    input_excel_path: str,
    input_Sheet_name: str = "SemanticAugmentation",
    sentence_sheet_name: str = "SentenceAugmentation",
    word_sheet_name: str = "WordAugmentation",
    n_augmented: int = 9,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)

    def smart_sentence_split(text):
        splits = re.split(r'\.\s+(?=[A-Z])', text)
        return [s.strip() + '.' for s in splits if s.strip()]

    # --- Load original MoA descriptions ---
    df = pd.read_excel(input_excel_path, sheet_name=input_Sheet_name)
    desc_col = df.columns[1]   # e.g. "Description"

    # === Sentence-Level Augmentation ===
    sentence_augmented_df = pd.DataFrame(
        columns=["MoA"] + [f"Aug_{i}" for i in range(n_augmented + 1)]
    )

    for idx, row in df.iterrows():
        moa = row["MoA"]
        base_text = row[desc_col]
        if not isinstance(base_text, str):
            continue

        parts = smart_sentence_split(base_text)
        row_augments = [base_text]

        if len(parts) <= 1:
            print(f"⚠️ Row {idx}: only {len(parts)} sentence(s) found – cannot create unique shuffles.")
            row_augments += [base_text] * n_augmented

        else:
            attempts = 0
            max_attempts = n_augmented * 10
            while len(row_augments) < n_augmented + 1 and attempts < max_attempts:
                shuffled = random.sample(parts, len(parts))
                new_text = " ".join(shuffled)
                if new_text not in row_augments:
                    row_augments.append(new_text)
                attempts += 1

            if len(row_augments) < n_augmented + 1:
                missing = (n_augmented + 1) - len(row_augments)
                generated = len(row_augments) - 1
                print(f"⚠️ Row {idx}: only generated {generated} unique sentence-shuffles "
                    f"(requested {n_augmented}), padding {missing} with original.")
                row_augments += [base_text] * missing

        sentence_augmented_df.loc[len(sentence_augmented_df)] = [moa] + row_augments

    # === Word-Level Augmentation ===
    word_augmented_df = pd.DataFrame(
        columns=["MoA"] + [f"Aug_{i}" for i in range(n_augmented + 1)]
    )

    for idx, row in df.iterrows():
        moa = row["MoA"]
        base_text = row[desc_col]
        if not isinstance(base_text, str):
            continue

        words = base_text.split()
        row_augments = [base_text]

        if len(words) <= 1:
            print(f"⚠️ Row {idx}: only {len(words)} word(s) found – cannot create unique shuffles.")
            row_augments += [base_text] * n_augmented

        else:
            attempts = 0
            max_attempts = n_augmented * 10
            while len(row_augments) < n_augmented + 1 and attempts < max_attempts:
                shuffled = random.sample(words, len(words))
                new_text = " ".join(shuffled)
                if new_text not in row_augments:
                    row_augments.append(new_text)
                attempts += 1

            if len(row_augments) < n_augmented + 1:
                missing = (n_augmented + 1) - len(row_augments)
                print(f"⚠️ Row {idx}: only generated {len(row_augments)-1} unique word‐shuffles "
                    f"(requested {n_augmented}), padding {missing} with original.")
                row_augments += [base_text] * missing

        word_augmented_df.loc[len(word_augmented_df)] = [moa] + row_augments

    # === Save to Excel ===
    with pd.ExcelWriter(input_excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        sentence_augmented_df.to_excel(writer, sheet_name=sentence_sheet_name, index=False)
        word_augmented_df.to_excel(writer, sheet_name=word_sheet_name, index=False)

    print(f"✅ Augmentation completed and saved to '{sentence_sheet_name}' and '{word_sheet_name}' sheets.")


def compute_llm_perplexities(
    input_excel_path: str,
    sheet_name: str,
    model_checkpoints: dict,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> dict:
    """
    Compute perplexity of augmented MoA texts using different LLMs.

    Returns:
        dict: Mapping model name -> list of perplexities.
    """

    df = pd.read_excel(input_excel_path, sheet_name=sheet_name)
    texts = df.drop(columns=["MoA"], errors="ignore").values.flatten()
    texts = [t for t in texts if isinstance(t, str)]

    def compute_perplexity(model, tokenizer, text, collator):
        batch = collator([tokenizer(text, truncation=True, max_length=512)])
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            perplexity = torch.exp(loss)
        return perplexity.item()

    results = {}

    for model_name, model_ckpt in model_checkpoints.items():
        print(f"🔵 Running {model_name}...")
        start_time = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        model = AutoModelForMaskedLM.from_pretrained(model_ckpt)
        model.eval()
        model.to(device)

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15,
            return_tensors="pt"
        )

        perplexities = []
        for text in texts:
            try:
                ppl = compute_perplexity(model, tokenizer, text, collator)
                perplexities.append(ppl)
            except Exception as e:
                print(f"Error on text: {e}")
                perplexities.append(np.nan)

        avg_ppl = np.nanmean(perplexities)
        runtime = time.time() - start_time

        print(f"{model_name} ✅ Avg Perplexity: {avg_ppl:.2f} | Time: {runtime:.1f}s")
        results[model_name] = perplexities

    return results

def generate_text_embeddings(
    
    input_excel_path: str,
    sheet_name: str,
    model_checkpoints: dict,
    output_pickle_path: str
):
    """
    Generate text embeddings for augmented MoA descriptions using different LLMs.
    """
    
    df = pd.read_excel(input_excel_path, sheet_name=sheet_name)
    df.columns = df.columns.map(str)

    augmentation_columns = [col for col in df.columns if col != "MoA"]

    records = []
    for idx, row in df.iterrows():
        for col in augmentation_columns:
            if isinstance(row[col], str):
                records.append({
                    "MoA": row["MoA"],
                    "Augmentation": col,
                    "Text": row[col]
                })

    flat_df = pd.DataFrame(records)

    all_records = []

    for model_name, model_ckpt in model_checkpoints.items():
        print(f"🟢 Encoding with {model_name}...")
        embedder = TextEmbedding(model_checkpoint=model_ckpt)

        for idx, row in flat_df.iterrows():
            try:
                embedder.set_text(row["Text"], n_augmentations=None)
                emb = embedder.get_embedding(output="numpy")
                emb = np.asarray(emb).flatten()

                # ——— shape check ———
                if emb.ndim != 1 or emb.shape[0] != 768:
                    raise ValueError(
                        f"Embedding for model {model_name!r}, record {idx} has shape "
                        f"{emb.shape!r}, but expected (768,)"
                    )
                all_records.append({
                    "MoA": row["MoA"],
                    "Augmentation": row["Augmentation"],
                    "Model": model_name,
                    "Embedding": emb
                })
            except Exception as e:
                print(f"Embedding failed: {e}")
                all_records.append({
                    "MoA": row["MoA"],
                    "Augmentation": row["Augmentation"],
                    "Model": model_name,
                    "Embedding": None
                })

    embedding_df = pd.DataFrame(all_records)
    embedding_df.to_pickle(output_pickle_path)
    print(f"✅ Embeddings saved to {output_pickle_path}")


