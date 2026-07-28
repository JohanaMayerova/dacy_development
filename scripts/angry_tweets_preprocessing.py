import sys
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split


def csv_to_docbin(df, nlp, text_col="text", label_col="label"):
    doc_bin = DocBin()
    for text, label in zip(df[text_col], df[label_col]):
        doc = nlp.make_doc(str(text))
        doc.cats = {cat: (1.0 if cat == label else 0.0) for cat in df[label_col].unique()}
        doc_bin.add(doc)
    return doc_bin


def main():
    train_csv, test_csv, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nlp = spacy.blank("da")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_df, dev_df = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df["label"],
        random_state=0,
    )

    print(f"train: {len(train_df)}  dev: {len(dev_df)}  test: {len(test_df)}")
    print("train label distribution:")
    print(train_df["label"].value_counts(normalize=True))

    csv_to_docbin(train_df, nlp).to_disk(out_dir / "train.spacy")
    csv_to_docbin(dev_df, nlp).to_disk(out_dir / "dev.spacy")
    csv_to_docbin(test_df, nlp).to_disk(out_dir / "test.spacy")


if __name__ == "__main__":
    main()