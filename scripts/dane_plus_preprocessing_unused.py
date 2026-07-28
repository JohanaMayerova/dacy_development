# this is practically useless :( DaNe plus dataset had all of the labels only in testing split
# keep the mapping, but using combine.py instead and preprocessing in yml
# 
from pathlib import Path
from collections import defaultdict
import spacy
import unicodedata
from datasets import load_dataset
from spacy.tokens import DocBin
from spacy.util import filter_spans


project_root = Path(__file__).resolve().parents[1]
out_dir = project_root / "corpus" / "ddt_dane"
out_dir.mkdir(parents=True, exist_ok=True)

nlp = spacy.blank("da")

ds = load_dataset("alexandrainst/dane")

# label mapping
LABEL_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",

    "ORG": "ORG",
    "ORGANIZATION": "ORG",

    "LOC": "LOC",
    "LOCATION": "LOC",
    "GPE": "LOC",
    "FACILITY": "LOC",

    "MISC": "MISC",
    "NORP": "MISC",

    "EVENT": "EVENT",
    "PRODUCT": "PRODUCT",
    "WORK_OF_ART": "WORK_OF_ART",
    "WORK OF ART": "WORK_OF_ART",

    "DATE": "DATE",
    "TIME": "TIME",
    "MONEY": "MONEY",
    "CARDINAL": "CARDINAL",
    "ORDINAL": "ORDINAL",
    "PERCENT": "PERCENT",
    "QUANTITY": "QUANTITY",
    "LAW": "LAW",
}


def load_ddt_split(path, nlp):
    return list(DocBin().from_disk(path).get_docs(nlp.vocab))


def build_dane_index(dataset_split):
    index = defaultdict(list)

    def normalize_text(s: str) -> str:
        return " ".join(unicodedata.normalize("NFC", s).split())

    for ex in dataset_split:
        key = normalize_text(ex["text"])
        index[key].append(ex["ents"])

    return index

def inject_dane(doc, dane_index):
    text = doc.text
    added = []

    def normalize_text(s: str) -> str:
        return " ".join(unicodedata.normalize("NFC", s).split())

    key = normalize_text(text)
    if key not in dane_index:
        return doc

    for ents_group in dane_index[key]:
        for ent in ents_group:

            label = LABEL_MAP.get(ent["label"])
            if not label:
                continue

            span = doc.char_span(
                ent["start"],
                ent["end"],
                label=label,
                alignment_mode="contract",
            )

            if span is not None:
                added.append(span)

    # merge safely with existing DDT ents
    doc.ents = filter_spans(list(doc.ents) + added)

    return doc

project_root = Path(__file__).resolve().parents[1]

ddt_dir = project_root / "corpus" / "UD_Danish-DDT"
out_dir = project_root / "corpus" / "ddt_dane"
out_dir.mkdir(parents=True, exist_ok=True)

# does not work in the same way as loading dane plus for some reason? moved it into projct.yml to keep it organized
ds = load_dataset("alexandrainst/dane")

for split in ["train", "dev", "test"]:

    print(f"\n=== PROCESSING {split.upper()} ===")
    ddt_docs = load_ddt_split(ddt_dir / f"{split}.spacy", nlp)
    dane_index = build_dane_index(ds[split])
    updated_docs = []
    missing = 0

    for doc in ddt_docs:
        old_ents = len(doc.ents)

        doc = inject_dane(doc, dane_index)

        if len(doc.ents) == old_ents:
            missing += 1

        updated_docs.append(doc)
    db = DocBin(store_user_data=True)

    label_counts = defaultdict(int)

    for doc in updated_docs:
        for ent in doc.ents:
            label_counts[ent.label_] += 1
        db.add(doc)

    out_path = out_dir / f"{split}.spacy"
    db.to_disk(out_path)

    print("Docs:", len(updated_docs))
    print("No new NER added:", missing)
    print("Labels:")
    for k, v in sorted(label_counts.items()):
        print(f"  {k:15s} {v}")

    print("Saved:", out_path)