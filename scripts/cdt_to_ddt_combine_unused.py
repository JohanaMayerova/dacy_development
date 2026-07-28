"""
Combine the Copenhagen Dependency Treebank (CDT / DaCoref, coreference
annotations) with the UD Danish Dependency Treebank (DDT, dependency
parses) into a single document-level spaCy corpus.

Key design decision: DDT's official train/dev/test split is done at the
sentence level and does not respect CDT's document boundaries (roughly
69% of CDT documents pull sentences from more than one DDT split). So
the final train/dev/test assignment for merged documents comes entirely
from CDT's own split, not
from DDT. DDT/DDT-derived sentences are only used to provide dependency
parses and surface features; their original split is only kept for the
small number of leftover DDT sentences that have no matching CDT
document at all.
"""

from collections import defaultdict
from pathlib import Path
from danlp.datasets import Dacoref
import spacy
from conllu import parse
from spacy.tokens import Doc, DocBin
from spacy.training.corpus import Corpus


Doc.set_extension("domain", default=None)
Doc.set_extension("sent_id", default=None)
Doc.set_extension("sent_ids", default=None)
Doc.set_extension("conllu", default=None)
Doc.set_extension("doc_id", default=None)
Doc.set_extension("split", default=None)

# danlp in this case is just a quick way to get the ds for me rn - there is most probably a smarter way (and less reliant on other packges) to do this
def export_dacoref(assets_dir: Path):
    cdt = Dacoref()
    train, dev, test = cdt.load_as_conllu(predefined_splits=True)

    dacoref_dir = assets_dir / "dacoref"
    dacoref_dir.mkdir(parents=True, exist_ok=True)

    for split_name, sentences in zip(
        ["train", "dev", "test"],
        [train, dev, test]
    ):
        out_path = dacoref_dir / f"{split_name}.conllu"

        with out_path.open("w", encoding="utf-8") as f:
            for sent in sentences:
                f.write(sent.serialize())

        print(f"{split_name}: {len(sentences)} sentences -> {out_path}")

DDT_CONLLU_FILES = {
    "train": "da_ddt-ud-train.conllu",
    "dev": "da_ddt-ud-dev.conllu",
    "test": "da_ddt-ud-test.conllu",
}

DOMAIN_MAPPING = {
    "mz": "magazine",
    "bn": "broadcast",
    "nw": "newswire",
}


def load_cdt(assets_path: Path):
    cdt_paths = {
        "train": assets_path / "dacoref" / "train.conllu",
        "dev": assets_path / "dacoref" / "dev.conllu",
        "test": assets_path / "dacoref" / "test.conllu",
    }
    fields = [
        "id",
        "form",
        "lemma",
        "upos",
        "xpos",
        "feats",
        "head",
        "deprel",
        "deps",
        "misc",
        "coref_id",
        "coref_rel",
        "doc_id",
        "qid",
    ]

    sentences = []
    split_ids = {"train": [], "dev": [], "test": []}
    for split, path in cdt_paths.items():
        text = path.read_text(encoding="utf-8")
        sents = parse(text, fields=fields)
        sentences.extend(sents)
        split_ids[split] = [sent.metadata["sent_id"] for sent in sents]

    return sentences, split_ids

# function to get a doc id - split mapping from cdts official splits and verifying how doc_id falls within a single split
def build_doc_id_to_split(cdt_sentences, cdt_split_ids):
    sent_id_to_cdt_split = {}
    for split, sent_ids in cdt_split_ids.items():
        for sent_id in sent_ids:
            sent_id_to_cdt_split[sent_id] = split

    doc_id_to_splits = defaultdict(set)
    for sent in cdt_sentences:
        sent_id = sent.metadata["sent_id"]
        doc_id = sent[0]["doc_id"]
        doc_id_to_splits[doc_id].add(sent_id_to_cdt_split[sent_id])

    mixed = {
        doc_id: splits
        for doc_id, splits in doc_id_to_splits.items()
        if len(splits) > 1
    }
    if mixed:
        example_items = list(mixed.items())[:5]
        raise AssertionError(
            f"{len(mixed)} CDT doc_ids span multiple CDT splits, "
            f"e.g. {example_items}"
        )

    doc_id_to_split = {doc_id: splits.pop() for doc_id, splits in doc_id_to_splits.items()}
    return doc_id_to_split


def _split_to_sentence_docs(docs):
    sentence_docs = []
    for doc in docs:
        for sent in doc.sents:
            sentence_docs.append(sent.as_doc())
    return sentence_docs

# function for similarly adding id to sentences from DDT 
def _add_sent_id(docs, split, dataset, assets_path):
    path = assets_path / dataset / DDT_CONLLU_FILES[split]
    text = path.read_text(encoding="utf-8")
    sentences = parse(text)
    for sent, doc in zip(sentences, docs):
        assert doc.text.strip() == sent.metadata["text"].strip()
        doc._.sent_id = sent.metadata["sent_id"]
        doc._.conllu = sent

# actually loading the DDT as sentence level docs, split is recording label of DDT (just so it does not get mixed up for final eval), keeping only the sentences that end up with no matching CDT document
def load_da_ddt(corpus_path: Path, assets_path: Path):
    nlp = spacy.blank("da")
    ddt_path = corpus_path / "UD_Danish-DDT"

    ddt = {}
    for split in ["train", "dev", "test"]:
        path = ddt_path / f"{split}.spacy"
        corpus = Corpus(path, shuffle=False)
        examples = list(corpus(nlp))
        docs = [e.reference for e in examples]
        docs = _split_to_sentence_docs(docs)
        for doc in docs:
            doc._.split = split
        ddt[split] = docs
        _add_sent_id(docs, split, dataset="UD_Danish-DDT", assets_path=assets_path)

    return ddt

# finally combining these two - I defined the split in the later steps than kenneths though
def combine_docs(cdt_sentences, ddt_docs):
    sent_id_to_doc_instance = {}
    for doc in ddt_docs:
        assert doc._.sent_id not in sent_id_to_doc_instance
        sent_id_to_doc_instance[doc._.sent_id] = doc

    doc_to_be_created: dict[str, list[str]] = {}
    sent_id_to_sent = {}
    for sent in cdt_sentences:
        sent_id = sent.metadata["sent_id"]
        doc_id = sent[0]["doc_id"]
        if doc_id not in doc_to_be_created:
            doc_to_be_created[doc_id] = []
        doc_to_be_created[doc_id].append(sent_id)
        sent_id_to_sent[sent_id] = sent

    docs = []
    for doc_id, sent_ids in doc_to_be_created.items():
        _docs = [sent_id_to_doc_instance.pop(sent_id) for sent_id in sent_ids]

        doc = Doc.from_docs(_docs)
        doc._.doc_id = doc_id
        doc._.sent_ids = sent_ids
        doc._.domain = DOMAIN_MAPPING[doc_id.split("/")[0]]
        doc._.conllu = [sent_id_to_sent[sent_id] for sent_id in sent_ids]
        docs.append(doc)

    for sent_id in list(sent_id_to_doc_instance.keys()):
        doc = sent_id_to_doc_instance.pop(sent_id)
        docs.append(doc)  # orphan DDT-only docs, keep their original DDT split

    return docs

# lowkey stolen from combine.py
def add_coreference(cdt_sentences, docs):
    doc_id_to_doc_instance = {

        doc._.doc_id: doc for doc in docs if doc._.doc_id is not None
    }
    doc_id_to_cdt_sent = defaultdict(list)
    for sent in cdt_sentences:
        doc_id = sent[0]["doc_id"]
        doc_id_to_cdt_sent[doc_id].append(sent)

    for doc_id, sents in doc_id_to_cdt_sent.items():
        clustermap = defaultdict(list)
        doc = doc_id_to_doc_instance[doc_id]
        tokens = [t for sent in sents for t in sent]
        assert len(doc) == len(tokens)
        for token, s_token in zip(tokens, doc):
            coref_rel = token["coref_rel"]
            if coref_rel == "-":
                continue
            clusters = sorted(coref_rel.split("|"), reverse=True)
            for mention in clusters:
                full_mention = mention.startswith("(") and mention.endswith(")")
                start_mention = mention.startswith("(")
                end_mention = mention.endswith(")")
                if full_mention:
                    cid = mention[1:-1]
                    clustermap[cid].insert(0, (s_token.i, s_token.i + 1))
                elif start_mention:
                    cid = mention[1:]
                    clustermap[cid].append(s_token.i)
                elif end_mention:
                    cid = mention[:-1]
                    start = clustermap[cid].pop()
                    clustermap[cid].insert(0, (start, s_token.i + 1))

        for i, (_key, vals) in enumerate(clustermap.items()):
            spans = [doc[start:end] for start, end in vals]
            skey = f"coref_clusters_{i}"
            doc.spans[skey] = spans
        
        # parse and get heads
        for i, (_key, val) in enumerate(clustermap.items()):
            heads = [doc[start:end].root.i for start, end in val]
            heads = list(set(heads))
            if len(heads) == 1:
                continue
            spans = [doc[hh : hh + 1] for hh in heads]
            skey = f"coref_head_clusters_{i}"
            doc.spans[skey] = spans

    return docs



if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    ASSETS_DIR = BASE_DIR / "assets"
    CORPUS_DIR = BASE_DIR / "corpus"

    export_dacoref(ASSETS_DIR)


    cdt_sentences, cdt_split_ids = load_cdt(ASSETS_DIR)
    print("CDT sentences loaded:", len(cdt_sentences))

    doc_id_to_split = build_doc_id_to_split(cdt_sentences, cdt_split_ids)
    print("CDT doc_ids with a confirmed single split:", len(doc_id_to_split))

    ddt = load_da_ddt(CORPUS_DIR, ASSETS_DIR)
    all_ddt_docs = [doc for split in ["train", "dev", "test"] for doc in ddt[split]]
    print("DDT sentence-docs loaded:", len(all_ddt_docs))

    combined_docs = combine_docs(cdt_sentences, all_ddt_docs)
    print("Combined docs:", len(combined_docs))

    combined_docs = add_coreference(cdt_sentences, combined_docs)

    output_dir = CORPUS_DIR / "cdt_ddt"
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev", "test"]:
        doc_bin = DocBin(store_user_data=True)
        n_docs = 0
        for doc in combined_docs:
            if doc._.doc_id is not None:
                doc_split = doc_id_to_split[doc._.doc_id]
            else:
                doc_split = doc._.split 
            if doc_split != split:
                continue
            doc_bin.add(doc)
            n_docs += 1

        output_path = output_dir / f"{split}.spacy"
        doc_bin.to_disk(output_path)
        print(f"{split}: {n_docs} docs -> {output_path}")