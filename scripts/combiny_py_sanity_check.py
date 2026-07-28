"""
Post-hoc validation for combine.py

Checks two things flagged in review:
  1. Coverage: does every CDT doc_id referenced in the merged `docs` list
     actually have a split assignment? (catches the silent `continue` drop
     in the final split-assembly loop)
  2. Coref cluster ID ordering: does `sorted(coref_rel.split("|"), reverse=True)`
     ever see multi-digit cluster IDs, where lexicographic sort would put
     them in the wrong order relative to numeric sort?

Run this in the same directory as combine.py (or adjust the import below).
It does NOT write any .spacy files — read-only diagnostics.
"""

import re
from collections import Counter

# --- adjust this import to match your actual module/file name ---
import combine as c
# -------------------------------------------------------------

def check_split_coverage():
    print("=" * 60)
    print("SPLIT COVERAGE CHECK")
    print("=" * 60)

    total = 0
    dropped = []
    counted_by_split = Counter()

    for doc in c.docs:
        total += 1
        doc_id = doc._.doc_id
        sent_id = doc._.sent_id

        if doc_id is not None and doc_id in c.doc_id_to_split_mapping:
            counted_by_split[c.doc_id_to_split_mapping[doc_id]] += 1
        elif sent_id is not None and sent_id in c.sent_id_to_split_mapping:
            counted_by_split[c.sent_id_to_split_mapping[sent_id]] += 1
        else:
            dropped.append((doc_id, sent_id, doc.text[:60]))

    print(f"Total docs in `docs`: {total}")
    print(f"Assigned to a split:  {total - len(dropped)}")
    print(f"Silently dropped:     {len(dropped)}")
    print(f"Per-split counts:     {dict(counted_by_split)}")

    if dropped:
        print("\nDropped docs (doc_id, sent_id, text preview):")
        for doc_id, sent_id, text in dropped[:20]:
            print(f"  doc_id={doc_id!r} sent_id={sent_id!r} text={text!r}")
        if len(dropped) > 20:
            print(f"  ... and {len(dropped) - 20} more")
    else:
        print("\nNo docs dropped — every doc has a split assignment.")

    # Specifically: are there CDT doc_ids with no split mapping at all?
    cdt_doc_ids = {sent[0]["doc_id"] for sent in c.cdt_sentences}
    unmapped_cdt_ids = cdt_doc_ids - set(c.doc_id_to_split_mapping.keys())
    print(f"\nCDT doc_ids with no entry in doc_id_to_split_mapping: {len(unmapped_cdt_ids)}")
    if unmapped_cdt_ids:
        print(f"  e.g. {list(unmapped_cdt_ids)[:10]}")


def check_coref_cluster_ids():
    print("\n" + "=" * 60)
    print("COREF CLUSTER ID ORDERING CHECK")
    print("=" * 60)

    multidigit_seen = set()
    tokens_with_multiple_simultaneous = 0
    total_coref_tokens = 0

    for sent in c.cdt_sentences:
        for token in sent:
            coref_rel = token.get("coref_rel", "-")
            if coref_rel == "-":
                continue
            total_coref_tokens += 1
            mentions = coref_rel.split("|")
            if len(mentions) > 1:
                tokens_with_multiple_simultaneous += 1
            for mention in mentions:
                cid = re.sub(r"[()]", "", mention)
                if len(cid) > 1:
                    multidigit_seen.add(cid)

    print(f"Total tokens with coref annotation: {total_coref_tokens}")
    print(f"Tokens with >1 simultaneous mention (open/close overlap): {tokens_with_multiple_simultaneous}")
    print(f"Distinct multi-digit cluster IDs seen: {len(multidigit_seen)}")
    if multidigit_seen:
        sample = sorted(multidigit_seen, key=int)[:10]
        print(f"  sample: {sample}")
        # demonstrate whether lexicographic vs numeric sort actually differs
        lex_sorted = sorted(multidigit_seen, reverse=True)
        num_sorted = sorted(multidigit_seen, key=int, reverse=True)
        if lex_sorted != num_sorted:
            print("  -> lexicographic and numeric sort ORDER DIFFERS for these IDs.")
            print(f"     lexicographic: {lex_sorted[:10]}")
            print(f"     numeric:       {num_sorted[:10]}")
            if tokens_with_multiple_simultaneous > 0:
                print("  -> RISK IS LIVE: multi-digit IDs exist AND tokens have simultaneous "
                      "mentions, so the sort order in add_coreference could mis-pair spans.")
            else:
                print("  -> low risk in practice: no tokens had simultaneous mentions, "
                      "so the sort order never actually mattered.")
        else:
            print("  -> lexicographic and numeric sort agree here; not currently a live bug.")
    else:
        print("  -> no multi-digit cluster IDs found; sort order is a non-issue for this data.")


if __name__ == "__main__":
    check_split_coverage()
    check_coref_cluster_ids()