# /// script
# requires-python = ">=3.10"
# dependencies = [
#      "click",
#     "spacy",
#     "cupy-cuda13x[ctk]",
#     "en-core-web-sm",
# ]# 
# [tool.uv.sources]
# en-core-web-sm = { url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" }
# ///
"""Minimal repro: spaCy on a CUDA 13 / Blackwell GPU (e.g. B200, sm_100).

Run with:  uv run setup/gpu_setup.py
"""

import cupy
import spacy

# Sanity: CuPy sees the GPU
print("CuPy:", cupy.__version__, "| device:",
      cupy.cuda.runtime.getDeviceProperties(0)["name"].decode())

print("Require gpu:", spacy.require_gpu())

# Run a real pipeline on the GPU
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying a U.K. startup for $1 billion.")
print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])