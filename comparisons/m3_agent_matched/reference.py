"""Execute selected, unchanged upstream definitions without import-time GPU/API setup.

The source files are frozen with the experiment. AST selection removes imports
and top-level configuration/model construction, not function or class bodies.
Every selected definition is fingerprinted in the run's reference receipt.
"""
import ast
import base64
import hashlib
import json
import logging
import random
import re
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


class CachedEmbeddings:
    def __init__(self, embedder):
        self.embedder = embedder
        self.cache = {}
        self.calls = 0
        self.texts = 0

    def __call__(self, model, texts):
        if model not in ("text-embedding", "text-embedding-3-large"):
            raise ValueError(f"Unexpected upstream embedding route: {model}")
        texts = list(texts)
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            vectors = self.embedder.embed_texts(missing)
            if len(vectors) != len(missing):
                raise ValueError("Embedding service returned an incomplete batch")
            for text, vector in zip(missing, vectors):
                self.cache[text] = list(vector)
            self.calls += 1
            self.texts += len(missing)
        return [self.cache[t] for t in texts], 0


def load_reference(root: Path, embedding_callback):
    config = json.loads((root / "configs/processing_config.json").read_text())
    namespace = {
        "__name__": "m3_matched_reference",
        "np": np, "random": random, "logging": logging,
        "logger": logging.getLogger("m3_matched.reference"),
        "re": re, "json": json, "base64": base64, "BytesIO": BytesIO, "Image": Image,
        "DBSCAN": DBSCAN, "cosine_similarity": cosine_similarity,
        "processing_config": config, "MAX_RETRIES": config["max_retries"],
        "parallel_get_embedding": embedding_callback,
    }
    definitions = [
        ("mmagent/memory_processing.py", ["parse_video_caption"]),
        ("mmagent/videograph.py", ["VideoGraph"]),
        ("mmagent/memory_processing_qwen.py", ["process_memories"]),
        ("mmagent/retrieve.py", ["translate", "back_translate", "get_related_nodes",
                                "retrieve_from_videograph", "search"]),
    ]
    receipt = {}
    for name, wanted in definitions:
        path = root / name
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                    and node.name in wanted]
        if {n.name for n in selected} != set(wanted):
            raise ValueError(f"Missing expected reference definitions in {name}")
        executable = ast.Module(body=selected, type_ignores=[])
        exec(compile(executable, str(path), "exec"), namespace)
        receipt[name] = {"sha256": hashlib.sha256(source.encode()).hexdigest(),
                         "definitions": wanted}
    tree = ast.parse((root / "mmagent/prompts.py").read_text())
    prompt = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "prompt_generate_memory_with_ids_sft"
                          for t in n.targets))
    # Fix only the documented singular/plural interface inconsistency.
    prompt = prompt.replace('"video_description":', '"video_descriptions":')
    return SimpleNamespace(**{name: namespace[name] for _, names in definitions for name in names},
                           memory_config=json.loads((root / "configs/memory_config.json").read_text()),
                           prompt=prompt, receipt=receipt)


def normalise_memory(value):
    if not isinstance(value, dict):
        raise ValueError("Invalid M3-Agent output: expected object")
    value = dict(value)
    if "video_description" in value:
        if ("video_descriptions" in value
                and value["video_descriptions"] != value["video_description"]):
            raise ValueError("Invalid M3-Agent output: ambiguous description keys")
        value["video_descriptions"] = value.pop("video_description")
    for key in ("video_descriptions", "high_level_conclusions"):
        if not isinstance(value.get(key), list) or not all(
                isinstance(x, str) and x.strip() for x in value[key]):
            raise ValueError(f"Invalid M3-Agent output: {key} must contain strings")
    if not value["video_descriptions"]:
        raise ValueError("Invalid M3-Agent output: empty episodic memory")
    return {key: value[key] for key in ("video_descriptions", "high_level_conclusions")}
