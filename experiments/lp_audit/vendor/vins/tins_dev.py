"""TINS on the 900-class dev label space, reusing the upstream functions unchanged.

Only the logging hooks (7 added lines, inactive when hook=None) differ from commit 194759d.
"""
import collections
import importlib.util
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from . import config as C


def import_tins(module_file="eval_tins_w_init.py", name="eval_tins_w_init", tins_dir=None):
    tins_dir = Path(tins_dir or C.TINS_DIR)
    if str(tins_dir) not in sys.path:
        sys.path.insert(0, str(tins_dir))
    os.chdir(tins_dir)            # upstream reads data/ImageNet/... relative to cwd
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, tins_dir / module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_args(t, cache_dir, name, stream_seed=123):
    saved = sys.argv
    sys.argv = C.tins_argv(cache_dir, name, stream_seed)
    try:
        return t.process_args()
    finally:
        sys.argv = saved


def get_logger(path=None):
    log = logging.getLogger("vins")
    if not log.handlers:
        log.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(handler)
        if path:
            file_handler = logging.FileHandler(path)
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            log.addHandler(file_handler)
    return log


def load_clip(t, args):
    """Upstream loader; the download is redirected to the sha256-verified local checkpoint."""
    original = t.official_clip.clip._download
    t.official_clip.clip._download = lambda url, root=None: original(url, str(C.CLIP_WEIGHTS_DIR))
    try:
        net, preprocess = t.load_official_clip(args)
    finally:
        t.official_clip.clip._download = original
    return net, preprocess


def build_order(n_id, n_ood, seed):
    """Identical to upstream build_mixed_stream_loader (eval_tins_w_init.py L1390-1393)."""
    order = [(0, i) for i in range(n_id)] + [(1, i) for i in range(n_ood)]
    random.Random(seed).shuffle(order)
    return order


class Recorder:
    """Read-only hook: copies tensors to CPU, never modifies them, never touches an RNG."""

    def __init__(self):
        self.batches = []
        self.cur = None

    def on_arrival(self, start, scores):
        self.cur = {"start": int(start), "S_arrival": scores.detach().float().cpu().numpy().copy()}

    def on_activate(self, mask):
        self.cur["activated"] = mask.detach().cpu().numpy().copy()

    def on_criterion(self, activate_mask, intermodal_sim, base_sim, mask):
        delta = (base_sim.unsqueeze(0) - intermodal_sim).mean(dim=1)
        self.cur["inv_pass"] = mask.detach().cpu().numpy().copy()
        self.cur["inv_delta"] = delta.detach().float().cpu().numpy().copy()

    def on_final(self, start, scores, bank_size, buffer_size):
        assert self.cur is not None and self.cur["start"] == int(start)
        self.cur["S_final"] = scores.detach().float().cpu().numpy().copy()
        self.cur["bank_size"] = int(bank_size)
        self.cur["buffer_size"] = int(buffer_size)
        self.batches.append(self.cur)
        self.cur = None

    def per_sample(self, n):
        s_arrival = np.concatenate([b["S_arrival"] for b in self.batches])
        s_final = np.concatenate([b["S_final"] for b in self.batches])
        activated = np.concatenate([b["activated"] for b in self.batches])
        inv_pass = np.full(n, -1, dtype=np.int8)       # -1: not seeded
        inv_delta = np.full(n, np.nan, dtype=np.float32)
        batch_index = np.empty(n, dtype=np.int32)
        for index, batch in enumerate(self.batches):
            size = len(batch["S_arrival"])
            batch_index[batch["start"]:batch["start"] + size] = index
            if "inv_pass" in batch:
                pos = batch["start"] + np.flatnonzero(batch["activated"])
                inv_pass[pos] = batch["inv_pass"].astype(np.int8)
                inv_delta[pos] = batch["inv_delta"]
        assert len(s_arrival) == n == len(s_final)
        return {"S_arrival": s_arrival, "S_final": s_final, "seeded": activated, "inv_pass": inv_pass,
                "inv_delta": inv_delta, "batch_index": batch_index}

    def bank_trace(self):
        return [{"batch": i, "start": b["start"], "n_seeded": int(b["activated"].sum()),
                 "n_pass": int(b["inv_pass"].sum()) if "inv_pass" in b else 0,
                 "bank_size": b["bank_size"], "buffer_size": b["buffer_size"]} for i, b in enumerate(self.batches)]


def normalize_word(text):
    return " ".join(str(text).replace("_", " ").lower().split())


def leak_check(t, args, positive_labels, selected_words, selected_texts, heldout):
    """Record (do not exclude) held-out class names / WordNet synonyms among TINS's static negatives."""
    nouns, adjs = t.collect_negative_words(t.resolve_wordnet_dir(args), positive_labels)
    pool = {normalize_word(w) for w in list(nouns) + list(adjs)}
    selected = collections.defaultdict(list)
    for word, text in zip(selected_words, selected_texts):
        selected[normalize_word(word)].append(text)
    rows = []
    for item in heldout:
        names = {normalize_word(item["clean_name"]), normalize_word(item["raw_name"])}
        names |= {normalize_word(lemma) for lemma in item["lemmas"]}
        hits = sorted(names & set(selected))
        rows.append({
            "wnid": item["wnid"], "clean_name": item["clean_name"], "names_checked": sorted(names),
            "in_candidate_pool": sorted(names & pool), "in_selected_static_negatives": hits,
            "selected_texts": [text for hit in hits for text in selected[hit]],
        })
    return {
        "n_heldout": len(rows),
        "n_heldout_with_name_in_selected_static_negatives": sum(bool(r["in_selected_static_negatives"]) for r in rows),
        "n_heldout_with_name_in_candidate_pool": sum(bool(r["in_candidate_pool"]) for r in rows),
        "n_selected_static_negatives": len(selected_words),
        "rows": rows,
    }


def setup_to_device(setup, device="cuda"):
    init = dict(setup["init_candidates"])
    init["candidate_features"] = init["candidate_features"].to(device)
    init["candidate_reg_loss"] = init["candidate_reg_loss"].to(device)
    return {
        "positive_features": setup["positive_features"].to(device),
        "negative_features": setup["negative_features"].to(device),
        "class_prototypes": setup["class_prototypes"].to(device),
        "base_sim": setup["base_sim"].to(device),
        "init_candidates": init,
    }


def run_stream(t, args, net, dev_setup, image_features, hook=None):
    return t.compute_tins_scores_from_image_features(
        image_features=image_features,
        args=args,
        model=net,
        positive_features=dev_setup["positive_features"],
        negative_features=dev_setup["negative_features"],
        inversion_init_candidates=dev_setup["init_candidates"],
        class_prototypes=dev_setup["class_prototypes"],
        base_sim=dev_setup["base_sim"],
        hook=hook,
    )


def shadow_inversion(t, args, net, dev_setup, feats, chunk=C.BATCH):
    """TINS inversion + ID-prototype-separated criterion on given samples, bank untouched.
    Inversion and the criterion depend only on (feature, prototypes, base_sim, init candidates)."""
    device = dev_setup["positive_features"].device
    dim = dev_setup["positive_features"].shape[1]
    fixed = torch.cat([dev_setup["positive_features"], dev_setup["negative_features"]], dim=0)
    passes, deltas = [], []
    for lo in range(0, feats.shape[0], chunk):
        batch = feats[lo:lo + chunk].to(device)
        recorder = Recorder()
        recorder.cur = {"start": lo}
        empty_f = torch.zeros((0, dim), dtype=torch.float16)
        empty_s = torch.zeros(0, dtype=torch.float32)
        t.maybe_expand_dynamic_bank(
            args=args, model=net, image_features=batch,
            current_scores=torch.zeros(batch.shape[0], device=device),   # forces activation (0 < beta)
            class_prototypes=dev_setup["class_prototypes"], base_sim=dev_setup["base_sim"],
            fixed_text_bank=fixed, bank_features=empty_f, bank_scores=empty_s,
            placeholder_token_id=None, init_candidates=dev_setup["init_candidates"],
            buffer_features=empty_f.clone(), buffer_scores=empty_s.clone(), hook=recorder,
        )
        passes.append(recorder.cur["inv_pass"])
        deltas.append(recorder.cur["inv_delta"])
    return np.concatenate(passes), np.concatenate(deltas)
