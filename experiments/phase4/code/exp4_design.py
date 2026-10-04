"""Exp 4 design (registered in prereg_p4.json): per U1 split, 50 probe / 50 background held-out classes, queries,
same-class pools, near/far partner probe classes (L/14 class means of the same-class pools), the fixed 1,024-image
history, nested replacement positions and the near-duplicate sources. Uses features only (no score)."""
import json

import numpy as np
import pandas as pd

from bank_data import Features
from common import BANKS, dump

SEED = 20260928


def design(k, F, pool):
    sp = json.loads((BANKS / "U1" / f"split{k}.json").read_text())
    held = list(sp["heldout"])
    rng = np.random.default_rng([SEED, 40, k])
    perm = [held[i] for i in rng.permutation(len(held))]
    probe, background = perm[:50], perm[50:]
    ood = pool[pool.role == "oodeval"]
    by = {w: ood[ood.wnid == w].sort_values("slot").sample_id.tolist() for w in held}
    assert all(len(v) == 50 for v in by.values())
    same = {w: by[w][8:28] for w in probe}
    means = np.stack([F.get("L14", same[w]).mean(0) for w in probe])
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    sim = means @ means.T
    np.fill_diagonal(sim, np.nan)
    near = {w: probe[int(np.nanargmax(sim[i]))] for i, w in enumerate(probe)}
    far = {w: probe[int(np.nanargmin(sim[i]))] for i, w in enumerate(probe)}
    bg_ood = [s for w in background for s in by[w][:6]]
    bg_ood = [bg_ood[i] for i in rng.permutation(len(bg_ood))[:256]]
    idw = set(sp["id_wnids"])
    ide = pool[(pool.role == "ideval") & pool.wnid.isin(idw)].sort_values(["idx_1k", "slot"]).sample_id.tolist()
    pid = rng.permutation(len(ide))
    hist_id, id_queries = [ide[i] for i in pid[:768]], [ide[i] for i in pid[768:832]]
    items = [(s, 0) for s in hist_id] + [(s, 1) for s in bg_ood]
    order = rng.permutation(len(items))
    history = [items[i][0] for i in order]
    is_ood = [items[i][1] for i in order]
    ood_pos = [i for i, f in enumerate(is_ood) if f]
    replace_order = [ood_pos[i] for i in rng.permutation(len(ood_pos))]
    probes = {w: {"queries": by[w][:8], "same": same[w], "dup_src": by[w][28], "near": near[w], "far": far[w],
                  "near_sim": float(sim[probe.index(w), probe.index(near[w])]),
                  "far_sim": float(sim[probe.index(w), probe.index(far[w])])} for w in probe}
    return {"split": k, "probe": probe, "background": background, "probes": probes, "history": history,
            "history_is_ood": is_ood, "replace_order": replace_order, "id_queries": id_queries, "r": [0, 1, 2, 5, 10, 20]}


def main():
    F = Features()
    pool = pd.read_parquet(BANKS / "imagenet_pool.parquet")
    for k in range(1, 6):
        dump(BANKS / "exp4" / f"split{k}.json", design(k, F, pool))
    print("exp4 designs written")


if __name__ == "__main__":
    main()
