"""Phase 7 stream builders. A stream is (ids in arrival order, is_ood flags). Labels are used here to compose streams
(prevalence, recurrence, arrival pattern) and never reach the engine."""
import numpy as np

from p7common import SEED
from vins.tins_dev import build_order


def standard(P, seed):
    """The TINS order rule on the evaluation pools (as Phase 4 / 5)."""
    order = build_order(len(P.id_ids), len(P.ood_ids), seed)
    ids = [P.id_ids[i] if o == 0 else P.ood_ids[i] for o, i in order]
    return ids, np.array([o for o, _ in order], bool)


def matched(P, seed, n_unknown=40, m=25, n_id=4000):
    """Same composition for every bank: n_unknown classes x m images and n_id ID images, uniformly random order."""
    rng = np.random.default_rng([SEED, 81, seed, abs(hash_name(P.name))])
    oc = P.cls(P.ood_ids)
    classes, counts = np.unique(oc, return_counts=True)
    ok = classes[counts >= m]
    chosen = np.sort(rng.permutation(len(ok))[:n_unknown])
    ood = []
    for c in ok[chosen]:
        idx = np.flatnonzero(oc == c)
        ood += [P.ood_ids[i] for i in np.sort(rng.permutation(idx)[:m])]
    idx = np.sort(rng.permutation(len(P.id_ids))[:n_id])
    ide = [P.id_ids[i] for i in idx]
    items = [(s, False) for s in ide] + [(s, True) for s in ood]
    perm = rng.permutation(len(items))
    return [items[i][0] for i in perm], np.array([items[i][1] for i in perm], bool), {"n_unknown": int(len(chosen)), "n_ood": len(ood), "n_id": len(ide)}


def hash_name(name):
    import zlib

    return zlib.crc32(name.encode())


# ---------------------------------------------------------------------------------------------- E (U1 design space)
def e_pools(P):
    """ID pools (evaluation / long) and every pool image of the split's unknown classes (210 per class)."""
    pool = P.pool
    idset = set(P.id_wnids)
    ide = pool[(pool.role == "ideval") & pool.wnid.isin(idset)].sort_values(["idx_1k", "slot"])
    lng = pool[(pool.role == "long") & pool.wnid.isin(idset)].sort_values(["idx_1k", "slot"])
    held = pool[pool.wnid.isin(set(P.heldout))].sort_values(["idx_1k", "slot"])
    return ide, lng, {w: g.sample_id.tolist() for w, g in held.groupby("wnid")}


def compose(P, U, m, n_id, rng):
    """U unknown classes x m images; n_id in {4500 (5 per class), 18000 (20 per class), 72000 (evaluation + long)}."""
    ide, lng, by = e_pools(P)
    held = sorted(by)
    classes = [held[i] for i in np.sort(rng.permutation(len(held))[:U])]
    ood = {}
    for w in classes:
        imgs = by[w]
        assert len(imgs) >= m, (w, len(imgs))
        ood[w] = [imgs[i] for i in np.sort(rng.permutation(len(imgs))[:m])]
    if n_id == 18000:
        ids = ide.sample_id.tolist()
    elif n_id == 72000:
        ids = ide.sample_id.tolist() + lng.sample_id.tolist()
    else:
        per = n_id // len(P.id_wnids)
        ids = []
        for w, g in ide.groupby("wnid", sort=False):
            s = g.sample_id.tolist()
            ids += [s[i] for i in np.sort(rng.permutation(len(s))[:per])]
    assert len(ids) == n_id, (len(ids), n_id)
    return ids, ood


def arrange(ide, ood, pattern, rng, id_cls=None):
    """ide: ID ids; ood: {class: ids}. Returns (ids, flags). Patterns as registered (E2)."""
    n_ood = sum(len(v) for v in ood.values())
    N = len(ide) + n_ood
    U = len(ood)
    cls_list = [c for c in ood]
    cls_list = [cls_list[i] for i in rng.permutation(U)]
    t_id = rng.random(len(ide))
    t_ood = {}
    if pattern == "random":
        for c in cls_list:
            t_ood[c] = rng.random(len(ood[c]))
    elif pattern == "id_burst10":
        cl = np.asarray(id_cls)
        blocks = []
        for w in np.unique(cl):
            members = [ide[i] for i in np.flatnonzero(cl == w)]
            members = [members[i] for i in rng.permutation(len(members))]
            blocks += [members[lo:lo + 10] for lo in range(0, len(members), 10)]
        blocks = [blocks[i] for i in rng.permutation(len(blocks))]
        seq = [s for b in blocks for s in b]
        flat = [s for c in cls_list for s in ood[c]]
        flat = [flat[i] for i in rng.permutation(len(flat))]
        slots = set(np.sort(rng.choice(N, size=n_ood, replace=False)).tolist())
        it_id, it_ood = iter(seq), iter(flat)
        ids, flag = [], []
        for j in range(N):
            o = j in slots
            ids.append(next(it_ood) if o else next(it_id))
            flag.append(o)
        return ids, np.array(flag, bool)
    elif pattern == "ood_burst":
        for c in cls_list:
            t_ood[c] = rng.random() + 1e-9 * np.arange(len(ood[c]))
    elif pattern == "id_first":
        first = rng.permutation(len(ide))[:len(ide) // 2]
        frac = (len(ide) // 2) / N
        t_id = frac + (1 - frac) * rng.random(len(ide))
        t_id[first] = frac * rng.random(len(first))
        for c in cls_list:
            t_ood[c] = frac + (1 - frac) * rng.random(len(ood[c]))
    elif pattern == "ood_early":
        head = max(N // 3, n_ood)
        n_head_id = head - n_ood
        hid = rng.permutation(len(ide))[:n_head_id]
        t_id = head / N + (1 - head / N) * rng.random(len(ide))
        t_id[hid] = (head / N) * rng.random(len(hid))
        for c in cls_list:
            t_ood[c] = (head / N) * rng.random(len(ood[c]))
    elif pattern == "emerging":
        for j, c in enumerate(cls_list):
            s = 0.8 * j / U
            t_ood[c] = s + (1 - s) * rng.random(len(ood[c]))
    elif pattern == "two_visits":
        w = 0.01
        for c in cls_list:
            n = len(ood[c])
            a, b = rng.random() * (0.4 - w), 0.6 + rng.random() * (0.4 - w)
            t = np.r_[a + w * rng.random(n // 2), b + w * rng.random(n - n // 2)]
            t_ood[c] = t
    else:
        raise ValueError(pattern)
    ids = list(ide) + [s for c in cls_list for s in ood[c]]
    t = np.r_[t_id, np.concatenate([np.asarray(t_ood[c], float) for c in cls_list])]
    flag = np.r_[np.zeros(len(ide), bool), np.ones(n_ood, bool)]
    order = np.argsort(t, kind="stable")
    return [ids[i] for i in order], flag[order]
