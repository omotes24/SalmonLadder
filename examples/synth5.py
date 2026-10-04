"""Synthetic streams for code tests only (never used for selection or reported numbers).

Sibling structure: every group holds `n_id_per` ID classes and `n_ood_per` unknown classes that share a group mean;
samples add a shared anisotropic within-class covariance; two views are different noisy projections of one latent."""
import numpy as np


def l2n(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def make(seed=0, groups=20, n_id_per=3, n_ood_per=1, latent=24, dims=(48, 64), n_sup=12, n_cal=4, n_id=20, n_ood=50,
         group_scale=1.0, class_scale=0.45, noise=0.55, far=0, batch=64, views=("B14", "L14"), view_noise=0.25):
    rng = np.random.default_rng(seed)
    gm = rng.normal(size=(groups, latent)) * group_scale
    id_mu = np.concatenate([gm[g] + rng.normal(size=(n_id_per, latent)) * class_scale for g in range(groups)])
    ood_mu = np.concatenate([gm[g] + rng.normal(size=(n_ood_per, latent)) * class_scale for g in range(groups)])
    scale = np.exp(rng.normal(size=latent) * 0.5) * noise
    C, Co = len(id_mu), len(ood_mu)

    def draw(mu, n):
        return mu[:, None, :] + rng.normal(size=(len(mu), n, latent)) * scale

    lat = {"sup": draw(id_mu, n_sup), "cal": draw(id_mu, n_cal), "id": draw(id_mu, n_id), "ood": draw(ood_mu, n_ood)}
    if far:
        lat["far"] = rng.normal(size=(1, far, latent)) * 2.0
    n_stream = C * n_id + Co * n_ood + far
    order = rng.permutation(n_stream)
    is_ood = np.r_[np.zeros(C * n_id, bool), np.ones(Co * n_ood + far, bool)][order]
    cls = np.r_[np.repeat(np.arange(C), n_id), C + np.repeat(np.arange(Co), n_ood), np.full(far, -1)][order]
    out = {"is_ood": is_ood, "cls": cls, "bidx": np.arange(n_stream) // batch, "C": C, "views": {}}
    for v, D in zip(views, dims):
        P = rng.normal(size=(latent, D)) / np.sqrt(latent)

        def proj(z):
            return l2n(z @ P + rng.normal(size=z.shape[:-1] + (D,)) * view_noise)

        st = np.concatenate([lat["id"].reshape(-1, latent), lat["ood"].reshape(-1, latent)] +
                            ([lat["far"].reshape(-1, latent)] if far else []))
        out["views"][v] = {"sup": proj(lat["sup"]), "cal": proj(lat["cal"]).reshape(-1, D), "sf": proj(st)[order]}
    # candidate classes: nearest prototypes in a third noisy projection (stands in for the CLIP zero-shot top-K)
    P = rng.normal(size=(latent, 32)) / np.sqrt(latent)
    mu = l2n(id_mu @ P)
    k = min(20, C)
    st = np.concatenate([lat["id"].reshape(-1, latent), lat["ood"].reshape(-1, latent)] +
                        ([lat["far"].reshape(-1, latent)] if far else []))[order]
    out["cand_sf"] = np.argsort(-(l2n(st @ P + rng.normal(size=(len(st), 32)) * 0.3) @ mu.T), axis=1)[:, :k]
    cal = lat["cal"].reshape(-1, latent)
    out["cand_cal"] = np.argsort(-(l2n(cal @ P + rng.normal(size=(len(cal), 32)) * 0.3) @ mu.T), axis=1)[:, :k]
    return out
