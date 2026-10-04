"""Write and freeze the Phase 3 pre-registration (r5/phase3/prereg_phase3.json) from the dev decisions.

v5 comes from r5/round2/decision.json (pre-registered round 2). Baseline hyper-parameters are the dev1 selections
(x TINS, mean near FPR95 over the 15 conditions): kNN / Mahalanobis++ (r5/summary/stageA.json, maha_ext.json, e1.json),
AdaNeg-type and OODD-type memories over the union of the original and the extended grids (r5/<dev1>/e1, e1_ext).
The file records the sha256 of every script that will read test data. Refuses to overwrite.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vins import r5  # noqa: E402

P3 = r5.R5 / "phase3"
CODE = ["scripts/p3_eval.py", "scripts/p3_static.py", "scripts/p3_extract.py", "scripts/p3_m2b.py",
        "scripts/p3_streams.py", "scripts/p3_cub.py", "scripts/p3_locoop.py", "scripts/p3_locoop_run.py",
        "scripts/p3_report.py", "scripts/r5_e1.py", "scripts/run_tins_test.py", "vins/r5.py", "vins/memory.py",
        "vins/dview.py"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dev1_best(prefixes):
    """lowest mean dev1 near FPR95 (x TINS) among keys starting with any prefix, over e1 and e1_ext."""
    vals = {}
    for sub in ("e1", "e1_ext"):
        for d in range(5):
            for sd in (123, 124, 125):
                f = r5.R5 / "dev1" / sub / f"draw{d}" / f"near_seed{sd}.json"
                if not f.exists():
                    continue
                for k, v in json.loads(f.read_text())["metrics"].items():
                    if any(k.startswith(p) for p in prefixes):
                        vals.setdefault(k, {}).setdefault((d, sd), v["FPR95"])
    full = {k: np.mean(list(v.values())) for k, v in vals.items() if len(v) == 15}
    best = min(full, key=full.get)
    return best, {k: float(x) for k, x in sorted(full.items(), key=lambda kv: kv[1])[:8]}


def main():
    out = P3 / "prereg_phase3.json"
    if out.exists():
        raise SystemExit("already frozen")
    dec = json.loads((r5.R5 / "round2" / "decision.json").read_text())
    v5 = dict(dec["v5"])
    th = {2: [0.40, 0.30, 0.10], 1: [0.30, 0.20, 0.10191613435745239]}
    v5["thresholds"] = th[int(v5["m"])]
    v4 = {"n0": 12, "K": 5, "m": 2, "gamma": 3.0, "lam": 0.9, "kg": 10, "thresholds": th[2]}
    ad, ad_top = dev1_best(["s_adaneg_"])
    od, od_top = dev1_best(["s_oodd_"])
    tau, lam = float(ad.split("_t")[1].split("_l")[0]), float(ad.split("_l")[-1])
    K = int(od.split("_K")[1])
    maha = json.loads((r5.R5 / "summary" / "maha_ext.json").read_text())
    e1 = json.loads((r5.R5 / "summary" / "e1.json").read_text())["selection_dev1"]
    stage = json.loads((r5.R5 / "summary" / "stageA.json").read_text())["baseline_selection_dev1"]
    base = {"knn_k": int(stage["s"]["knn"]), "maha_lam": float(maha["s"]["best"]),
            "knn_k_l14": int(e1["sL14_knn"].split("knn")[1]), "maha_lam_l14": float(maha["sL14"]["best"]),
            "knnF_k": int(e1["s_knnF"].split("knnF")[1]), "mahaF_lam": float(e1["s_mahaF"].split("mahaF")[1]),
            "adaneg_tau": tau, "adaneg_lam": lam, "oodd_K": K,
            "dev1_selection_evidence": {"adaneg_top": ad_top, "oodd_top": od_top}}
    pr = {
        "name": "R5 Phase 3: final evaluation on the test data, run once",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": {"round2_decision_sha256": sha(r5.R5 / "round2" / "decision.json"), "round2_verdict": dec["verdict"],
                       "prereg_phase2_round2_sha256": sha(r5.R5 / "prereg_phase2_round2.json")},
        "configs": {"v4": v4, "v5": v5},
        "baselines": base,
        "shots": "the TINS 16-shot list used throughout (12 support + 4 calibration per class); M2b replaces the 4 "
                 "calibration shots by 4 OpenOOD ID-val images per class (seeded), support unchanged",
        "benchmarks": {
            "F1_openood": "OpenOOD v1.5 ImageNet-1K; one stream per OOD set (ID = test_imagenet); near = SSB-hard, NINCO; "
                          "far = iNaturalist, Textures, OpenImage-O; order seeds 123..127; TINS from the unchanged-upstream runs",
            "F1_fourood": "Four-OOD (ID = ImageNet-1K val 50,000; iNaturalist, SUN, Places, DTD); order seeds 123..125",
            "E2_E4_test": "p3_streams.py specs (OOD ratio, classes x images, ID bursts, ID first, pseudo-bursts, mixed OOD, "
                          "delayed classes), order seeds 123..125 each",
            "F2_cub": "SSB CUB (known 100 / unknown Easy, Medium, Hard), 16 train shots per known class, seeds 123..125",
            "F3_camera_trap": "not run in this registration (time / camera metadata of the felid data not available on hades)",
        },
        "methods": ["tins", "v4", "v4_lp", "v5", "v5_lp", "v5_static", "v5_mem", "v5_vis", "reprise_clip", "adaneg", "oodd",
                    "knn16_2view", "maha16_2view", "knn16_l14", "maha16_l14", "knnF_l14", "mahaF_l14", "protoF_l14",
                    "v5_m2b (OpenOOD)", "v5_dino1", "v5_l14only", "E5: v5_noK, v5_mem_pure, v5_mem_allood, v5_lp_keep_id, "
                    "v5_lp_keep_ood (oracles are diagnostics)", "LoCoOp MCM / GL-MCM (official 16-shot checkpoints, seeds 1-3)",
                    "GalLoP: not run (no public checkpoint; to be added later by the user)"],
        "metrics": "AUROC and FPR95 (upstream get_measures) per stream; OpenOOD near / far = means over their sets; "
                   "Four-OOD average; within-batch AUROC; ID admission and Pr(p_t <= 0.1 | ID)",
        "hypotheses": {
            "H1": "entrance + memory add over LP on test: v5 - v5_lp near FPR95 < 0 (order-level paired 95% t interval, OpenOOD)",
            "H2": "v5 vs v4 near FPR95 (the round-2 changes transfer to test)",
            "H3": "v5 vs TINS and vs the strongest same-feature baseline (maha16_2view) near FPR95",
            "H4": "M2b: ID-val calibration brings Pr(p_t <= 0.1 | ID) and the ID admission rate close to nominal",
            "H5": "SSB-hard contamination: v5_dino1 vs v5 on SSB-hard relative to NINCO",
        },
        "rules": "nothing is selected or tuned on these results; every number is reported; the test data are read only "
                 "by the scripts listed under code, and only after this file is frozen (unseal token = its sha256)",
        "code": {p: sha(ROOT / p) for p in CODE},
        "sha256_of_this_file_is_logged": True,
    }
    blob = json.dumps(pr, indent=1) + "\n"
    out.write_text(blob)
    out.chmod(0o444)
    digest = hashlib.sha256(blob.encode()).hexdigest()
    (P3 / "prereg_phase3.sha256").write_text(digest + "\n")
    print(json.dumps({"sha256": digest, "v5": v5, "baselines": {k: v for k, v in base.items() if k != "dev1_selection_evidence"}}))


if __name__ == "__main__":
    main()
