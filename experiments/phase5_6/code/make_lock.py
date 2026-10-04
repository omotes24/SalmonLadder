"""Write the Phase 5 selection lock (stage 1) and its sha256. Refuses to overwrite."""
import json
import sys
import time
from pathlib import Path

from common import sha_file

P5 = Path("/home/omote/reprise_p5_20261002")
LOCK = P5 / "selection_lock_p5.json"
if LOCK.exists():
    raise SystemExit("lock exists")
REF = [10, 1.0, 1.0, 0.9]
SET = "IB0.2s<B0.5s"
lock = {
    "locked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "stage": 1,
    "name": "REPRISE two-sided (frozen memory x set memory x support mass x seed mass; two rounds of Storey-BH self-labelling, q = 0.5 then 0.2)",
    "prereg_sha256": sha_file(P5 / "code" / "prereg_p5.json"),
    "views": [["B14", "raw", 0.0], ["L14", "raw", 0.0]],
    "spec": {"cfgs": [REF], "ref": REF, "neg": [], "classcond": False,
             "dyn": [{"kind": "bh", "q": 0.5, "storey": True}, {"kind": "iter", "base": "B0.5s", "q": 0.2, "storey": True}]},
    "score": [["M0", 1.0], [SET, 1.0], ["lp|k10g1b1l0.9", 1.0], [f"neg:{SET}", 1.0]],
    "score_rule": "sum over the two views of the log p-values; x TINS / x MCM adds the log base score (weight 1)",
    "unchanged_from_v5": {"n0": 48, "K": 20, "m": 1, "entrance_thresholds": [0.3, 0.2, 0.10191613435745239],
                          "graph": {"k": 10, "gamma": 1.0, "lambda": 0.9, "sweeps": 15}},
    "changes_from_v5": ["label propagation starts from 0 at every batch (was: warm start)",
                        "seed set: earlier stream images rejected by Storey-BH (q = 0.5) on the joint conformal p of the current support mass, "
                        "then a second Storey-BH round (q = 0.2) on support mass x mass received from the first set",
                        "two new factors per view: memory p against the seed set (nearest member) and the rank of the mass propagated from the seed set"],
    "dev1_selection": {"standalone_near": {"AUROC": 95.06, "FPR95": 22.04, "dFPR95_vs_frozen": [-7.17, -7.46, -6.89]},
                       "standalone_far": {"FPR95": 16.10, "dFPR95_vs_frozen": 0.07},
                       "xTINS_near": {"FPR95": 22.11, "dFPR95_vs_frozen": [-7.55, -7.80, -7.30]},
                       "frozen": {"standalone_near": [93.13, 29.21], "standalone_far": [96.82, 16.03], "xTINS_near": [93.05, 29.65]},
                       "rule": "lowest standalone near FPR95 among candidates with far <= frozen + 0.5 and x TINS near <= frozen; the two-round set "
                               "was preferred to the three-round set IB0.2s<IB0.25s<B0.5s (21.98) as the simpler candidate within 0.3 points"},
    "explored_on_dev1": "16 run configurations (base, tier2, center, white03/06/09, clip3, clip3c, fuse, fuse3, fuseall, r2-r7), about 600 score combinations; "
                        "logs under logs/, analyses in the bridge logs 467-485",
    "dev2_attempt": 1,
}
LOCK.write_text(json.dumps(lock, indent=1) + "\n")
(P5 / "selection_lock_p5.sha256").write_text(sha_file(LOCK) + "  selection_lock_p5.json\n")
LOCK.chmod(0o444)
print(sha_file(LOCK))
print(json.dumps(lock["spec"]))
