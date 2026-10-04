"""Write selection_lock_p7.json (amendment 02): the candidate chosen on dev1 by the registered rule and confirmed on
dev2. Must run before any '@0' metric is computed on U1 / U4."""
import json
import sys

from p7common import P7, RESULTS, sha_file, utc

sel = json.loads((RESULTS / "a2_dev1_selection.json").read_text())
con = json.loads((RESULTS / "a2_dev2_confirm.json").read_text())
assert sel["selected"] and con["confirmed"] and sel["chosen"]["spec"] == con["spec"]
out = P7 / "selection_lock_p7.json"
if out.exists():
    sys.exit("lock exists; refusing to overwrite")
for f in ("an_lock_eval.json",):
    assert not (RESULTS / f).exists(), "U1 / U4 metrics of the candidate already exist"
lock = {"utc": utc(), "amendment": "amendment_02.json", "spec": con["spec"], "name": sel["chosen"]["name"],
        "meaning": "per view: one-sided memory g = d + max(0, 2 - (rho - med)/mad) with rho the distance to the nearest memory member admitted up to and including the current batch "
                   "(the image itself excluded), times the calibrated rank of the zero-start propagation; entrance, graph and thresholds as frozen v5",
        "dev1": {k: sel["chosen"]["d"][k] for k in ("near_AUROC", "near_FPR95", "far_FPR95", "nearT_FPR95", "F1")},
        "dev2": {k: con["difference"][k] for k in ("near_AUROC", "near_FPR95", "nearT_AUROC", "nearT_FPR95", "far_FPR95", "F1", "A1")},
        "code_sha256": {f: sha_file(P7 / "code" / f) for f in ("engine7.py", "run7.py", "dev7.py", "an7.py", "an_a_dev.py", "an_within.py")}}
out.write_text(json.dumps(lock, indent=1) + "\n")
(P7 / "selection_lock_p7.sha256").write_text(sha_file(out) + "  selection_lock_p7.json\n")
print("LOCKED", lock["name"], sha_file(out), lock["utc"])
