"""Fixed paths, seeds and constants for the VINS go/no-go experiment (hades)."""
import os
from pathlib import Path

HOME = Path.home()
WORK = Path(os.environ.get("VINS_WORK", HOME / "vins_gonogo_20260925"))

# Upstream TINS (pinned) + our logging-hook commit
TINS_DIR = WORK / "tins"
TINS_COMMIT = "194759d716b27534bf2e8eeb0d71f5c4f0dabc40"

# Inputs reused from Codex's verified TINS reproduction (read-only)
CODEX_TINS = HOME / "ood_large_best_20260923" / "tins_20260925"
CLIP_WEIGHTS_DIR = CODEX_TINS / "weights"                       # ViT-B-16.pt
CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
PROTO_LIST_SRC = CODEX_TINS / "prototype_train16.txt"          # TINS 16-shot list (materialized fallback)
PROTO_RESOLUTION_SRC = CODEX_TINS / "prototype_overlap_resolution.json"
SEAL_MANIFEST = HOME / "ood_large_best_20260923" / "manifest.jsonl"  # sha256 of every sealed image

# Data
IMAGENET_ROOT = Path("/home/omote/datasets/openood_official/images_largescale/imagenet_1k")  # has train/
OPENOOD_DATA = Path("/home/omote/openood_simple_sota/data")
OPENOOD_IMGLIST_DIR = OPENOOD_DATA / "benchmark_imglist" / "imagenet"
OPENOOD_IMAGES = OPENOOD_DATA / "images_largescale"
FAR_DEV_LIST = OPENOOD_IMGLIST_DIR / "val_openimage_o.txt"
# Sealed splits: never read as images. Only sealing.py may reference these names.
SEALED_LIST_FILES = (
    "test_imagenet.txt", "test_ssb_hard.txt", "test_ninco.txt", "test_inaturalist.txt",
    "test_textures.txt", "test_openimage_o.txt", "val_imagenet.txt",
)

# Models
DINO_HUB = HOME / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
DINO_CKPT = HOME / ".cache" / "torch" / "hub" / "checkpoints" / "dinov2_vitb14_pretrain.pth"
DINO_SHA256 = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
NLTK_DATA = HOME / "nltk_data"

# Layout inside WORK
SPLITS_DIR = WORK / "splits"
INPUTS_DIR = WORK / "inputs"
FEATURES_DIR = WORK / "features"
RUNS_DIR = WORK / "runs"
REPORTS_DIR = WORK / "reports"
DATASETS_DIR = WORK / "datasets"          # datasets/ImageNet -> IMAGENET_ROOT (TINS --root-dir)
CRITERIA_PATH = WORK / "criteria.json"

# Spec constants
SPLIT_SEED = int(os.environ.get("VINS_SPLIT_SEED", "0"))
# dev2 (confirmation split): classes held out in another dev and images it used are excluded
EXCLUDE_HELDOUT_FROM = os.environ.get("VINS_EXCLUDE_HELDOUT_FROM")   # path to that dev's heldout.json
EXCLUDE_SAMPLES_FROM = os.environ.get("VINS_EXCLUDE_SAMPLES_FROM")   # path to that dev's samples.parquet
ORDER_SEEDS = (123, 124, 125)
N_HELDOUT = 100
N_SHOT = 16
N_SUPPORT = 12
N_CALIB = 4
N_ID_DEV_PER_CLASS = 20
N_NEAR_PER_CLASS = 50
K_TOP = 5
N0 = 12
MAD_SCALE = 1.4826
M_LIST = (1, 2)
EPS_LIST = (0.005, 0.01, 0.02)
BETA = 0.3
BATCH = 256
FEATURES = ("clip", "dino")

# TINS hyper-parameters: identical to Codex's reproduction (arguments.json), paths replaced.
def tins_argv(cache_dir, name, stream_seed=123):
    return [
        "eval_tins_w_init.py",
        "--eval-protocol", "four_ood",            # label only; eval setup is never called
        "--root-dir", str(DATASETS_DIR),
        "--train-imglist", str(INPUTS_DIR / "prototype_train16.txt"),
        "--wordnet-dir", str(TINS_DIR / "txtfiles"),
        "--cache-dir", str(cache_dir),
        "--name", name,
        "--gpu", "0",
        "--CLIP_ckpt", "ViT-B/16",
        "--batch-size", str(BATCH),
        "--prototype-batch-size", "256",
        "--text-batch-size", "1000",
        "--seed", "0",
        "--stream-seed", str(stream_seed),
        "--inversion-steps", "30",
        "--inversion-reg-lambda", "0.3",
        "--ood-threshold", str(BETA),
        "--group-num", "5",
        "--ood-number", "2000",
        "--extra-text-length", "2000",
        "--bank-buffer-size", "2000",
        "--use-buffer",
        "--no-image-feature-cache",
    ]
