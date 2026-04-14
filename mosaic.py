"""
convert_mosaic.py
-----------------
Loads Flimdejong/how2sign-3s (all splits), converts every video to a
per-frame mosaic (left-hand | face | right-hand) at TARGET_FPS, and
pushes the result to TimWijma/how2sign-3s-mosaic on HuggingFace.

Usage
-----
    pip install datasets huggingface_hub mediapipe torchcodec Pillow numpy tqdm
    huggingface-cli login          # needs write access to TimWijma/*
    python convert_mosaic.py

Tune the constants at the top of the file to your hardware.

Root cause of "Provided stream index=0 was not previously added"
----------------------------------------------------------------
torchcodec's VideoDecoder is backed by a C++ object. When multiprocessing
pickles the object and reconstructs it in a worker, the stream state is lost
(add_video_stream() was never called there). Calling bytes() on a VideoDecoder
also fails because it iterates it via __getitem__, which hits the same error.

Fix: extract raw video bytes from the source IN THE MAIN PROCESS before
building the work items, so workers receive plain bytes — never a decoder.
A fresh decoder is created inside each worker from those bytes.
"""

import io
import time
import logging
import argparse
import traceback
import multiprocessing as mp
from pathlib import Path
import pickle

import numpy as np
import torch
import mediapipe as mediapipe_lib
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

# ---------------------------------------------------------------------------
# Constants — tune these
# ---------------------------------------------------------------------------
SRC_DATASET    = "Flimdejong/how2sign-3s"
DST_DATASET    = "TimWijma/how2sign-3s-mosaic"
ORIGINAL_FPS   = 30
TARGET_FPS     = 24
CROP_SIZE      = 112      # px — size of each mosaic tile (square)
FACE_MARGIN    = 0.35
HAND_MARGIN    = 0.45
N_WORKERS      = max(1, mp.cpu_count() - 1)
UPLOAD_BATCH   = 256      # rows per push_to_hub call
CHECKPOINT_DIR = Path("./checkpoints")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Video bytes extraction  (runs in the MAIN process only)
# ---------------------------------------------------------------------------

def extract_video_bytes(video_field) -> bytes:
    """
    Pull raw video bytes out of whatever object HuggingFace gives us.

    This MUST be called in the main process before handing rows to workers,
    because VideoDecoder objects are not safely picklable.

    Handles:
      - dict {"bytes": ..., "path": ...}  — standard HF video feature
      - torchcodec VideoDecoder           — read from its backing file/buffer
      - file-like object                  — seek(0) + read()
      - str / Path                        — read from disk
    """
    # --- dict (most common HF representation) ---
    if isinstance(video_field, dict):
        if video_field.get("bytes"):
            return video_field["bytes"]
        if video_field.get("path"):
            return Path(video_field["path"]).read_bytes()

    # --- torchcodec VideoDecoder ---
    # DO NOT call bytes() on it — that triggers __getitem__ which needs an
    # initialised stream and will throw the same RuntimeError we are fixing.
    try:
        from torchcodec.decoders import VideoDecoder as _VD
        if isinstance(video_field, _VD):
            # torchcodec stores the original source in a few possible places
            # depending on the version; try them in order.
            if hasattr(video_field, "_hf_encoded"):
                enc = video_field._hf_encoded
                if isinstance(enc, dict):
                    return enc.get("bytes") or Path(enc["path"]).read_bytes()
                return enc  # already bytes

            for attr in ("_reader", "_filename", "_path", "_source"):
                src = getattr(video_field, attr, None)
                if src is None:
                    continue
                if isinstance(src, (str, Path)):
                    return Path(src).read_bytes()
                if hasattr(src, "getvalue"):          # BytesIO
                    return src.getvalue()
                if hasattr(src, "read"):
                    src.seek(0)
                    return src.read()
            raise RuntimeError(
                "VideoDecoder found but could not locate its backing source. "
                f"Available attrs: {[a for a in dir(video_field) if not a.startswith('__')]}"
            )
    except ImportError:
        pass

    # --- file-like ---
    if hasattr(video_field, "read"):
        if hasattr(video_field, "seek"):
            video_field.seek(0)
        return video_field.read()

    # --- plain path string ---
    if isinstance(video_field, (str, Path)):
        return Path(video_field).read_bytes()

    raise TypeError(
        f"Cannot extract video bytes from {type(video_field)}. "
        "Please open a bug report with the dataset name."
    )


# ---------------------------------------------------------------------------
# Frame / crop helpers
# ---------------------------------------------------------------------------

def blank(size: int) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


def safe_crop_bbox(frame, x_min, x_max, y_min, y_max, margin, crop_size):
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    half = max((x_max - x_min), (y_max - y_min)) / 2.0 * (1 + margin)
    return safe_crop_center(frame, cx, cy, half, crop_size)


def safe_crop_center(frame, cx, cy, half_norm, crop_size):
    h, w = frame.shape[:2]
    x0 = max(0, int((cx - half_norm) * w))
    x1 = min(w, int((cx + half_norm) * w))
    y0 = max(0, int((cy - half_norm) * h))
    y1 = min(h, int((cy + half_norm) * h))
    if x0 >= x1 or y0 >= y1:
        return blank(crop_size)
    return np.array(
        Image.fromarray(frame[y0:y1, x0:x1]).resize((crop_size, crop_size), Image.BILINEAR)
    )


def process_frame(frame: np.ndarray, results) -> np.ndarray:
    """Build a [left_hand | face | right_hand] mosaic from one frame."""
    lm = results
    mp_pose = mediapipe_lib.solutions.pose.PoseLandmark

    nose_x = nose_y = shoulder_w = None
    if lm.pose_landmarks:
        pl = lm.pose_landmarks.landmark
        nose_x, nose_y = pl[mp_pose.NOSE].x, pl[mp_pose.NOSE].y
        ls, rs = pl[mp_pose.LEFT_SHOULDER], pl[mp_pose.RIGHT_SHOULDER]
        shoulder_w = abs(ls.x - rs.x) + 1e-6
    if shoulder_w is None:
        shoulder_w = 0.3

    if lm.face_landmarks:
        xs = [p.x for p in lm.face_landmarks.landmark]
        ys = [p.y for p in lm.face_landmarks.landmark]
        face_crop = safe_crop_bbox(frame, min(xs), max(xs), min(ys), max(ys), FACE_MARGIN, CROP_SIZE)
    elif nose_y is not None:
        half_norm = max(0.10, shoulder_w * 0.55)
        face_crop = safe_crop_center(frame, nose_x, nose_y, half_norm, CROP_SIZE)
    else:
        face_crop = blank(CROP_SIZE)

    def hand_crop(hand_lm):
        if hand_lm is None:
            return blank(CROP_SIZE)
        xs = [p.x for p in hand_lm.landmark]
        ys = [p.y for p in hand_lm.landmark]
        return safe_crop_bbox(frame, min(xs), max(xs), min(ys), max(ys), HAND_MARGIN, CROP_SIZE)

    return np.concatenate(
        [hand_crop(lm.left_hand_landmarks), face_crop, hand_crop(lm.right_hand_landmarks)],
        axis=1,
    )


def frames_from_bytes(video_bytes: bytes,
                      original_fps: float = ORIGINAL_FPS,
                      target_fps: float = TARGET_FPS) -> list:
    """
    Decode video from raw bytes inside a worker process.
    Creates a fresh VideoDecoder + adds the stream — safe because we are
    never crossing a process boundary with the decoder object itself.
    """
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(io.BytesIO(video_bytes), stream_index=0)

    total_frames = decoder.metadata.num_frames
    duration_sec = total_frames / original_fps
    n_frames = max(4, int(float(target_fps) * duration_sec))
    indices = torch.linspace(0, total_frames - 1, n_frames).long().tolist()

    frames = decoder.get_frames_at(indices).data  # (T, C, H, W)
    frames = frames.permute(0, 2, 3, 1)           # (T, H, W, C)
    # frames = frames[:, :, 128:1152, :]            # horizontal crop
    return [f.numpy() for f in frames]


def encode_gif(mosaic_frames: list, fps: float) -> bytes:
    duration_ms = int(1000 / fps)
    pil_frames = [Image.fromarray(f) for f in mosaic_frames]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf, format="GIF", save_all=True,
        append_images=pil_frames[1:], duration=duration_ms, loop=0,
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------

def _worker_init():
    global _holistic
    _holistic = mediapipe_lib.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
    )


def _process_one(item):
    """
    item = (original_row_index, row_dict_with_video_bytes)

    row_dict has a "_video_bytes" key (plain bytes) instead of "video"
    (VideoDecoder), so it is safe to pickle.
    """
    idx, row = item
    try:
        np_frames = frames_from_bytes(row["_video_bytes"])

        mosaic_frames = []
        for f in np_frames:
            rgb = f if f.shape[2] == 3 else f[:, :, :3]
            mosaic_frames.append(process_frame(rgb, _holistic.process(rgb)))

        gif_bytes = encode_gif(mosaic_frames, TARGET_FPS)

        out = {k: v for k, v in row.items() if k != "_video_bytes"}
        out["mosaic_gif"] = gif_bytes
        out["n_frames"]   = len(mosaic_frames)
        out["fps"]        = float(TARGET_FPS)
        return idx, out

    except Exception:
        log.warning("Row %d failed, skipping:\n%s", idx, traceback.format_exc())
        return idx, None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_path(split: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"{split}.done"


def _load_done(split: str) -> set:
    p = _ckpt_path(split)
    if not p.exists():
        return set()
    return {int(x) for x in p.read_text().split() if x.strip()}


def _mark_done(split: str, indices: list):
    with _ckpt_path(split).open("a") as f:
        f.write("\n".join(str(i) for i in indices) + "\n")


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def process_split(split_name: str, dataset) -> list:
    results_path = CHECKPOINT_DIR / f"{split_name}.results.pkl"
    if results_path.exists():
        log.info("[%s] Loading results from disk …", split_name)
        with open(results_path, "rb") as f:
            return pickle.load(f)

    done = _load_done(split_name)

    log.info("[%s] Extracting video bytes in main process …", split_name)
    items = []
    for i, row in enumerate(tqdm(dataset, desc=f"{split_name}/extract", unit="row")):
        if i in done:
            continue
        try:
            video_bytes = extract_video_bytes(row["video"])
        except Exception:
            log.warning("Row %d: byte extraction failed, skipping:\n%s",
                        i, traceback.format_exc())
            continue
        # Replace the video field with plain bytes — safe to pickle
        sanitized = {k: v for k, v in row.items() if k != "video"}
        sanitized["_video_bytes"] = video_bytes
        items.append((i, sanitized))

    if not items:
        log.info("[%s] All rows already done. Skipping.", split_name)
        return []

    log.info("[%s] Processing %d / %d rows with %d workers …",
             split_name, len(items), len(dataset), N_WORKERS)

    results = []
    pending_ckpt = []

    with mp.Pool(processes=N_WORKERS, initializer=_worker_init) as pool:
        with tqdm(total=len(items), desc=split_name, unit="vid") as pbar:
            for idx, result in pool.imap_unordered(_process_one, items, chunksize=4):
                if result is not None:
                    results.append(result)
                pending_ckpt.append(idx)
                if len(pending_ckpt) >= 64:
                    _mark_done(split_name, pending_ckpt)
                    pending_ckpt.clear()
                pbar.update(1)

    if pending_ckpt:
        _mark_done(split_name, pending_ckpt)

    log.info("[%s] Saving results to disk …", split_name)
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    return results


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_split(split_name: str, rows: list, api: HfApi, token: str = None):
    if not rows:
        log.info("[%s] Nothing to upload.", split_name)
        return
    log.info("[%s] Uploading %d rows to %s …", split_name, len(rows), DST_DATASET)
    Dataset.from_list(rows).push_to_hub(
        DST_DATASET, 
        split=split_name, 
        token=token,
        )
    log.info("[%s] Upload complete.", split_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global N_WORKERS

    parser = argparse.ArgumentParser(description="Convert how2sign-3s to mosaic dataset.")
    parser.add_argument("--splits", nargs="*", default=None,
                        help="Splits to process (default: all)")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"Worker processes (default: {N_WORKERS})")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace write token")

    args = parser.parse_args()
    N_WORKERS = args.workers

    log.info("Loading source dataset …")
    src = load_dataset(SRC_DATASET)

    splits = args.splits or list(src.keys())
    log.info("Splits to process: %s", splits)

    api = HfApi()
    api.create_repo(DST_DATASET, repo_type="dataset", exist_ok=True)

    for split in splits:
        if split not in src:
            log.warning("Split '%s' not found, skipping.", split)
            continue

        t0 = time.time()
        converted = process_split(split, src[split])
        log.info("[%s] Done in %.1f min. Got %d rows.",
                 split, (time.time() - t0) / 60, len(converted))

        upload_split(split, converted, api, token=args.token)

    log.info("All done → https://huggingface.co/datasets/%s", DST_DATASET)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()