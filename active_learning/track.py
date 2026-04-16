"""
track.py — Run the trained model over a sequence of TEM images, produce masks,
track dislocations across frames, and output statistics.

Tracking design:
  - Centroids and equivalent diameters are computed from the binary mask via
    skimage regionprops (NOT predicted by the U-Net).  The mask encodes full
    spatial extent so regionprops centroids are more accurate than any point
    prediction head, and equivalent_diameter falls out for free.

  - Diameter gate: a detection can only be matched to a track whose last
    known (or predicted) centroid is within one diameter of the detection.
    Cost = distance / diameter; gate threshold = 1.0.  This prevents the
    "long jump" problem when two dislocations are close together.

  - Candidate → Confirmed: a track must be associated for 5 consecutive
    frames before it appears in output.  Suppresses single-frame flicker.

  - Grace period (3 frames): a missed track is kept alive with its last
    known velocity extrapolated forward.  Handles brief false negatives.

  - Graveyard (40 frames): after the grace period expires the track is
    moved to a graveyard and can still be revived if a detection reappears
    within its diameter gate.  Handles longer contrast dropouts.

Usage:
    python active_learning/track.py
    python active_learning/track.py --image-dir active_learning_data/raw_data/ --threshold 0.4 --out-dir results/
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from math import pi, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from skimage.measure import label as sk_label, regionprops

sys.path.insert(0, str(Path(__file__).parent))

from model import ResAttentionUNet
from patch_utils import extract_patches, merge_patches
from geometry_utils import min_dist_to_grain_boundaries

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATCH_SIZE      = 128
CHECKPOINT_PATH = Path("active_learning_data/checkpoints/model.pt")
IMAGE_EXTS      = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

MIN_AREA        = 5      # pixels — smaller regions ignored
MIN_ROUNDNESS   = 0.5    # 4πA/p² filter before tracking
CONFIRM_AGE     = 5      # consecutive hits before a track is "confirmed"
GRACE_PERIOD    = 3      # invisible frames before a track leaves active pool
GRAVEYARD_AGE   = 40     # frames a dead track stays revivable


# ---------------------------------------------------------------------------
# Model loading + patch inference  (unchanged)
# ---------------------------------------------------------------------------

def load_model(device: str) -> ResAttentionUNet:
    if not CHECKPOINT_PATH.exists():
        print(f"No checkpoint found at {CHECKPOINT_PATH}.")
        print("Train first:  python active_learning/main.py --train")
        sys.exit(1)
    model = ResAttentionUNet(n_channels=1, n_classes=1, dropout_rate=0.3)
    ckpt  = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"Loaded checkpoint — round {ckpt.get('round','?')}, "
          f"{ckpt.get('num_verified','?')} verified samples.")
    return model


def predict_mask(
    model: ResAttentionUNet,
    img_gray: np.ndarray,
    device: str,
    threshold: float = 0.5,
    n_passes: int = 1,
) -> np.ndarray:
    """Patch → stitch → threshold.  Returns uint8 binary mask (255 = dislocation)."""
    h, w    = img_gray.shape
    patches = extract_patches(img_gray,
                              patch_size=(PATCH_SIZE, PATCH_SIZE),
                              stride=(PATCH_SIZE, PATCH_SIZE))
    model.eval()
    if n_passes > 1:
        model.enable_dropout()

    prob_patches = []
    with torch.no_grad():
        for patch_img, x, y in patches:
            t = (torch.from_numpy(patch_img).float()
                 .unsqueeze(0).unsqueeze(0) / 255.0).to(device)
            preds = [torch.sigmoid(model(t)) for _ in range(n_passes)]
            mean_p = torch.stack(preds).mean(dim=0).squeeze().cpu().numpy()
            prob_patches.append((mean_p, x, y))

    prob_map = merge_patches(prob_patches, (h, w)).astype(np.float32)
    return (prob_map >= threshold).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# CentroidExtractor
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    label:    int
    x:        float          # centroid column
    y:        float          # centroid row
    diameter: float          # equivalent circle diameter (px)
    area:     int
    bbox:     Tuple[int, int, int, int]   # row, col, h, w  (skimage convention)
    contour:  np.ndarray


class CentroidExtractor:
    """
    Extract dislocation centroids and equivalent diameters from a binary mask.

    Uses skimage.measure.label + regionprops so that centroid is computed from
    the actual pixel region (moment-based) and equivalent_diameter = 2√(A/π)
    is available directly — both needed for the diameter gate.
    """

    def __init__(self, min_area: int = MIN_AREA, min_roundness: float = MIN_ROUNDNESS):
        self.min_area      = min_area
        self.min_roundness = min_roundness

    def extract(self, mask: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """
        Args:
            mask: uint8 binary mask (255 = foreground).
        Returns:
            detections: list of Detection objects.
            label_map:  int32 array where each blob has a unique integer label.
        """
        binary    = (mask > 128).astype(np.uint8)
        label_map = sk_label(binary, connectivity=2).astype(np.int32)
        props     = regionprops(label_map)

        detections: List[Detection] = []
        for prop in props:
            area = prop.area
            if area < self.min_area:
                continue

            # Roundness filter using perimeter from skimage
            perim = prop.perimeter
            if perim > 0:
                roundness = 4.0 * pi * area / (perim * perim)
                if roundness < self.min_roundness:
                    continue

            # regionprops centroid is (row, col)
            cy, cx = prop.centroid
            diam   = prop.equivalent_diameter_area  # 2 * sqrt(area / pi)

            # Get contour for visualisation
            blob_mask = (label_map == prop.label).astype(np.uint8)
            cnts, _   = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            contour = max(cnts, key=cv2.contourArea) if cnts else np.array([])

            # bbox: (min_row, min_col, max_row, max_col) → convert to (x, y, w, h)
            r0, c0, r1, c1 = prop.bbox
            detections.append(Detection(
                label    = int(prop.label),
                x        = float(cx),
                y        = float(cy),
                diameter = float(diam),
                area     = int(area),
                bbox     = (int(c0), int(r0), int(c1 - c0), int(r1 - r0)),
                contour  = contour,
            ))

        return detections, label_map


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

CANDIDATE = "candidate"
CONFIRMED = "confirmed"


class Track:
    """State for a single dislocation track."""

    def __init__(self, track_id: int, det: Detection, frame_idx: int):
        self.track_id         = track_id
        self.status           = CANDIDATE
        self.consecutive_hits = 1
        self.invisible_count  = 0
        self.diameter         = det.diameter    # updated on each match
        self._pos             = (det.x, det.y)
        self._velocity        = (0.0, 0.0)      # px/frame, updated after ≥2 hits
        # history: (frame_idx, x, y, vx, vy)
        self.history: List[Tuple] = [(frame_idx, det.x, det.y, 0.0, 0.0)]

    def predict_pos(self) -> Tuple[float, float]:
        """Extrapolate one frame forward using last known velocity."""
        return (self._pos[0] + self._velocity[0],
                self._pos[1] + self._velocity[1])

    def update(self, det: Detection, frame_idx: int) -> None:
        """Record a matched detection."""
        prev_x, prev_y = self._pos
        vx = det.x - prev_x
        vy = det.y - prev_y
        self._pos             = (det.x, det.y)
        self._velocity        = (vx, vy)
        self.diameter         = det.diameter
        self.invisible_count  = 0
        self.consecutive_hits += 1
        if self.status == CANDIDATE and self.consecutive_hits >= CONFIRM_AGE:
            self.status = CONFIRMED
        self.history.append((frame_idx, det.x, det.y, vx, vy))

    def mark_invisible(self) -> None:
        """Advance one frame without a detection; extrapolate position."""
        px, py        = self.predict_pos()
        self._pos     = (px, py)
        # velocity decays slightly so prediction doesn't drift far
        self._velocity = (self._velocity[0] * 0.5, self._velocity[1] * 0.5)
        self.invisible_count  += 1
        self.consecutive_hits  = 0   # reset confirmation streak


# ---------------------------------------------------------------------------
# VideoTracker
# ---------------------------------------------------------------------------

class VideoTracker:
    """
    Spatio-temporal tracker using a diameter-gated Hungarian cost matrix.

    Matching stages (per frame):
      1. Active tracks  ← new detections  (diameter-gated)
      2. Graveyard      ← remaining unmatched detections  (diameter-gated revival)
      3. Remaining unmatched detections → new Candidate tracks
      4. Unmatched active tracks → mark_invisible or move to graveyard
      5. Expired graveyard entries purged
    """

    def __init__(
        self,
        confirm_age:   int = CONFIRM_AGE,
        grace_period:  int = GRACE_PERIOD,
        graveyard_age: int = GRAVEYARD_AGE,
    ):
        self.confirm_age   = confirm_age
        self.grace_period  = grace_period
        self.graveyard_age = graveyard_age

        self.active:              Dict[int, Track] = {}
        self.graveyard:           Dict[int, Track] = {}
        self.graveyard_last_seen: Dict[int, int]   = {}
        self.next_id = 1

        # All confirmed records accumulated across frames
        self._records: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Detection],
        frame_idx:  int,
        grain_boundaries: Optional[List] = None,
    ) -> Tuple[List[Detection], List[Dict]]:
        """
        Process one frame.

        Returns:
            confirmed_dets:  Detection objects for confirmed tracks only
                             (for visualisation).
            new_records:     CSV-ready dicts added this frame.
        """
        unmatched_det_indices = set(range(len(detections)))

        # --- Stage 1: match to active tracks ---
        if self.active and detections:
            matches, unmatched_det_indices = self._hungarian_match(
                detections, unmatched_det_indices, self.active
            )
            matched_track_ids = set()
            for det_idx, tid in matches.items():
                self.active[tid].update(detections[det_idx], frame_idx)
                matched_track_ids.add(tid)

            # Age out unmatched active tracks
            for tid in list(self.active.keys()):
                if tid not in matched_track_ids:
                    track = self.active[tid]
                    track.mark_invisible()
                    if track.invisible_count > self.grace_period:
                        self.graveyard[tid]           = self.active.pop(tid)
                        self.graveyard_last_seen[tid] = frame_idx

        # --- Stage 2: try to revive graveyard tracks ---
        if unmatched_det_indices and self.graveyard:
            gy_matches, unmatched_det_indices = self._hungarian_match(
                detections, unmatched_det_indices, self.graveyard
            )
            for det_idx, tid in gy_matches.items():
                track = self.graveyard.pop(tid)
                self.graveyard_last_seen.pop(tid, None)
                track.update(detections[det_idx], frame_idx)
                # Reset invisible state on revival
                track.invisible_count = 0
                self.active[tid] = track

        # --- Stage 3: new candidates for unmatched detections ---
        for det_idx in unmatched_det_indices:
            tid = self.next_id
            self.next_id += 1
            self.active[tid] = Track(tid, detections[det_idx], frame_idx)

        # --- Stage 4: purge stale graveyard entries ---
        expired = [
            tid for tid, ls in self.graveyard_last_seen.items()
            if (frame_idx - ls) > self.graveyard_age
        ]
        for tid in expired:
            self.graveyard.pop(tid, None)
            self.graveyard_last_seen.pop(tid, None)

        # --- Collect confirmed output for this frame ---
        confirmed_dets = []
        new_records    = []

        # Build a det lookup for confirmed tracks matched this frame
        det_by_track: Dict[int, Detection] = {}
        # Re-derive from history: any track whose last history entry is frame_idx
        active_det_map: Dict[int, Detection] = {}
        for det_idx, det in enumerate(detections):
            pass  # we need to reverse-map track → det

        # Simpler: iterate active tracks and find those updated this frame
        for tid, track in self.active.items():
            if track.status != CONFIRMED:
                continue
            if not track.history or track.history[-1][0] != frame_idx:
                continue
            # Find the matching detection by closest centroid
            best_det = None
            best_d   = float("inf")
            for det in detections:
                d = sqrt((det.x - track._pos[0]) ** 2 + (det.y - track._pos[1]) ** 2)
                if d < best_d:
                    best_d, best_det = d, det
            if best_det is not None and best_d <= track.diameter:
                confirmed_dets.append(best_det)

            _, x, y, vx, vy = track.history[-1]
            speed  = sqrt(vx ** 2 + vy ** 2)
            gb_dist = (min_dist_to_grain_boundaries((x, y), grain_boundaries)
                       if grain_boundaries else float("nan"))
            record = {
                "frame":         frame_idx,
                "track_id":      tid,
                "x":             round(x, 2),
                "y":             round(y, 2),
                "velocity":      round(speed, 4),
                "diameter":      round(track.diameter, 2),
                "area":          best_det.area if best_det else "",
                "dist_to_gb_px": gb_dist,
                "status":        track.status,
            }
            new_records.append(record)
            self._records.append(record)

        return confirmed_dets, new_records

    def get_all_records(self) -> List[Dict]:
        return list(self._records)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hungarian_match(
        self,
        detections: List[Detection],
        unmatched_indices: set,
        pool: Dict[int, "Track"],
    ) -> Tuple[Dict[int, int], set]:
        """
        Diameter-gated Hungarian matching.

        Cost = distance / track.diameter  (normalised so gate = 1.0).
        Cells where distance > diameter are set to inf.

        Returns (det_idx → track_id, remaining_unmatched_indices).
        """
        if not pool or not unmatched_indices:
            return {}, unmatched_indices

        det_list  = [detections[i] for i in sorted(unmatched_indices)]
        det_idx   = sorted(unmatched_indices)
        tids      = list(pool.keys())
        n_d, n_t  = len(det_list), len(tids)
        INF       = 1e9
        cost      = np.full((n_d, n_t), INF)

        for i, det in enumerate(det_list):
            for j, tid in enumerate(tids):
                track = pool[tid]
                px, py = track.predict_pos() if track.invisible_count > 0 else track._pos
                dist   = sqrt((det.x - px) ** 2 + (det.y - py) ** 2)
                diam   = max(track.diameter, 1.0)
                if dist <= diam:
                    cost[i, j] = dist / diam   # normalised cost in [0, 1]

        row_ind, col_ind = linear_sum_assignment(cost)
        matches:   Dict[int, int] = {}
        still_unmatched = set(unmatched_indices)

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < INF:
                orig_det_idx    = det_idx[r]
                matches[orig_det_idx] = tids[c]
                still_unmatched.discard(orig_det_idx)

        return matches, still_unmatched


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color_from_id(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.RandomState(track_id * 9973 + 17)
    return tuple(int(x) for x in rng.randint(60, 255, size=3))


def _extract_frame_index(path: Path, default_idx: int) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else default_idx


def _loop_stats(frame_idx: int, detections: List[Detection]) -> Dict:
    n = len(detections)
    if n == 0:
        return {"frame": frame_idx, "number_of_loops": 0,
                "average_loop_size": 0.0, "average_diameter": 0.0}
    areas     = np.array([d.area     for d in detections], dtype=np.float64)
    diameters = np.array([d.diameter for d in detections], dtype=np.float64)
    return {"frame": frame_idx, "number_of_loops": n,
            "average_loop_size": float(areas.mean()),
            "average_diameter":  float(diameters.mean())}


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _draw_overlay(img_gray: np.ndarray, detections: List[Detection]) -> np.ndarray:
    overlay  = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    contours = [d.contour for d in detections if len(d.contour) > 0]
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    return overlay


def _draw_tracked(
    label_map:  np.ndarray,
    detections: List[Detection],
    tracker:    VideoTracker,
    frame_idx:  int,
) -> np.ndarray:
    h, w   = label_map.shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    label_to_tid = {d.label: None for d in detections}

    # Map detection labels to confirmed track IDs
    for tid, track in tracker.active.items():
        if track.status != CONFIRMED:
            continue
        for det in detections:
            px, py = track._pos
            if abs(det.x - px) < 2 and abs(det.y - py) < 2:
                label_to_tid[det.label] = tid
                break

    for det in detections:
        tid = label_to_tid.get(det.label)
        if tid is None:
            continue
        color = _color_from_id(tid)
        canvas[label_map == det.label] = color
        cv2.putText(canvas, str(tid),
                    (int(det.x) - 6, int(det.y) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw trajectory lines for all confirmed tracks
    for tid, track in tracker.active.items():
        if track.status != CONFIRMED:
            continue
        pts = [(int(x), int(y)) for f, x, y, vx, vy in track.history if f <= frame_idx]
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)],
                          isClosed=False, color=_color_from_id(tid),
                          thickness=1, lineType=cv2.LINE_AA)
    return canvas


def _draw_colorized(label_map: np.ndarray, detections: List[Detection],
                    tracker: VideoTracker) -> np.ndarray:
    h, w   = label_map.shape
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for det in detections:
        # Find track ID for this detection
        for tid, track in tracker.active.items():
            if track.status != CONFIRMED:
                continue
            px, py = track._pos
            if abs(det.x - px) < 2 and abs(det.y - py) < 2:
                canvas[label_map == det.label] = _color_from_id(tid)
                break
    return canvas


# ---------------------------------------------------------------------------
# Per-track summary
# ---------------------------------------------------------------------------

def _build_track_summary(records: List[Dict]) -> List[Dict]:
    """
    Collapse the per-frame records into one summary row per track.

    Columns:
        track_id          — unique track identifier
        first_frame       — frame index when the track was first confirmed
        last_frame        — frame index of the last confirmed detection
        track_lifespan    — last_frame - first_frame  (frames)
        num_detections    — number of frames the track was detected
        total_path_px     — sum of frame-to-frame distances (cumulative travel)
        displacement_px   — straight-line distance from first to last position
        mean_diameter_px  — mean equivalent circle diameter across all detections
        mean_area_px2     — mean blob area in pixels²
        mean_velocity_px  — mean per-frame speed
        mean_dist_to_gb_px— mean distance to nearest grain boundary (nan if none)
    """
    if not records:
        return []

    from collections import defaultdict
    grouped: Dict[int, List[Dict]] = defaultdict(list)
    for r in records:
        grouped[r["track_id"]].append(r)

    summary = []
    for tid, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: r["frame"])

        frames     = [r["frame"]    for r in rows]
        xs         = [r["x"]        for r in rows]
        ys         = [r["y"]        for r in rows]
        diameters  = [r["diameter"] for r in rows]
        areas      = [r["area"]     for r in rows if r["area"] != ""]
        velocities = [r["velocity"] for r in rows]
        gb_dists   = [r["dist_to_gb_px"] for r in rows
                      if not (isinstance(r["dist_to_gb_px"], float)
                              and np.isnan(r["dist_to_gb_px"]))]

        # Total path: sum of step distances
        path = 0.0
        for i in range(1, len(rows)):
            path += sqrt((xs[i] - xs[i-1]) ** 2 + (ys[i] - ys[i-1]) ** 2)

        # Straight-line displacement
        displacement = sqrt((xs[-1] - xs[0]) ** 2 + (ys[-1] - ys[0]) ** 2)

        summary.append({
            "track_id":           tid,
            "first_frame":        frames[0],
            "last_frame":         frames[-1],
            "track_lifespan":     frames[-1] - frames[0],
            "num_detections":     len(rows),
            "total_path_px":      round(path, 2),
            "displacement_px":    round(displacement, 2),
            "mean_diameter_px":   round(float(np.mean(diameters)), 2),
            "mean_area_px2":      round(float(np.mean(areas)), 2) if areas else "",
            "mean_velocity_px":   round(float(np.mean(velocities)), 4),
            "mean_dist_to_gb_px": round(float(np.mean(gb_dists)), 2) if gb_dists else float("nan"),
        })

    return summary


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    image_dir:        Path,
    out_dir:          Path,
    model:            ResAttentionUNet,
    device:           str,
    threshold:        float = 0.5,
    n_passes:         int   = 1,
    grain_boundaries: Optional[List] = None,
    confirm_age:      int   = CONFIRM_AGE,
    grace_period:     int   = GRACE_PERIOD,
    graveyard_age:    int   = GRAVEYARD_AGE,
    min_area:         int   = MIN_AREA,
    min_roundness:    float = MIN_ROUNDNESS,
) -> None:
    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_paths:
        print(f"No images found in {image_dir}")
        return

    masks_dir    = out_dir / "masks"
    overlays_dir = out_dir / "overlays"
    tracks_dir   = out_dir / "tracks"
    color_dir    = out_dir / "colorized"
    for d in [masks_dir, overlays_dir, tracks_dir, color_dir]:
        d.mkdir(parents=True, exist_ok=True)

    extractor = CentroidExtractor(min_area=min_area, min_roundness=min_roundness)
    tracker   = VideoTracker(confirm_age=confirm_age, grace_period=grace_period,
                              graveyard_age=graveyard_age)

    loop_csv_path = out_dir / "loop_stats.csv"
    loop_file     = loop_csv_path.open("w", newline="")
    loop_writer   = csv.DictWriter(
        loop_file,
        fieldnames=["frame", "number_of_loops", "average_loop_size", "average_diameter"],
    )
    loop_writer.writeheader()

    print(f"Processing {len(image_paths)} images…")
    for idx, img_path in enumerate(image_paths):
        frame_idx = _extract_frame_index(img_path, idx)
        img_gray  = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            print(f"  Skipping unreadable file: {img_path.name}")
            continue

        # 1. Inference
        binary_mask = predict_mask(model, img_gray, device,
                                   threshold=threshold, n_passes=n_passes)

        # 2. Extract centroids + diameters from mask
        detections, label_map = extractor.extract(binary_mask)

        # 3. Track — returns only confirmed detections
        confirmed_dets, _ = tracker.update(detections, frame_idx, grain_boundaries)

        stem = img_path.stem
        cv2.imwrite(str(masks_dir    / f"{stem}_mask.png"),    binary_mask)
        cv2.imwrite(str(overlays_dir / f"{stem}_overlay.png"), _draw_overlay(img_gray, confirmed_dets))
        cv2.imwrite(str(tracks_dir   / f"{stem}_tracked.png"), _draw_tracked(label_map, confirmed_dets, tracker, frame_idx))
        cv2.imwrite(str(color_dir    / f"{stem}_color.png"),   _draw_colorized(label_map, confirmed_dets, tracker))

        loop_writer.writerow(_loop_stats(frame_idx, confirmed_dets))
        loop_file.flush()

        n_all       = len(detections)
        n_confirmed = len(confirmed_dets)
        n_active    = len(tracker.active)
        print(f"  [{idx+1}/{len(image_paths)}] {img_path.name} — "
              f"{n_all} detected, {n_confirmed} confirmed, {n_active} active tracks")

    # Write full per-detection tracks CSV (confirmed only)
    tracks_csv = out_dir / "tracks.csv"
    records    = tracker.get_all_records()
    if records:
        with open(tracks_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    # Write per-track summary CSV
    summary_csv = out_dir / "track_summary.csv"
    summary_rows = _build_track_summary(records)
    if summary_rows:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    loop_file.close()
    unique_ids = len(summary_rows)
    print(f"\nDone.  {unique_ids} confirmed track(s) total.")
    print(f"  tracks.csv       → {tracks_csv}")
    print(f"  track_summary.csv→ {summary_csv}")
    print(f"  loop_stats.csv   → {loop_csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained model on TEM image sequence, track dislocations, output stats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-dir",     type=Path,
                        default=Path("active_learning_data/raw_data"),
                        help="Directory of input images (default: active_learning_data/raw_data).")
    parser.add_argument("--out-dir",       type=Path,  default=Path("active_learning_data/tracking"))
    parser.add_argument("--threshold",     type=float, default=0.5,
                        help="Probability threshold for dislocation vs background.")
    parser.add_argument("--passes",        type=int,   default=1,
                        help="MC Dropout passes per patch (1=fast, >1=smoother).")
    parser.add_argument("--gb-path",       type=Path,  default=None,
                        help="Path to grain_boundaries.json.")
    parser.add_argument("--confirm-age",   type=int,   default=CONFIRM_AGE,
                        help="Consecutive hits before a track is confirmed.")
    parser.add_argument("--grace-period",  type=int,   default=GRACE_PERIOD,
                        help="Invisible frames kept alive with velocity prediction.")
    parser.add_argument("--graveyard-age", type=int,   default=GRAVEYARD_AGE,
                        help="Frames a dead track stays revivable.")
    parser.add_argument("--min-area",      type=int,   default=MIN_AREA,
                        help="Minimum blob area in pixels.")
    parser.add_argument("--min-roundness", type=float, default=MIN_ROUNDNESS,
                        help="Minimum roundness (0–1) to accept a blob.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(device)

    grain_boundaries = []
    gb_path = args.gb_path or (args.image_dir / "grain_boundaries.json")
    if gb_path.exists():
        with open(gb_path) as f:
            data = json.load(f)
            grain_boundaries = [((p[0][0], p[0][1]), (p[1][0], p[1][1])) for p in data]
        print(f"Loaded {len(grain_boundaries)} grain boundary segment(s).")

    run(
        image_dir        = args.image_dir,
        out_dir          = args.out_dir,
        model            = model,
        device           = device,
        threshold        = args.threshold,
        n_passes         = args.passes,
        grain_boundaries = grain_boundaries,
        confirm_age      = args.confirm_age,
        grace_period     = args.grace_period,
        graveyard_age    = args.graveyard_age,
        min_area         = args.min_area,
        min_roundness    = args.min_roundness,
    )


if __name__ == "__main__":
    main()
