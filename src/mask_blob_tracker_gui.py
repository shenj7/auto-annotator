import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from geometry_utils import min_dist_to_grain_boundaries


MIN_AREA = 5
MAX_TRACK_DISTANCE = 25.0
CACHE_SIZE = 24


def _color_from_id(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.RandomState(track_id * 9973 + 17)
    return tuple(int(x) for x in rng.randint(60, 255, size=3))


def _extract_frame_index(path: Path, default_idx: int) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else default_idx


class MaskBlobTrackerGUI:
    def __init__(self, mask_dir: Path, image_dir: Path) -> None:
        self.mask_dir = mask_dir
        self.image_dir = image_dir

        self.mask_paths, self.frame_numbers = self._load_mask_paths()
        if not self.mask_paths:
            raise FileNotFoundError(f"No mask files found in {self.mask_dir}")

        self.image_map, self.image_paths = self._load_image_map()

        self.current_pos = 0
        self.tracks: Dict[int, Dict] = {}
        self.next_track_id = 1

        self.grain_boundaries: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self.gb_mode = False
        self.current_gb_start: Optional[Tuple[int, int]] = None
        self._load_grain_boundaries()

        self._mask_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._component_cache: "OrderedDict[int, List[Dict]]" = OrderedDict()

        cv2.namedWindow("Mask Tracking", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Mask Tracking", self._on_mouse)

    def _load_mask_paths(self) -> Tuple[List[Path], List[int]]:
        paths = sorted(self.mask_dir.glob("*.png"))
        entries: List[Tuple[int, Path]] = []
        for idx, path in enumerate(paths):
            frame_num = _extract_frame_index(path, idx)
            entries.append((frame_num, path))
        entries.sort(key=lambda x: x[0])
        frame_numbers = [frame_num for frame_num, _ in entries]
        mask_paths = [path for _, path in entries]
        return mask_paths, frame_numbers

    def _load_image_map(self) -> Tuple[Dict[int, Path], List[Path]]:
        image_paths = sorted(self.image_dir.glob("*.jpg"))
        image_map: Dict[int, Path] = {}
        for idx, path in enumerate(image_paths):
            frame_num = _extract_frame_index(path, idx)
            image_map.setdefault(frame_num, path)
        return image_map, image_paths

    def _prune_cache(self, cache: OrderedDict) -> None:
        while len(cache) > CACHE_SIZE:
            cache.popitem(last=False)

    def _get_mask(self, pos: int) -> np.ndarray:
        if pos in self._mask_cache:
            self._mask_cache.move_to_end(pos)
            return self._mask_cache[pos]
        mask_path = self.mask_paths[pos]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        self._mask_cache[pos] = mask
        self._prune_cache(self._mask_cache)
        return mask

    def _foreground_from_mask(self, mask: np.ndarray) -> np.ndarray:
        dark = mask < 128
        dark_ratio = float(dark.mean())
        fg = dark if dark_ratio <= 0.5 else ~dark
        return (fg.astype(np.uint8) * 255)

    def _get_components(self, pos: int) -> List[Dict]:
        if pos in self._component_cache:
            self._component_cache.move_to_end(pos)
            return self._component_cache[pos]

        mask = self._get_mask(pos)
        fg = self._foreground_from_mask(mask)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        components: List[Dict] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < MIN_AREA:
                continue
            m = cv2.moments(cnt)
            if m["m00"] == 0:
                continue
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
            components.append(
                {
                    "contour": cnt,
                    "centroid": (cx, cy),
                    "area": area,
                }
            )

        components.sort(key=lambda c: c["area"], reverse=True)
        self._component_cache[pos] = components
        self._prune_cache(self._component_cache)
        return components

    def _get_image(self, pos: int, fallback_shape: Tuple[int, int]) -> np.ndarray:
        frame_num = self.frame_numbers[pos]
        image_path = self.image_map.get(frame_num)
        if image_path is None and pos < len(self.image_paths):
            image_path = self.image_paths[pos]

        if image_path is None:
            h, w = fallback_shape
            return np.zeros((h, w, 3), dtype=np.uint8)

        image = cv2.imread(str(image_path))
        if image is None:
            h, w = fallback_shape
            return np.zeros((h, w, 3), dtype=np.uint8)

        if image.shape[:2] != fallback_shape:
            # Keep overlays aligned to the mask coordinates.
            image = cv2.resize(image, (fallback_shape[1], fallback_shape[0]))
        return image

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.gb_mode:
            if self.current_gb_start is None:
                self.current_gb_start = (x, y)
            else:
                self.grain_boundaries.append((self.current_gb_start, (x, y)))
                self.current_gb_start = None
                self._save_grain_boundaries()
            return

        pos = self.current_pos
        components = self._get_components(pos)
        if not components:
            return

        for track_id, track in list(self.tracks.items()):
            contour = track["contours"].get(pos)
            if contour is None:
                continue
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                del self.tracks[track_id]
                return

        for comp in components:
            if cv2.pointPolygonTest(comp["contour"], (x, y), False) >= 0:
                self._add_track(pos, comp)
                break

    def _add_track(self, pos: int, component: Dict) -> None:
        track_id = self.next_track_id
        self.next_track_id += 1
        color = _color_from_id(track_id)

        self.tracks[track_id] = {
            "id": track_id,
            "color": color,
            "start_pos": pos,
            "last_pos": pos,
            "active": True,
            "positions": {pos: component["centroid"]},
            "contours": {pos: component["contour"]},
        }

    def _closest_component(
        self, centroid: Tuple[float, float], components: List[Dict]
    ) -> Optional[Dict]:
        best = None
        best_dist = MAX_TRACK_DISTANCE + 1.0
        for comp in components:
            cx, cy = comp["centroid"]
            dist = float(np.hypot(cx - centroid[0], cy - centroid[1]))
            if dist < best_dist:
                best_dist = dist
                best = comp
        if best is None or best_dist > MAX_TRACK_DISTANCE:
            return None
        return best

    def _extend_track(self, track: Dict, target_pos: int) -> None:
        if not track["active"]:
            return
        pos = track["last_pos"]
        while pos < target_pos:
            pos += 1
            components = self._get_components(pos)
            if not components:
                track["active"] = False
                break
            prev_centroid = track["positions"][track["last_pos"]]
            closest = self._closest_component(prev_centroid, components)
            if closest is None:
                track["active"] = False
                break
            track["positions"][pos] = closest["centroid"]
            track["contours"][pos] = closest["contour"]
            track["last_pos"] = pos

    def _ensure_tracks(self, target_pos: int) -> None:
        for track in self.tracks.values():
            if track["last_pos"] < target_pos:
                self._extend_track(track, target_pos)

    def _draw_label(self, image: np.ndarray, text: str, origin: Tuple[int, int]) -> None:
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_tracks(self, image: np.ndarray, pos: int, draw_paths: bool) -> None:
        for track in self.tracks.values():
            if pos < track["start_pos"]:
                continue
            points = [
                (int(x), int(y))
                for p, (x, y) in sorted(track["positions"].items())
                if p <= pos
            ]
            if draw_paths and len(points) > 1:
                cv2.polylines(
                    image,
                    [np.array(points, dtype=np.int32)],
                    isClosed=False,
                    color=track["color"],
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

            if pos in track["contours"]:
                cv2.drawContours(image, [track["contours"][pos]], -1, track["color"], 2)

            if pos in track["positions"]:
                cx, cy = track["positions"][pos]
                cv2.circle(image, (int(cx), int(cy)), 4, track["color"], -1)
                cv2.putText(
                    image,
                    str(track["id"]),
                    (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image,
                    str(track["id"]),
                    (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    track["color"],
                    1,
                    cv2.LINE_AA,
                )

    def _draw_grain_boundaries(self, image: np.ndarray) -> None:
        for start, end in self.grain_boundaries:
            p1 = (int(start[0]), int(start[1]))
            p2 = (int(end[0]), int(end[1]))
            cv2.line(image, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)
        
        if self.current_gb_start:
            # Drawing current line preview is hard with current waitKey loop 
            # but we can at least show the start point
            cv2.circle(image, self.current_gb_start, 4, (0, 255, 255), -1)

    def _render(self) -> None:
        self._ensure_tracks(self.current_pos)

        mask = self._get_mask(self.current_pos)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self._draw_tracks(mask_bgr, self.current_pos, draw_paths=True)
        self._draw_grain_boundaries(mask_bgr)

        frame_num = self.frame_numbers[self.current_pos]
        label = f"Frame {self.current_pos + 1}/{len(self.mask_paths)} (id {frame_num})"
        self._draw_label(mask_bgr, label, (10, 22))
        mode_str = "GB MODE" if self.gb_mode else "TRACK MODE"
        self._draw_label(mask_bgr, f"{mode_str} | g: toggle GB | c: clear GB | q: quit", (10, 44))

        image = self._get_image(self.current_pos, mask.shape)
        self._draw_tracks(image, self.current_pos, draw_paths=False)
        self._draw_grain_boundaries(image)
        self._draw_label(image, label, (10, 22))

        composite = np.hstack([mask_bgr, image])
        cv2.imshow("Mask Tracking", composite)

    def _step(self, delta: int) -> None:
        self.current_pos = int(np.clip(self.current_pos + delta, 0, len(self.mask_paths) - 1))

    def _clear_tracks(self) -> None:
        self.tracks.clear()
        self.next_track_id = 1

    def run(self) -> None:
        print("Left-click a blob to track. Use left/right arrow keys to move frames. q quits.")
        while True:
            self._render()
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("g"):
                self.gb_mode = not self.gb_mode
                self.current_gb_start = None
            elif key == ord("c"):
                self.grain_boundaries.clear()
                self._save_grain_boundaries()
            elif key == 81:
                self._step(-1)
            elif key == 83:
                self._step(1)

        cv2.destroyAllWindows()

    def _save_grain_boundaries(self) -> None:
        gb_path = self.mask_dir / "grain_boundaries.json"
        try:
            with open(gb_path, "w") as f:
                json.dump(self.grain_boundaries, f)
        except Exception as e:
            print(f"Error saving grain boundaries: {e}")

    def _load_grain_boundaries(self) -> None:
        gb_path = self.mask_dir / "grain_boundaries.json"
        if gb_path.exists():
            try:
                with open(gb_path, "r") as f:
                    data = json.load(f)
                    self.grain_boundaries = [((p1[0], p1[1]), (p2[0], p2[1])) for p1, p2 in data]
            except Exception as e:
                print(f"Error loading grain boundaries: {e}")

    def write_summary_csv(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "track_id",
            "blob_size_px",
            "distance_px",
            "dist_to_gb_px",
            "lifespan_frames",
        ]
        rows: List[Dict[str, float]] = []
        for track in self.tracks.values():
            positions = [
                track["positions"][pos]
                for pos in sorted(track["positions"].keys())
            ]
            if not positions:
                continue

            distance = 0.0
            gb_distances = []
            for idx in range(len(positions)):
                p = positions[idx]
                gb_distances.append(min_dist_to_grain_boundaries(p, self.grain_boundaries))
                
                if idx > 0:
                    x0, y0 = positions[idx - 1]
                    x1, y1 = positions[idx]
                    distance += float(np.hypot(x1 - x0, y1 - y0))

            mean_gb_dist = float(np.nanmean(gb_distances)) if gb_distances else 0.0

            areas = []
            for contour in track["contours"].values():
                area = float(cv2.contourArea(contour))
                if area > 0:
                    areas.append(area)

            blob_size = float(np.mean(areas)) if areas else 0.0
            rows.append(
                {
                    "track_id": int(track["id"]),
                    "blob_size_px": blob_size,
                    "distance_px": distance,
                    "dist_to_gb_px": mean_gb_dist,
                    "lifespan_frames": int(len(positions)),
                }
            )

        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Wrote track summary to {output_path}")

    def write_detailed_csv(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "track_id",
            "frame",
            "centroid_x",
            "centroid_y",
            "dist_to_gb_px",
        ]
        
        rows: List[Dict] = []
        for track_id, track in self.tracks.items():
            for pos in sorted(track["positions"].keys()):
                p = track["positions"][pos]
                frame_idx = self.frame_numbers[pos]
                dist = min_dist_to_grain_boundaries(p, self.grain_boundaries)
                
                rows.append({
                    "track_id": track_id,
                    "frame": frame_idx,
                    "centroid_x": p[0],
                    "centroid_y": p[1],
                    "dist_to_gb_px": dist,
                })

        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Wrote detailed tracks to {output_path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_mask_dir = script_dir / "data" / "auto_images" / "masks"
    default_image_dir = script_dir / "data" / "images"
    default_summary_csv = default_mask_dir / "selected_blob_tracks.csv"
    default_detailed_csv = default_mask_dir / "detailed_blob_tracks.csv"

    parser = argparse.ArgumentParser(
        description="Select and track blobs from mask frames with image overlay."
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=default_mask_dir,
        help="Directory with mask PNGs (default: src/data/auto_images/masks).",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=default_image_dir,
        help="Directory with corresponding images (default: src/data/images).",
    )
    parser.add_argument(
        "--out-summary-csv",
        type=Path,
        default=default_summary_csv,
        help="CSV path for track summary (default: .../selected_blob_tracks.csv).",
    )
    parser.add_argument(
        "--out-detailed-csv",
        type=Path,
        default=default_detailed_csv,
        help="CSV path for detailed tracks (default: .../detailed_blob_tracks.csv).",
    )
    args = parser.parse_args()

    gui = MaskBlobTrackerGUI(mask_dir=args.mask_dir, image_dir=args.image_dir)
    gui.run()
    gui.write_summary_csv(args.out_summary_csv)
    gui.write_detailed_csv(args.out_detailed_csv)


if __name__ == "__main__":
    main()
