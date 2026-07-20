import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "clips", "input_video.mp4")
DEFAULT_PROJECT_MODEL_PATH = os.path.join(PROJECT_ROOT, "Model", "yolov8n.pt")

USERPROFILE = os.environ.get("USERPROFILE", "")
DEFAULT_EXTERNAL_MODEL_PATH = (
    os.path.join(USERPROFILE, "Desktop", "F11_TENNIS", "Model", "yolov8n.pt") if USERPROFILE else ""
)

MODEL_CANDIDATES = [
    DEFAULT_PROJECT_MODEL_PATH,
    os.path.join(PROJECT_ROOT, "data", "Model", "yolov5s.pt"),
    os.path.join(PROJECT_ROOT, "data", "Model", "yolov8n.pt"),
    os.path.join(PROJECT_ROOT, "data", "Model", "yolov8s.pt"),
    os.path.join(PROJECT_ROOT, "data", "Model", "yolo11n.pt"),
    os.path.join(PROJECT_ROOT, "data", "Model", "yolov5last.pt"),
    os.path.join(PROJECT_ROOT, "Model", "yolov8n.pt"),
    os.path.join(PROJECT_ROOT, "Model", "yolov5su.pt"),
]

if DEFAULT_EXTERNAL_MODEL_PATH:
    MODEL_CANDIDATES.append(DEFAULT_EXTERNAL_MODEL_PATH)

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "racketpose")
DEFAULT_OUT_CSV = os.path.join(DEFAULT_OUTPUT_DIR, "racket2d.csv")
DEFAULT_OUT_REPORT = os.path.join(DEFAULT_OUTPUT_DIR, "quality_report.txt")
DEFAULT_OUT_DEBUG_VIDEO = os.path.join(DEFAULT_OUTPUT_DIR, "overlay_racket_detection.mp4")


@dataclass
class DetectionResult:
    frame_index: int
    time_sec: float
    detected: int
    x1: Optional[float]
    y1: Optional[float]
    x2: Optional[float]
    y2: Optional[float]
    conf: Optional[float]


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _file_must_exist(path: str, what: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def _try_import_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is required for racket detection but could not be imported. "
            "Install it in your project environment and retry. "
            f"Original error: {repr(e)}"
        )


def _resolve_model_path(user_path: str) -> str:
    if user_path:
        if os.path.isfile(user_path):
            return user_path
        raise FileNotFoundError(f"Model weights not found at provided path: {user_path}")

    for c in MODEL_CANDIDATES:
        if c and os.path.isfile(c):
            return c

    msg = "No model weights were found. Place a model file in one of these locations:\n"
    for c in MODEL_CANDIDATES:
        if c:
            msg += f" - {c}\n"
    raise FileNotFoundError(msg)


def _find_tennis_racket_class_id(names: dict) -> int:
    target = "tennis racket"
    for k, v in names.items():
        if str(v).strip().lower() == target:
            return int(k)
    raise ValueError(
        "Class 'tennis racket' not found in model class names. "
        "Use a COCO-style model that includes this class, or specify --class-id."
    )


def _best_racket_box_from_result(
    result, racket_class_id: int, conf_th: float
) -> Optional[Tuple[float, float, float, float, float]]:
    if result is None or getattr(result, "boxes", None) is None:
        return None

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    best = None
    best_conf = -1.0

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        if cls_id != racket_class_id:
            continue
        if conf < conf_th:
            continue
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
        if conf > best_conf:
            best_conf = conf
            best = (x1, y1, x2, y2, conf)

    return best


def _write_csv(path: str, rows: List[DetectionResult]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_sec", "detected", "x1", "y1", "x2", "y2", "conf"])
        for r in rows:
            w.writerow([r.frame_index, f"{r.time_sec:.6f}", r.detected, r.x1, r.y1, r.x2, r.y2, r.conf])


def _write_report(
    path: str,
    total_frames: int,
    detected_frames: int,
    det_rate: float,
    pass_threshold: float,
    model_path: str,
    conf_th: float,
    video_path: str,
    imgsz: int,
    use_augment: bool,
) -> None:
    _ensure_parent_dir(path)
    passed = det_rate >= pass_threshold
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Racket Detection (2D) - Quality Report ===\n\n")
        f.write(f"Project root: {PROJECT_ROOT}\n")
        f.write(f"Video path: {video_path}\n")
        f.write(f"Model path: {model_path}\n")
        f.write(f"Inference size: {imgsz}\n")
        f.write(f"Multi-scale augment: {use_augment}\n")
        f.write(f"Confidence threshold: {conf_th}\n\n")
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Detected frames: {detected_frames}\n")
        f.write(f"Detection rate: {det_rate:.4f} ({det_rate*100:.2f}%)\n")
        f.write(f"Pass threshold: {pass_threshold:.4f} ({pass_threshold*100:.2f}%)\n\n")
        f.write(f"PASSED: {passed}\n")


def _open_video(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def main() -> int:
    parser = argparse.ArgumentParser(
        description="2D tennis racket detection with optimizations for small objects."
    )

    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, help="Input video path.")
    parser.add_argument("--model", default="", help="YOLO weights path (optional, auto-search is default).")
    parser.add_argument("--class-id", type=int, default=-1, help="Optional explicit class id for 'tennis racket'.")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold (default: 0.10 for better recall).")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size (higher = better small object detection).")
    parser.add_argument("--device", default=None, help="Inference device (e.g., 'cpu', '0').")
    parser.add_argument("--no-augment", action="store_true", help="Disable multi-scale inference augmentation.")

    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="Output CSV path.")
    parser.add_argument("--out-report", default=DEFAULT_OUT_REPORT, help="Output report path.")

    parser.add_argument("--no-debug-video", action="store_true", help="Disable writing debug video with detections.")
    parser.add_argument("--out-debug-video", default=DEFAULT_OUT_DEBUG_VIDEO, help="Debug video output path.")
    parser.add_argument("--pass-threshold", type=float, default=0.70, help="Minimum acceptable detection rate.")

    parser.add_argument("--progress-interval", type=int, default=100, help="Print progress every N frames.")

    args = parser.parse_args()

    _file_must_exist(args.video, "Input video")

    YOLO = _try_import_ultralytics()
    model_path = _resolve_model_path(args.model)
    _file_must_exist(model_path, "Model weights")

    model = YOLO(model_path)

    if args.class_id >= 0:
        racket_class_id = int(args.class_id)
    else:
        names = getattr(model, "names", None)
        if names is None:
            raise RuntimeError("Model does not expose class names (model.names). Provide --class-id explicitly.")
        racket_class_id = _find_tennis_racket_class_id(names)

    print(f"\n{'='*60}")
    print("RACKET DETECTION 2D - OPTIMIZED FOR SMALL OBJECTS")
    print(f"{'='*60}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Video: {args.video}")
    print(f"Model: {model_path}")
    print(f"Inference size: {args.imgsz}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Multi-scale augment: {not args.no_augment}")
    print(f"Pass threshold: {args.pass_threshold:.2%}")
    print(f"{'='*60}\n")

    cap = _open_video(args.video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 0.0

    total_frames_prop = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = int(total_frames_prop) if total_frames_prop and total_frames_prop > 0 else -1

    rows: List[DetectionResult] = []
    detected_frames = 0

    writer = None
    if not args.no_debug_video:
        out_path = str(args.out_debug_video).strip()
        if not out_path:
            raise ValueError("Debug video enabled but --out-debug-video is empty.")
        _ensure_parent_dir(out_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Invalid video dimensions; cannot write debug video.")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 30.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")
        print(f"Debug video will be saved to: {out_path}\n")

    use_augment = not args.no_augment

    frame_index = 0
    print("Processing frames...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        time_sec = (frame_index / fps) if fps > 0 else 0.0

        try:
            preds = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                augment=use_augment,
            )
        except Exception as e:
            raise RuntimeError(f"Model inference failed at frame {frame_index}: {repr(e)}")

        best = None
        if preds is not None and len(preds) > 0:
            best = _best_racket_box_from_result(preds[0], racket_class_id, args.conf)

        if best is not None:
            x1, y1, x2, y2, conf = best
            detected = 1
            detected_frames += 1
        else:
            x1 = y1 = x2 = y2 = conf = None
            detected = 0

        rows.append(
            DetectionResult(
                frame_index=frame_index,
                time_sec=time_sec,
                detected=detected,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                conf=conf,
            )
        )

        if writer is not None:
            vis = frame.copy()

            if detected == 1 and x1 is not None and y1 is not None and x2 is not None and y2 is not None and conf is not None:
                p1 = (int(max(0, x1)), int(max(0, y1)))
                p2 = (int(max(0, x2)), int(max(0, y2)))
                cv2.rectangle(vis, p1, p2, (0, 255, 0), 3)
                
                label = f"RACKET {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_bg_p1 = (p1[0], p1[1] - label_size[1] - 10)
                label_bg_p2 = (p1[0] + label_size[0], p1[1])
                cv2.rectangle(vis, label_bg_p1, label_bg_p2, (0, 255, 0), -1)
                cv2.putText(vis, label, (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                
                status_text = f"DETECTED"
                status_color = (0, 255, 0)
            else:
                status_text = "NOT DETECTED"
                status_color = (0, 0, 255)

            cv2.putText(vis, f"Frame: {frame_index}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"Status: {status_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)
            cv2.putText(vis, f"Detected: {detected_frames}/{frame_index+1}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
            
            writer.write(vis)

        if args.progress_interval > 0 and (frame_index + 1) % args.progress_interval == 0:
            current_rate = detected_frames / (frame_index + 1) if frame_index + 1 > 0 else 0.0
            print(f"  Frame {frame_index + 1}: Detected {detected_frames} / {frame_index + 1} ({current_rate:.2%})")

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    total = frame_index if total_frames < 0 else total_frames
    if total <= 0:
        raise RuntimeError("No frames processed; cannot compute detection rate.")

    det_rate = detected_frames / float(total)

    _write_csv(args.out_csv, rows)
    _write_report(
        args.out_report,
        total,
        detected_frames,
        det_rate,
        args.pass_threshold,
        model_path,
        args.conf,
        args.video,
        args.imgsz,
        use_augment,
    )

    print(f"\n{'='*60}")
    print("DETECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Processed frames: {total}")
    print(f"Detected frames: {detected_frames}")
    print(f"Detection rate: {det_rate:.4f} ({det_rate*100:.2f}%)")
    print(f"Pass threshold: {args.pass_threshold:.4f} ({args.pass_threshold*100:.2f}%)")
    print(f"PASSED: {det_rate >= args.pass_threshold}")
    print(f"\nOutputs:")
    print(f"  CSV: {args.out_csv}")
    print(f"  Report: {args.out_report}")
    if not args.no_debug_video:
        print(f"  Debug video: {args.out_debug_video}")
    print(f"{'='*60}\n")

    return 0 if det_rate >= args.pass_threshold else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)
