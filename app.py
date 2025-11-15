# app.py
import io, os, time, requests
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from dotenv import load_dotenv, find_dotenv
import gradio as gr
import tempfile
import uuid
import csv, json
from datetime import datetime


# -------- config & env --------
load_dotenv(find_dotenv(), override=True)

RF_KEY     = os.getenv("ROBOFLOW_API_KEY", "").strip()
RF_MODEL   = os.getenv("ROBOFLOW_MODEL", "").strip()
RF_VERSION = os.getenv("ROBOFLOW_VERSION", "").strip()

print("MODEL:", RF_MODEL)
print("VERSION:", RF_VERSION)
print("API KEY starts with:", (RF_KEY[:5] if RF_KEY else ""))

if not (RF_KEY and RF_MODEL and RF_VERSION):
    print("⚠️  Set ROBOFLOW_API_KEY, ROBOFLOW_MODEL, ROBOFLOW_VERSION in .env")

DEBUG = True  # set True to print raw API responses in terminal

# -------- helpers --------
def _xywh_to_xyxy(x, y, w, h):
    return x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0

def _preprocess_image(pil_img: Image.Image, max_side=1600) -> Image.Image:
    # Fix phone rotations + optionally downscale for faster uploads
    img = ImageOps.exif_transpose(pil_img.convert("RGB"))
    w, h = img.size
    scale = max(w, h) / float(max_side)
    if scale > 1.0:
        new_w, new_h = int(w/scale), int(h/scale)
        img = img.resize((new_w, new_h))
    return img

# -------- Roboflow API --------
def roboflow_detect(pil_image: Image.Image, server_conf: float = 0.15, overlap: float = 0.45):
    """
    Returns (json_dict, status_code). Raises on non-HTTP errors only.
    """
    url = f"https://detect.roboflow.com/{RF_MODEL}/{RF_VERSION}"
    params = {"api_key": RF_KEY, "confidence": server_conf, "overlap": overlap, "format": "json"}

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)

    for attempt in range(3):
        resp = requests.post(
            url,
            params=params,
            files={"file": ("upload.png", buf.getvalue(), "image/png")},
            timeout=90
        )
        if DEBUG:
            print("POST", resp.url, "status", resp.status_code)
            print((resp.text or "")[:600])
        # retry transient server errors
        if resp.status_code in (502, 503, 504):
            time.sleep(1 + attempt)
            continue
        # success or client error -> return as-is
        try:
            data = resp.json()
        except Exception:
            data = {"_raw": (resp.text or "")[:600]}
        return data, resp.status_code

    return {"error": "Roboflow API unavailable after retries."}, 503

# -------- draw & summarize --------
def draw_detections(image: Image.Image, rf_json: dict, ui_conf: float):
    """
    Returns: annotated_img, df, summary_text, downloadable_tuple
    """
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    rf_w = (rf_json.get("image") or {}).get("width")
    rf_h = (rf_json.get("image") or {}).get("height")
    sx = (W / rf_w) if rf_w else 1.0
    sy = (H / rf_h) if rf_h else 1.0

    raw = rf_json.get("predictions", rf_json if isinstance(rf_json, list) else [])
    if isinstance(raw, dict) and "predictions" in raw:
        raw = raw["predictions"]

    rows = []
    for p in raw:
        score = float(p.get("confidence", p.get("score", 0.0)))
        if score < ui_conf:
            continue
        label = p.get("class") or p.get("label") or "object"

        if all(k in p for k in ("x","y","width","height")):
            xmin, ymin, xmax, ymax = _xywh_to_xyxy(float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"]))
        elif "bbox" in p and isinstance(p["bbox"], dict) and all(k in p["bbox"] for k in ("x","y","width","height")):
            b = p["bbox"]
            xmin, ymin, xmax, ymax = _xywh_to_xyxy(float(b["x"]), float(b["y"]), float(b["width"]), float(b["height"]))
        else:
            xmin = float(p.get("xmin", 0)); ymin = float(p.get("ymin", 0))
            xmax = float(p.get("xmax", xmin)); ymax = float(p.get("ymax", ymin))

        # scale from Roboflow space -> display space
        xmin, xmax = xmin * sx, xmax * sx
        ymin, ymax = ymin * sy, ymax * sy

        # clamp
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(W, xmax), min(H, ymax)
        if xmax < xmin: xmin, xmax = xmax, xmin
        if ymax < ymin: ymin, ymax = ymax, ymin

        # draw
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)
        cap = f"{label} {score:.2f}"
        text_w = 8 * len(cap) + 6
        draw.rectangle([(xmin, max(0, ymin-18)), (xmin + text_w, ymin)], fill="red")
        draw.text((xmin + 3, max(0, ymin-16)), cap, fill="white")

        rows.append({
            "label": label, "score": round(score, 4),
            "xmin": round(xmin, 1), "ymin": round(ymin, 1),
            "xmax": round(xmax, 1), "ymax": round(ymax, 1),
            "width": round(xmax - xmin, 1), "height": round(ymax - ymin, 1)
        })

    df = pd.DataFrame(rows, columns=["label","score","xmin","ymin","xmax","ymax","width","height"])
    total = len(df)
    per_class = (df["label"].value_counts().to_dict() if total else {})
    per_str = ", ".join(f"{k}: {v}" for k, v in per_class.items()) if per_class else "none"
    summary = f"Total detections: {total}  |  per-class: {per_str}"

    # Save to a temp file for Gradio File output
    tmp_path = os.path.join(tempfile.gettempdir(), f"annotated_{uuid.uuid4().hex}.png")
    img.save(tmp_path, format="PNG")
    return img, df, summary, tmp_path

# ===== Heuristic agent helpers: 1-step adaptive retry + CSV logging =====

# Where the CSV log will be written
LOG_CSV_PATH = os.path.join(os.path.dirname(__file__), "pizza_runs_log.csv")

def _filtered_count(rf_json: dict, ui_conf: float) -> int:
    preds = rf_json.get("predictions", []) if isinstance(rf_json, dict) else []
    return sum(1 for p in preds if float(p.get("confidence", p.get("score", 0.0))) >= ui_conf)

def _per_class_map(df: pd.DataFrame) -> dict:
    try:
        return df["label"].value_counts().to_dict()
    except Exception:
        return {}

def log_run(*, image_name: str|None, server_conf_initial: float, server_conf_used: float,
            ui_conf: float, retried: bool, http_status: int, total_count: int,
            per_class: dict, error: str|None):
    """Append one row to a CSV log (auto-creates header on first write)."""
    os.makedirs(os.path.dirname(LOG_CSV_PATH), exist_ok=True)
    row = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "image_name": image_name or "",
        "server_conf_initial": round(server_conf_initial, 4),
        "server_conf_used": round(server_conf_used, 4),
        "ui_conf": round(ui_conf, 4),
        "retried": bool(retried),
        "http_status": int(http_status),
        "total_count": int(total_count),
        "per_class_json": json.dumps(per_class, ensure_ascii=False),
        "error": error or ""
    }
    write_header = not os.path.exists(LOG_CSV_PATH)
    with open(LOG_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

def adaptive_detect_once(pil_image: Image.Image, first_conf: float, ui_conf: float,
                         overlap: float = 0.45, step: float = 0.05, min_conf: float = 0.05):
    """
    Try at first_conf; if zero detections after UI filter, retry once at (first_conf - step), floored at min_conf.
    Returns: (rf_json, http_status, used_conf, retried_flag)
    """
    # pass 1
    rf1, st1 = roboflow_detect(pil_image, server_conf=first_conf, overlap=overlap)
    cnt1 = _filtered_count(rf1 if st1 == 200 else {}, ui_conf=ui_conf)
    if st1 == 200 and cnt1 > 0:
        return rf1, st1, first_conf, False

    # pass 2 (lower by 5%)
    lower = max(min_conf, round(first_conf - step, 4))
    rf2, st2 = roboflow_detect(pil_image, server_conf=lower, overlap=overlap)
    return rf2, st2, lower, True
# =======================================================================



# -------- gradio callback --------
def infer(image: Image.Image, server_conf: float, ui_conf: float):
    """
    Heuristic agent:
      1) run detect at server_conf
      2) if zero detections (post UI filter), retry once at server_conf - 0.05
      3) draw results and log every run to CSV
    """
    if image is None:
        return None, pd.DataFrame([{"message": "Upload an image."}]), "Total detections: 0", None

    # Preprocess (rotate/downsize) to avoid timeouts and misalignment
    prep = _preprocess_image(image)

    # Adaptive retry once if needed
    rf_resp, status, used_conf, retried = adaptive_detect_once(
        prep, first_conf=server_conf, ui_conf=ui_conf, overlap=0.45, step=0.05, min_conf=0.05
    )

    # Handle HTTP errors
    if status != 200:
        msg = f"API error (HTTP {status}). "
        if isinstance(rf_resp, dict):
            if "error" in rf_resp: msg += str(rf_resp["error"])
            if "_raw" in rf_resp:  msg += " | " + rf_resp["_raw"][:200]
        # Log error run
        log_run(
            image_name=getattr(image, "name", None),
            server_conf_initial=server_conf,
            server_conf_used=used_conf,
            ui_conf=ui_conf,
            retried=retried,
            http_status=status,
            total_count=0,
            per_class={},
            error=msg
        )
        return image, pd.DataFrame([{"error": msg}]), msg, None

    # Draw detections (your existing renderer applies the UI filter)
    annotated, df, summary, downloadable = draw_detections(prep, rf_resp, ui_conf=ui_conf)

    total = len(df)
    per_cls = _per_class_map(df)
    summary = f"{summary}  |  conf_used={used_conf:.2f}{' (auto-retry)' if retried else ''}"

    # Log success run
    log_run(
        image_name=getattr(image, "name", None),
        server_conf_initial=server_conf,
        server_conf_used=used_conf,
        ui_conf=ui_conf,
        retried=retried,
        http_status=status,
        total_count=total,
        per_class=per_cls,
        error=None
    )

    return annotated, df, summary, downloadable


# -------- UI --------
with gr.Blocks(title="Pizza Box Detector — Roboflow API") as demo:
    gr.Markdown("## 🍕 Pizza Box Detector — Roboflow Hosted Inference\nUpload an image; get boxes, labels, and counts.")
    with gr.Row():
        with gr.Column(scale=1):
            in_img      = gr.Image(type="pil", label="Upload image")
            server_conf = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="Server confidence (Roboflow threshold)")
            ui_conf     = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="UI filter (client-side)")
            btn         = gr.Button("Detect")
        with gr.Column(scale=1):
            out_img  = gr.Image(type="pil", label="Annotated image")
            out_df   = gr.Dataframe(headers=["label","score","xmin","ymin","xmax","ymax","width","height"],
                                    label="Detections", wrap=True)
            out_txt  = gr.Markdown("Total detections: 0")
            out_file = gr.File(label="Download annotated image")

    btn.click(infer, inputs=[in_img, server_conf, ui_conf], outputs=[out_img, out_df, out_txt, out_file])

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
