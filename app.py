# app.py
import io, os, time, requests
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from dotenv import load_dotenv, find_dotenv
import gradio as gr
import tempfile
import uuid


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

# -------- gradio callback --------
def infer(image: Image.Image, server_conf: float, ui_conf: float):
    if image is None:
        return None, pd.DataFrame([{"message": "Upload an image."}]), "Total detections: 0", None

    # preprocess (rotate/downsize) to avoid timeouts and box misalignment
    prep = _preprocess_image(image)

    rf_resp, status = roboflow_detect(prep, server_conf=server_conf)

    # surface API errors in the UI
    if status != 200:
        msg = f"API error (HTTP {status}). "
        if isinstance(rf_resp, dict):
            if "error" in rf_resp: msg += str(rf_resp["error"])
            if "_raw" in rf_resp:  msg += " | " + rf_resp["_raw"][:200]
        return image, pd.DataFrame([{"error": msg}]), msg, None

    annotated, df, summary, downloadable = draw_detections(prep, rf_resp, ui_conf=ui_conf)
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
