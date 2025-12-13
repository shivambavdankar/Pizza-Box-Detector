# app.py
import io, os, time, requests, re
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, ImageFont
from dotenv import load_dotenv, find_dotenv
import gradio as gr
import tempfile
import uuid
import csv, json
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from pathlib import Path
import zipfile

# Optional: PyMuPDF for Google Doc (PDF export) rendering
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# -------- config & env --------
load_dotenv(find_dotenv(), override=True)

RF_KEY       = os.getenv("ROBOFLOW_API_KEY", "").strip()
RF_MODEL     = os.getenv("ROBOFLOW_MODEL", "").strip()
RF_VERSION   = os.getenv("ROBOFLOW_VERSION", "").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").strip().lower()
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "").strip()

print("MODEL:", RF_MODEL)
print("VERSION:", RF_VERSION)
print("API KEY starts with:", (RF_KEY[:5] if RF_KEY else ""))

if not (RF_KEY and RF_MODEL and RF_VERSION):
    print("⚠️  Set ROBOFLOW_API_KEY, ROBOFLOW_MODEL, ROBOFLOW_VERSION in .env")

DEBUG = True  # print raw API responses / fetch logs

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

# Small graphics helper for category banner
from PIL import ImageFont, ImageDraw, Image

CATEGORY_COLORS = {
    "with_pizza": (34, 139, 34, 210),   # green
    "torn":       (220, 20, 60, 210),   # crimson
    "good_shape": (30, 144, 255, 210),  # dodger blue
    "no_box":     (128, 128, 128, 210), # gray
}

def _load_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Try a common server font; fall back to default
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ):
        try:
            return ImageFont.truetype(path, max(12, px))
        except Exception:
            continue
    return ImageFont.load_default()

def _draw_banner(
    img: Image.Image,
    text: str,
    category: str | None = None,
    position: str = "bottom",  # "top" or "bottom"
):
    """
    Draws a semi-transparent, rounded pill banner with auto font sizing.
    """
    # Work in RGBA for transparency, then convert back to RGB
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    W, H = base.size

    # Auto font size relative to image; clamp to [16, 48]
    target_px = int(min(W, H) * 0.05)
    font = _load_font(max(16, min(48, target_px)))

    # Optional emoji for a bit of flair
    emoji_by_cat = {
        "with_pizza": "🍕",
        "torn": "🩹",
        "good_shape": "✅",
        "no_box": "❓",
    }
    em = emoji_by_cat.get((category or "").lower(), "📦")
    display_text = f"{em}  {text}"

    # Measure text
    bbox = odraw.textbbox((0, 0), display_text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Padding & pill geometry
    pad_x = int(th * 0.8)
    pad_y = int(th * 0.5)
    pill_w = tw + pad_x * 2
    pill_h = th + pad_y * 2
    radius = pill_h // 2

    # Position centered at top or bottom with margin
    margin = int(H * 0.02)
    cx = W // 2
    if position == "top":
        y0 = margin
    else:
        y0 = H - pill_h - margin
    x0 = max(0, cx - pill_w // 2)
    x1 = min(W, x0 + pill_w)
    y1 = y0 + pill_h

    # Background color w/ alpha
    bg = CATEGORY_COLORS.get((category or "").lower(), (0, 0, 0, 210))
    # Draw rounded rectangle
    odraw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=bg)

    # Text (centered)
    tx = x0 + (pill_w - tw) // 2
    ty = y0 + (pill_h - th) // 2
    odraw.text((tx, ty), display_text, fill=(255, 255, 255, 255), font=font)

    # Composite back to RGB
    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return composed


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
    Returns: annotated_img, df, summary_text, downloadable_path
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

# ========================= Robust URL ingestion (webpage & Google Doc) =========================
_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,image/png,*/*;q=0.8",
}

def _http_get(url: str, referer: str | None = None, timeout: int = 30):
    headers = dict(_BROWSER_HEADERS)
    if referer:
        headers["Referer"] = referer
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if DEBUG:
            print(f"[http] {resp.status_code} GET {url} ct={resp.headers.get('Content-Type','')}")
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"[http] GET fail {url}: {e}")
        return None

def _is_image_content_type(ct: str) -> bool:
    return (ct or "").lower().startswith("image/")

def _looks_like_image_url(url: str) -> bool:
    return url.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"))

def _download_image_to_pil(url: str, referer: str | None = None) -> Image.Image | None:
    if url.lower().endswith(".svg") or url.lower().startswith("data:"):
        if DEBUG: print(f"[img] skip non-raster {url}")
        return None
    resp = _http_get(url, referer=referer, timeout=40)
    if not resp:
        return None
    ct = resp.headers.get("Content-Type", "").lower()
    if "image/svg" in ct:
        if DEBUG: print(f"[img] skip svg {url}")
        return None
    try:
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"[img] decode fail {url}: {e}")
        return None

def _pick_from_srcset(srcset: str) -> str | None:
    best_url, best_score = None, -1.0
    for part in (srcset or "").split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"(\S+)\s+(\d+)w", part)
        if m:
            u, w = m.group(1), float(m.group(2))
            if w > best_score:
                best_url, best_score = u, w
            continue
        m = re.match(r"(\S+)\s+(\d+(?:\.\d+)?)x", part)
        if m:
            u, x = m.group(1), float(m.group(2))
            if x > best_score:
                best_url, best_score = u, x
            continue
        if not best_url:
            best_url = part.split()[0]
    return best_url

def fetch_images_from_webpage(page_url: str, max_images: int = 20) -> list[Image.Image]:
    if _looks_like_image_url(page_url):
        if DEBUG: print("[fetch] direct image URL detected")
        pil = _download_image_to_pil(page_url, referer=None)
        return [pil] if pil else []

    probe = _http_get(page_url, timeout=40)
    if not probe:
        return []
    if _is_image_content_type(probe.headers.get("Content-Type", "")):
        if DEBUG: print("[fetch] page URL actually returned image/*")
        try:
            pil = Image.open(io.BytesIO(probe.content)).convert("RGB")
            return [pil]
        except Exception as e:
            print(f"[img] direct decode fail: {e}")
            return []

    html = probe.text
    soup = BeautifulSoup(html, "html.parser")
    imgs: list[Image.Image] = []

    def _abs(u: str) -> str:
        return urljoin(page_url, u)

    img_tags = soup.find_all("img")
    if DEBUG: print(f"[parse] found {len(img_tags)} <img> tags on page")
    for tag in img_tags:
        cand = None
        srcset = tag.get("srcset") or tag.get("data-srcset") or ""
        if srcset:
            cand = _pick_from_srcset(srcset)
        for attr in ("src", "data-src", "data-original", "data-lazy-src"):
            if not cand and tag.get(attr):
                cand = tag.get(attr)
        if not cand:
            continue
        url = _abs(cand)
        pil = _download_image_to_pil(url, referer=page_url)
        if pil:
            imgs.append(pil)
            if len(imgs) >= max_images:
                return imgs

    for pic in soup.find_all("picture"):
        for src in pic.find_all("source"):
            cand = _pick_from_srcset(src.get("srcset") or "")
            if not cand:
                continue
            url = _abs(cand)
            pil = _download_image_to_pil(url, referer=page_url)
            if pil:
                imgs.append(pil)
                if len(imgs) >= max_images:
                    return imgs

    for tag in soup.find_all(style=True):
        style = tag.get("style") or ""
        m = re.search(r'background-image\s*:\s*url\((["\']?)(.*?)\1\)', style, flags=re.I)
        if not m:
            continue
        url = _abs(m.group(2))
        pil = _download_image_to_pil(url, referer=page_url)
        if pil:
            imgs.append(pil)
            if len(imgs) >= max_images:
                return imgs

    for prop in ("og:image", "twitter:image", "twitter:image:src"):
        meta = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if meta and meta.get("content"):
            url = _abs(meta["content"])
            pil = _download_image_to_pil(url, referer=page_url)
            if pil:
                imgs.append(pil)
                if len(imgs) >= max_images:
                    return imgs

    if DEBUG: print(f"[parse] collected {len(imgs)} images total")
    return imgs

def _is_gdoc(url: str) -> bool:
    return "docs.google.com/document" in url

def _gdoc_export_pdf(url: str) -> bytes | None:
    try:
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
        file_id = None
        for i, p in enumerate(parts):
            if p == "d" and i+1 < len(parts):
                file_id = parts[i+1]
                break
        if not file_id:
            return None
        pdf_url = f"https://docs.google.com/document/d/{file_id}/export?format=pdf"
        r = requests.get(pdf_url, timeout=45, headers=_BROWSER_HEADERS)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"[gdoc] export error: {e}")
        return None

def fetch_images_from_gdoc(url: str, dpi: int = 144) -> list[Image.Image]:
    imgs = []
    if not fitz:
        print("[gdoc] PyMuPDF not installed; cannot render PDF pages.")
        return imgs
    blob = _gdoc_export_pdf(url)
    if not blob:
        return imgs
    try:
        doc = fitz.open(stream=blob, filetype="pdf")
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            imgs.append(img)
        doc.close()
    except Exception as e:
        print(f"[gdoc] render error: {e}")
    return imgs

def load_any_images_from_url(url: str) -> list[Image.Image]:
    return fetch_images_from_gdoc(url) if _is_gdoc(url) else fetch_images_from_webpage(url)

# ========================= Category logic (heuristic + optional LLM refine) =========================
BOX_KEYWORDS     = ("box", "carton", "container")
PIZZA_KEYWORDS   = ("pizza",)
DAMAGED_KEYWORDS = ("torn", "damaged", "crushed", "broken")

def _center(row):
    cx = row["xmin"] + row["width"]  / 2.0
    cy = row["ymin"] + row["height"] / 2.0
    return cx, cy

def _point_in_bbox(px, py, row):
    return (row["xmin"] <= px <= row["xmax"]) and (row["ymin"] <= py <= row["ymax"])

def _has_keyword(label: str, keywords: tuple[str, ...]) -> bool:
    s = (label or "").lower()
    return any(k in s for k in keywords)

def classify_image_heuristic(df: pd.DataFrame) -> tuple[str, str]:
    """
    Returns (category, reason)
    Categories: no_box, with_pizza, torn, good_shape
    """
    if df is None or df.empty:
        return "no_box", "No detections after filtering."

    # any damaged keywords in labels
    has_damaged = any(_has_keyword(lbl, DAMAGED_KEYWORDS) for lbl in df["label"])
    # sets
    boxes  = df[df["label"].str.lower().apply(lambda s: _has_keyword(s, BOX_KEYWORDS))]
    pizzas = df[df["label"].str.lower().apply(lambda s: _has_keyword(s, PIZZA_KEYWORDS))]

    if not boxes.empty and not pizzas.empty:
        # if any pizza center lies inside any box -> with_pizza
        pizza_centers = [ _center(r) for _, r in pizzas.iterrows() ]
        for _, bx in boxes.iterrows():
            for (px, py) in pizza_centers:
                if _point_in_bbox(px, py, bx):
                    return "with_pizza", "Found a pizza whose center lies inside a detected box."
        # if both exist but no containment, still likely with_pizza
        return "with_pizza", "Detected both pizza and box classes."

    if has_damaged:
        return "torn", "Detected a label that suggests damage (e.g., torn/damaged)."

    if not boxes.empty:
        return "good_shape", "Detected a box with no signs of damage or visible pizza."

    return "no_box", "Only non-box classes detected."

def classify_image_with_llm(df: pd.DataFrame) -> tuple[str, str]:
    """
    Optional classifier using LLM over structured detections.
    Only called if OPENAI is configured.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)

        schema_hint = {
            "task": "Decide category for a pizza-related image from detections.",
            "categories": ["no_box","with_pizza","torn","good_shape"],
            "rule_of_thumb": [
                "with_pizza if pizza detection overlaps or co-exists with a box",
                "torn if any damaged/torn-like label",
                "good_shape if a box exists without pizza or damage",
                "no_box if no relevant box detection"
            ]
        }
        table = df.to_dict(orient="records") if df is not None else []
        prompt = (
            "You are a precise classifier. Given detections (label, score, x/y/width/height), "
            "choose exactly one category in {no_box, with_pizza, torn, good_shape} and explain briefly.\n\n"
            f"Guidelines: {json.dumps(schema_hint, ensure_ascii=False)}\n\n"
            f"Detections JSON: {json.dumps(table, ensure_ascii=False)}\n\n"
            "Return JSON only as: {\"category\": \"...\", \"reason\": \"...\"}"
        )
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        txt = r.choices[0].message.content.strip()
        data = json.loads(txt)
        cat = data.get("category", "")
        rsn = data.get("reason", "")
        if cat in {"no_box","with_pizza","torn","good_shape"}:
            return cat, rsn or "LLM-based classification."
    except Exception as e:
        print(f"[LLM classify] fallback due to: {e}")
    # fallback to heuristic on any failure
    return classify_image_heuristic(df)

def classify_image(df: pd.DataFrame) -> tuple[str, str]:
    # Try LLM refine only if configured
    if LLM_PROVIDER == "openai" and OPENAI_KEY:
        return classify_image_with_llm(df)
    return classify_image_heuristic(df)

# ========================= Agent: URL → detection (summary) =========================
def _zip_files(file_paths: list[str]) -> str | None:
    """Create a zip of given files in temp dir and return its path."""
    if not file_paths:
        return None
    zpath = os.path.join(tempfile.gettempdir(), f"annotated_{uuid.uuid4().hex}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in file_paths:
            try:
                arcname = Path(fp).name
                zf.write(fp, arcname=arcname)
            except Exception as e:
                print(f"[zip] skip {fp}: {e}")
    return zpath

def run_agent_from_url(url: str, server_conf: float = 0.15, ui_conf: float = 0.15):
    """
    Returns a triple:
      (result_json: dict, annotated_pils: list[PIL.Image], zip_path: str|None)
    """
    images = load_any_images_from_url(url)
    if not images:
        return ({"ok": False, "message": "No images found at the provided URL."}, [], None)

    per_item = []
    annotated_pils = []
    annotated_file_paths = []

    for idx, im in enumerate(images, start=1):
        prep = _preprocess_image(im)
        rf_resp, status, used_conf, retried = adaptive_detect_once(
            prep, first_conf=server_conf, ui_conf=ui_conf, overlap=0.45, step=0.05, min_conf=0.05
        )
        if status != 200:
            per_item.append({"index": idx, "status": status, "summary": f"HTTP {status}", "count": 0})
            continue

        annotated, df, summary, tmp_path = draw_detections(prep, rf_resp, ui_conf=ui_conf)

        # ---- NEW: categorize & banner ----
        category, reason = classify_image(df)
        annotated = _draw_banner(
            annotated,
            text=f"Category: {category}",
            category=category,      # <-- enables color mapping & emoji
            position="bottom"
        )


        annotated_pils.append(annotated)
        if tmp_path:
            # overwrite the saved file with bannered version
            annotated.save(tmp_path, format="PNG")
            annotated_file_paths.append(tmp_path)

        per_item.append({
            "index": idx,
            "status": status,
            "count": int(len(df)),
            "category": category,
            "reason": reason,
            "summary": f"{summary} | conf_used={used_conf:.2f}{' (auto-retry)' if retried else ''}"
        })

    zip_path = _zip_files(annotated_file_paths)
    result = {"ok": True, "images_processed": len(images), "items": per_item}
    return (result, annotated_pils, zip_path)

# ========================= LLM parser for Chat (OpenAI / regex fallback) =========================
def _find_url_in_text(txt: str) -> str | None:
    m = re.search(r'(https?://\S+)', txt or "", re.IGNORECASE)
    return m.group(1).strip(')];,') if m else None

def _llm_parse_freeform(msg: str) -> dict:
    """
    Return {"cmd": "agent"|"help", "url": str|None, "server_conf": float|None, "ui_conf": float|None}
    Uses OpenAI if LLM_PROVIDER=openai; otherwise regex fallback.
    """
    msg = (msg or "").strip()
    base = {"cmd": "help", "url": _find_url_in_text(msg), "server_conf": None, "ui_conf": None}

    if LLM_PROVIDER == "openai" and OPENAI_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_KEY)
            prompt = (
                "Extract a JSON with keys: cmd in {agent, help}, url (string or null), "
                "server_conf (0..1 or null), ui_conf (0..1 or null) from the user's message. Only return JSON."
            )
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":prompt},{"role":"user","content":msg}],
                temperature=0
            )
            data = json.loads(r.choices[0].message.content)
            if not data.get("url"):
                data["url"] = base["url"]
            return data
        except Exception as e:
            print(f"[LLM(OpenAI)] parse failed: {e}")

    # Regex fallback (no external API needed)
    out = dict(base)
    m = msg.lower()
    if any(k in m for k in ("run","detect","start","agent","process","scan")):
        out["cmd"] = "agent"
    sc = re.search(r"(server\s*conf(idence)?|threshold)\s*(to|=)\s*([0-1](\.\d+)?)", m)
    uc = re.search(r"(ui\s*conf(idence)?|ui\s*filter)\s*(to|=)\s*([0-1](\.\d+)?)", m)
    if sc: out["server_conf"] = float(sc.group(4))
    if uc: out["ui_conf"]     = float(uc.group(4))
    return out

# -------- gradio callback (single image) --------
def infer(image: Image.Image, server_conf: float, ui_conf: float):
    """
    Heuristic agent:
      1) run detect at server_conf
      2) if zero detections (post UI filter), retry once at server_conf - 0.05
      3) draw results and log every run to CSV
      4) NEW: classify image & draw banner
    """
    if image is None:
        return None, pd.DataFrame([{"message": "Upload an image."}]), "Total detections: 0", None

    prep = _preprocess_image(image)

    rf_resp, status, used_conf, retried = adaptive_detect_once(
        prep, first_conf=server_conf, ui_conf=ui_conf, overlap=0.45, step=0.05, min_conf=0.05
    )

    if status != 200:
        msg = f"API error (HTTP {status}). "
        if isinstance(rf_resp, dict):
            if "error" in rf_resp: msg += str(rf_resp["error"])
            if "_raw" in rf_resp:  msg += " | " + rf_resp["_raw"][:200]
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

    annotated, df, summary, downloadable = draw_detections(prep, rf_resp, ui_conf=ui_conf)

    # ---- NEW: categorize & banner ----
    category, reason = classify_image(df)
    annotated = _draw_banner(
        annotated,
        text=f"Category: {category}",
        category=category,
        position="bottom"
    )

    if downloadable:
        annotated.save(downloadable, format="PNG")

    total = len(df)
    per_cls = _per_class_map(df)
    summary = f"**Category: `{category}`**  \n{summary}  |  conf_used={used_conf:.2f}{' (auto-retry)' if retried else ''}"

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

# -------- UI (tabs) --------
with gr.Blocks(title="Pizza Box Detector — Agentic") as demo:
    gr.Markdown("## 🍕 Pizza Box Detector — Agentic\nSingle image detection, URL agent, and Chat control.\nCategories inferred: **no_box**, **with_pizza**, **torn**, **good_shape**.")

    with gr.Tabs():
        # --- Single Image tab (original flow) ---
        with gr.Tab("Single Image"):
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

        # --- Agent (URL) tab (button-based) ---
        with gr.Tab("Agent (URL)"):
            gr.Markdown("Paste a **Google Doc** (public link) or **webpage** URL with images. The agent fetches images and runs detection on each one.")
            url_in   = gr.Textbox(label="URL", placeholder="https://docs.google.com/document/d/... or https://example.com/page")
            a_sc     = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="Server confidence")
            a_uc     = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="UI filter")
            run_btn  = gr.Button("Run Agent")
            url_out  = gr.JSON(label="Agent result (now includes category per image)")
            url_gallery = gr.Gallery(label="Annotated images (banner shows category)", columns=2, height=400)
            url_zip  = gr.File(label="Download all annotated (zip)")

            def _run_agent(url, sc, uc):
                if not url or not url.strip():
                    return {"ok": False, "message": "Enter a URL first."}, [], None
                result, imgs, zpath = run_agent_from_url(url.strip(), server_conf=sc, ui_conf=uc)
                return result, imgs, zpath

            run_btn.click(_run_agent, inputs=[url_in, a_sc, a_uc], outputs=[url_out, url_gallery, url_zip])

        # --- Chat (regex or OpenAI parses your instruction) ---
        with gr.Tab("Chat"):
            gr.Markdown("Type natural commands like: *run agent at server 0.2 on https://…*")
            chatbox = gr.Chatbot(height=320, label="Chat Agent")  # tuple mode (compatible on Render)
            chat_msg = gr.Textbox(placeholder="e.g., run agent ui=0.3 on https://example.com/products", label="Message")
            chat_btn = gr.Button("Send")
            chat_gallery = gr.Gallery(label="Annotated images", columns=2, height=400)
            chat_zip = gr.File(label="Download all annotated (zip)")

            def chat_handler(history, msg):
                try:
                    user_text = (msg or "").strip()
                    history = list(history or [])
                    if not user_text:
                        history.append(("", "Please type a command (e.g., 'run agent on https://...')."))
                        return history, [], None

                    # show user message
                    history.append((user_text, ""))

                    intent = _llm_parse_freeform(user_text)
                    if (intent.get("cmd") or "help") != "agent":
                        help_text = (
                            "I can fetch images from a Google Doc or webpage and run detection.\n"
                            "Example: 'run agent at server 0.2 on "
                            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/Pizza_box.jpg/640px-Pizza_box.jpg'"
                        )
                        history[-1] = (user_text, help_text)
                        return history, [], None

                    url = (intent.get("url") or "").strip()
                    if not url:
                        history[-1] = (user_text, "Please include a URL (Google Doc or webpage).")
                        return history, [], None

                    sc = intent.get("server_conf")
                    uc = intent.get("ui_conf")
                    sc = sc if isinstance(sc, (int, float)) and 0.0 <= sc <= 1.0 else 0.15
                    uc = uc if isinstance(uc, (int, float)) and 0.0 <= uc <= 1.0 else 0.15

                    result, imgs, zpath = run_agent_from_url(url, server_conf=sc, ui_conf=uc)
                    if not result.get("ok"):
                        history[-1] = (user_text, f"Failed: {result.get('message','unknown error')}")
                        return history, [], None

                    items = result.get("items", [])
                    processed = result.get("images_processed", 0)
                    brief = json.dumps(items, ensure_ascii=False)[:1200]
                    reply = (f"Processed {processed} image(s) at server_conf={sc:.2f}, ui_conf={uc:.2f}.\n"
                             f"Items: {brief}{'...' if len(brief)==1200 else ''} (each item now has a category.)")

                    history[-1] = (user_text, reply)
                    return history, imgs, zpath

                except Exception as e:
                    import traceback; traceback.print_exc()
                    history.append(("", f"Unexpected error: {e}"))
                    return history, [], None

            chat_btn.click(chat_handler, inputs=[chatbox, chat_msg], outputs=[chatbox, chat_gallery, chat_zip])

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
