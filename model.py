"""
ClearCoast AI — Production Cloud Removal Engine v2
====================================================
Multi-stage pipeline: adaptive cloud segmentation, multi-scale inpainting,
LAB-space colour harmonisation, edge-preserving detail recovery, and
distance-aware confidence mapping.

Dependencies: opencv-python-headless, numpy, Pillow (decode only).
"""

import cv2
import numpy as np
from PIL import Image
import io, base64, logging

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Tuning constants
# ═══════════════════════════════════════════════════════════════════════════
PROC_DIM = 1024                # Internal processing resolution (longest edge)
CLOUD_V_THRESH = 175           # HSV V-channel minimum for cloud candidates
CLOUD_S_CEIL = 55              # HSV S-channel maximum for cloud candidates
CLOUD_BLUE_RATIO = 0.345       # Blue-ratio threshold (clouds scatter blue)
CLOUD_WHITENESS_THRESH = 30    # Max channel spread for "near-white" pixels
MORPH_CLOSE_K = 17             # Morphological close kernel size
MORPH_OPEN_K = 9               # Morphological open kernel size
MIN_CLOUD_AREA_FRAC = 0.003    # Minimum contour area as fraction of image
INPAINT_RADIUS_SMALL = 5       # Fine-detail inpaint radius
INPAINT_RADIUS_LARGE = 16      # Large-region inpaint radius
FEATHER_RADIUS = 11            # Gaussian feather radius for mask blending
DETAIL_SIGMA_S = 12            # Edge-preserving filter spatial sigma
DETAIL_SIGMA_R = 0.08          # Edge-preserving filter range sigma
SHARPEN_KERNEL_SIZE = 3        # Laplacian sharpening kernel
SHARPEN_WEIGHT = 0.35          # Sharpening blend factor
CLAHE_CLIP = 1.8               # CLAHE clip limit for local contrast
CLAHE_GRID = 8                 # CLAHE grid size


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resize_to_proc(img: np.ndarray) -> tuple:
    """Resize for processing; return (resized, original_hw, scale_used)."""
    h, w = img.shape[:2]
    if max(h, w) <= PROC_DIM:
        return img.copy(), (h, w), 1.0
    scale = PROC_DIM / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, (h, w), scale


def _upscale_to_original(img: np.ndarray, orig_hw: tuple) -> np.ndarray:
    """Up-scale processed result back to original dimensions."""
    oh, ow = orig_hw
    if img.shape[:2] == (oh, ow):
        return img
    return cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LANCZOS4)


# ── Cloud detection ──────────────────────────────────────────────────────

def _detect_clouds(img_bgr: np.ndarray) -> np.ndarray:
    """
    Four-channel adaptive cloud mask:
      1. HSV brightness + low saturation
      2. Blue-ratio heuristic
      3. Near-white (small inter-channel spread)
      4. Otsu on the V channel for adaptive threshold refinement
    Morphological cleanup + contour filtering to suppress false positives.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    b, g, r = [c.astype(np.float32) for c in cv2.split(img_bgr)]

    # 1. Primary: bright + desaturated
    mask_bright = (v_ch > CLOUD_V_THRESH).astype(np.uint8)
    mask_desat = (s_ch < CLOUD_S_CEIL).astype(np.uint8)
    primary = mask_bright & mask_desat

    # 2. Blue-ratio
    total = b + g + r + 1e-6
    blue_ratio = b / total
    mask_blue = ((blue_ratio > CLOUD_BLUE_RATIO) &
                 (v_ch > 140)).astype(np.uint8)

    # 3. Near-white: all channels close together AND bright
    chan_max = np.maximum(np.maximum(b, g), r)
    chan_min = np.minimum(np.minimum(b, g), r)
    spread = (chan_max - chan_min).astype(np.uint8)
    mask_white = ((spread < CLOUD_WHITENESS_THRESH) &
                  (v_ch > 160)).astype(np.uint8)

    # 4. Otsu adaptive on V-channel to catch variable brightness
    _, otsu = cv2.threshold(v_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Only keep Otsu regions that also have low saturation
    mask_otsu = ((otsu == 255) & (s_ch < CLOUD_S_CEIL + 15)).astype(np.uint8)

    # ── Combine with voting: at least 2 of 4 detectors must agree ──
    vote = primary.astype(np.int16) + mask_blue + mask_white + mask_otsu
    combined = (vote >= 2).astype(np.uint8) * 255

    # ── Morphological refinement ──
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_CLOSE_K, MORPH_CLOSE_K))
    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_OPEN_K, MORPH_OPEN_K))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_open, iterations=1)

    # ── Remove small noise contours ──
    area_thresh = img_bgr.shape[0] * img_bgr.shape[1] * MIN_CLOUD_AREA_FRAC
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < area_thresh:
            cv2.drawContours(combined, [cnt], -1, 0, cv2.FILLED)

    return combined


# ── Feathered blending mask ──────────────────────────────────────────────

def _feather_mask(mask: np.ndarray) -> np.ndarray:
    """Create a soft-edged alpha from the binary mask for seamless blending."""
    k = FEATHER_RADIUS * 2 + 1
    feathered = cv2.GaussianBlur(mask, (k, k), sigmaX=0)
    # Dilate slightly before feathering to cover fringe pixels
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    feathered = cv2.dilate(feathered, dilate_k, iterations=1)
    feathered = cv2.GaussianBlur(feathered, (k, k), sigmaX=0)
    return feathered


# ── Multi-scale inpainting ───────────────────────────────────────────────

def _multiscale_inpaint(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Two-pass inpainting at different radii + blending:
      Pass 1 — small radius: preserves fine edges near cloud borders.
      Pass 2 — large radius: fills interior of big cloud masses.
    Final blend weighted by distance from cloud edge.
    """
    fine = cv2.inpaint(img_bgr, mask, INPAINT_RADIUS_SMALL, cv2.INPAINT_TELEA)
    coarse = cv2.inpaint(img_bgr, mask, INPAINT_RADIUS_LARGE, cv2.INPAINT_NS)

    # Weight: near edges → prefer fine; deep interior → prefer coarse
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = dist.max() if dist.max() > 0 else 1.0
    weight = np.clip(dist / (max_dist * 0.5), 0, 1).astype(np.float32)
    w3 = np.stack([weight] * 3, axis=-1)

    blended = (w3 * coarse.astype(np.float32) +
               (1 - w3) * fine.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


# ── Colour harmonisation (LAB space) ─────────────────────────────────────

def _harmonise_colour(original: np.ndarray, inpainted: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
    """
    Transfer colour statistics from clear regions to inpainted cloud regions
    in LAB colour space for perceptually uniform correction.
    """
    inv = cv2.bitwise_not(mask)
    if cv2.countNonZero(inv) == 0 or cv2.countNonZero(mask) == 0:
        return inpainted

    src_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float64)
    dst_lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB).astype(np.float64)
    result_lab = dst_lab.copy()

    for c in range(3):
        sc = src_lab[:, :, c]
        dc = dst_lab[:, :, c]

        s_mean, s_std = cv2.meanStdDev(sc, mask=inv)
        d_mean, d_std = cv2.meanStdDev(dc, mask=mask)

        s_mean, s_std = s_mean[0][0], max(s_std[0][0], 1e-6)
        d_mean, d_std = d_mean[0][0], max(d_std[0][0], 1e-6)

        corrected = (dc - d_mean) * (s_std / d_std) + s_mean
        result_lab[:, :, c] = np.where(mask == 255, corrected, dc)

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


# ── Edge-preserving detail recovery ──────────────────────────────────────

def _recover_detail(img: np.ndarray) -> np.ndarray:
    """
    Two-stage detail recovery:
      1. Edge-preserving smoothing to remove inpaint smear without killing edges.
      2. Controlled Laplacian sharpening to restore coastline/building crispness.
    """
    # Edge-preserving filter (bilateral approximation via edgePreservingFilter)
    smooth = cv2.edgePreservingFilter(
        img, flags=cv2.RECURS_FILTER,
        sigma_s=DETAIL_SIGMA_S, sigma_r=DETAIL_SIGMA_R)

    # Laplacian sharpening
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=SHARPEN_KERNEL_SIZE)
    lap = np.clip(lap, -255, 255)
    lap_3ch = np.stack([lap] * 3, axis=-1)
    sharpened = smooth.astype(np.float64) - SHARPEN_WEIGHT * lap_3ch
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened


# ── Local contrast enhancement ───────────────────────────────────────────

def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB to boost local contrast without colour shift."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                            tileGridSize=(CLAHE_GRID, CLAHE_GRID))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ── Confidence map ───────────────────────────────────────────────────────

def _build_confidence(mask: np.ndarray, img_bgr: np.ndarray) -> np.ndarray:
    """
    Multi-factor confidence map (float32, 0–1):
      • Distance from cloud edge (further = higher confidence)
      • Local texture energy (textured areas reconstructed better)
      • Clear regions get full 1.0 confidence
    """
    h, w = mask.shape[:2]
    conf = np.ones((h, w), dtype=np.float32)

    if cv2.countNonZero(mask) == 0:
        return conf

    # Factor 1: normalised distance from cloud edge (inside cloud region)
    dist_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_d = dist_inside.max() if dist_inside.max() > 0 else 1.0
    # Deeper inside → lower confidence (inverted)
    dist_factor = 1.0 - np.clip(dist_inside / max_d, 0, 1)
    # Remap: edge of cloud = 0.85, centre of cloud = 0.30
    dist_factor = 0.30 + 0.55 * dist_factor  # range [0.30, 0.85]

    # Factor 2: texture energy (Laplacian variance in local 32×32 blocks)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3))
    # Normalise to [0,1] using a soft ceiling
    tex = np.clip(lap / 60.0, 0, 1).astype(np.float32)
    # Smooth to block level
    tex = cv2.GaussianBlur(tex, (31, 31), 0)
    # Textured reconstructed regions are more reliable
    tex_boost = 0.85 + 0.15 * tex  # range [0.85, 1.0]

    # Combine inside cloud pixels
    cloud_conf = dist_factor * tex_boost
    cloud_conf = np.clip(cloud_conf, 0.20, 0.92)

    # Write: clear pixels stay 1.0, cloud pixels get computed confidence
    conf[mask == 255] = cloud_conf[mask == 255]

    return conf


# ── Alert generation ─────────────────────────────────────────────────────

def _generate_alerts(cloud_pct: float, mask: np.ndarray,
                     confidence: np.ndarray) -> list:
    """Rich, context-aware dynamic alerts."""
    alerts = []
    h, w = mask.shape[:2]

    # ── Cloud severity ──
    if cloud_pct > 80:
        alerts.append(
            "🔴 Critical cloud cover (>{:.0f}%). AI reconstruction is highly "
            "speculative — treat results as approximate.".format(cloud_pct))
    elif cloud_pct > 55:
        alerts.append(
            "🟠 Heavy cloud cover ({:.1f}%). Significant inpainting applied; "
            "artefacts likely in large occluded areas.".format(cloud_pct))
    elif cloud_pct > 30:
        alerts.append(
            "🟡 Moderate cloud cover ({:.1f}%). Results are generally reliable "
            "but verify critical features manually.".format(cloud_pct))
    elif cloud_pct > 8:
        alerts.append(
            "🟢 Light cloud cover ({:.1f}%). High-quality reconstruction "
            "expected.".format(cloud_pct))
    else:
        alerts.append(
            "✅ Near-clear image ({:.1f}% cloud). Minimal processing "
            "applied.".format(cloud_pct))

    # ── Confidence warning ──
    if cv2.countNonZero(mask) > 0:
        mean_conf = float(np.mean(confidence[mask == 255]))
        if mean_conf < 0.50:
            alerts.append(
                "⚠️ Mean reconstruction confidence is low ({:.0f}%). "
                "Large continuous cloud regions limit accuracy.".format(
                    mean_conf * 100))
        elif mean_conf < 0.70:
            alerts.append(
                "ℹ️ Mean reconstruction confidence: {:.0f}%. "
                "Moderately sized cloud patches detected.".format(
                    mean_conf * 100))

    # ── Spatial distribution ──
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    n_clusters = len(contours)
    if n_clusters > 12:
        alerts.append(
            "☁️ Scattered cloud pattern ({} clusters). Reconstruction quality "
            "is typically higher for scattered clouds.".format(n_clusters))
    elif n_clusters > 0 and cloud_pct > 40:
        alerts.append(
            "☁️ {} dense cloud mass(es) detected. Edge regions have higher "
            "confidence than interiors.".format(n_clusters))

    # ── Recommendation ──
    if cloud_pct > 45:
        alerts.append(
            "🔍 Recommendation: cross-reference with a temporally adjacent "
            "clear-sky acquisition for validation.")
    if cloud_pct > 60:
        alerts.append(
            "📡 Consider multi-temporal compositing for operational "
            "decision-making in heavily clouded zones.")

    return alerts


# ═══════════════════════════════════════════════════════════════════════════
# Public class API  (requested signature)
# ═══════════════════════════════════════════════════════════════════════════

class CloudRemovalModel:
    """
    Stateless cloud-removal processor.

    Usage:
        model = CloudRemovalModel()
        clear, confidence, alerts = model.process(img_uint8)
    """

    def process(self, img_uint8: np.ndarray):
        """
        Parameters
        ----------
        img_uint8 : np.ndarray
            BGR uint8 image (any resolution, up to ~500 MP).

        Returns
        -------
        clear_image_uint8 : np.ndarray   – Cloud-removed BGR uint8 image
                                            (same resolution as input).
        confidence_map    : np.ndarray   – float32 [0,1], same H×W as input.
        dynamic_alerts    : list[str]    – Context-aware alert strings.
        """
        orig_h, orig_w = img_uint8.shape[:2]

        # ── 1. Down-sample for speed ──
        proc, orig_hw, scale = _resize_to_proc(img_uint8)
        ph, pw = proc.shape[:2]

        # ── 2. Cloud detection ──
        mask = _detect_clouds(proc)
        cloud_pct = round(float(mask.sum()) / 255.0 / (ph * pw) * 100, 1)
        logger.info("Cloud cover: %.1f%%  (%d×%d → %d×%d)",
                    cloud_pct, orig_w, orig_h, pw, ph)

        # ── 3. Multi-scale inpainting ──
        inpainted = _multiscale_inpaint(proc, mask)

        # ── 4. LAB colour harmonisation ──
        harmonised = _harmonise_colour(proc, inpainted, mask)

        # ── 5. Seamless feathered blend ──
        feathered = _feather_mask(mask).astype(np.float32) / 255.0
        alpha3 = np.stack([feathered] * 3, axis=-1)
        blended = (alpha3 * harmonised.astype(np.float32) +
                   (1 - alpha3) * proc.astype(np.float32))
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # ── 6. Edge-preserving detail recovery + sharpening ──
        detailed = _recover_detail(blended)

        # ── 7. Local contrast enhancement (CLAHE) ──
        enhanced = _enhance_contrast(detailed)

        # ── 8. Confidence map ──
        confidence = _build_confidence(mask, enhanced)

        # ── 9. Alerts ──
        alerts = _generate_alerts(cloud_pct, mask, confidence)

        # ── 10. Up-scale back to original resolution ──
        if scale < 1.0:
            result_full = _upscale_to_original(enhanced, (orig_h, orig_w))
            conf_full = cv2.resize(confidence, (orig_w, orig_h),
                                   interpolation=cv2.INTER_LINEAR)
        else:
            result_full = enhanced
            conf_full = confidence

        return result_full, conf_full, alerts


# ═══════════════════════════════════════════════════════════════════════════
# Functional API  (used by app.py  —  DO NOT CHANGE SIGNATURE)
# ═══════════════════════════════════════════════════════════════════════════

_model = CloudRemovalModel()


def process_image(image_bytes: bytes) -> dict:
    """
    Decode raw image bytes, run the cloud-removal pipeline, and return
    base-64 encoded results ready for JSON transport.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded image (JPEG / PNG / TIFF).

    Returns
    -------
    dict with keys:
        original_b64   – base-64 PNG of the (resized) original
        result_b64     – base-64 PNG of the cloud-removed result
        confidence_b64 – base-64 PNG of the confidence heat-map
        cloud_pct      – float, estimated cloud-cover percentage
        alerts         – list[str], context-aware alert strings
    """
    # Decode
    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # Prepare display-size original (capped at PROC_DIM for the UI)
    disp, _, _ = _resize_to_proc(img_bgr)

    # Run pipeline
    result, confidence, alerts = _model.process(img_bgr)

    # Resize result to match display original for side-by-side comparison
    disp_h, disp_w = disp.shape[:2]
    result_disp = cv2.resize(result, (disp_w, disp_h),
                             interpolation=cv2.INTER_LANCZOS4)

    # Cloud percentage (from the processing-size mask)
    proc, _, _ = _resize_to_proc(img_bgr)
    mask = _detect_clouds(proc)
    ph, pw = proc.shape[:2]
    cloud_pct = round(float(mask.sum()) / 255.0 / (ph * pw) * 100, 1)

    # Confidence colour map
    conf_disp = cv2.resize(confidence, (disp_w, disp_h),
                           interpolation=cv2.INTER_LINEAR)
    conf_u8 = (conf_disp * 255).astype(np.uint8)
    conf_color = cv2.applyColorMap(conf_u8, cv2.COLORMAP_INFERNO)

    # Encode
    def _to_b64(arr_bgr):
        _, buf = cv2.imencode(".png", arr_bgr)
        return base64.b64encode(buf).decode("utf-8")

    return {
        "original_b64": _to_b64(disp),
        "result_b64": _to_b64(result_disp),
        "confidence_b64": _to_b64(conf_color),
        "cloud_pct": cloud_pct,
        "alerts": alerts,
    }
