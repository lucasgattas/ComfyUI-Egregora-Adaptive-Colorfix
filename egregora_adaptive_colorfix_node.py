import numpy as np
import cv2
import pywt
import torch


def _ensure_bhwc_tensor(image):
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(image)}")

    image = image.float()

    if image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim != 4:
        raise ValueError(f"Expected image ndim 3 or 4, got shape {tuple(image.shape)}")

    return image.clamp(0.0, 1.0).contiguous()


def _tensor_batch_to_np_list(image):
    image = _ensure_bhwc_tensor(image)
    arr = image.detach().cpu().numpy()
    arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    return [np.ascontiguousarray(arr[i]) for i in range(arr.shape[0])]


def _np_list_to_tensor(images):
    if len(images) == 0:
        raise ValueError("Empty image list")

    out = []
    for img in images:
        img = np.ascontiguousarray(np.clip(img.astype(np.float32), 0.0, 1.0))
        out.append(torch.from_numpy(img))

    return torch.stack(out, dim=0).contiguous().float()


def _match_reference_list_to_target_list(ref_list, target_list):
    if len(ref_list) == len(target_list):
        return ref_list
    if len(ref_list) == 1 and len(target_list) > 1:
        return [ref_list[0] for _ in range(len(target_list))]
    raise ValueError(
        f"Batch mismatch: image_ref has {len(ref_list)} image(s), "
        f"image_target has {len(target_list)} image(s). "
        f"Use same batch size, or a single reference image."
    )


def _rgb_to_lab01(rgb):
    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] /= 255.0
    lab[..., 1] = (lab[..., 1] - 128.0) / 127.0
    lab[..., 2] = (lab[..., 2] - 128.0) / 127.0
    return lab


def _lab01_to_rgb(lab):
    lab = lab.copy().astype(np.float32)
    lab[..., 0] = np.clip(lab[..., 0], 0.0, 1.0) * 255.0
    lab[..., 1] = np.clip(lab[..., 1], -1.0, 1.0) * 127.0 + 128.0
    lab[..., 2] = np.clip(lab[..., 2], -1.0, 1.0) * 127.0 + 128.0
    lab8 = np.clip(lab, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab8, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)


def _safe_std(x, eps=1e-6):
    return max(float(np.std(x)), eps)


def _resize_like(img, ref_shape):
    h, w = ref_shape[:2]
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)


def _normalize01(x, eps=1e-6):
    mn = float(np.min(x))
    mx = float(np.max(x))
    return (x - mn) / max(mx - mn, eps)


def _sobel_mag(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def _sobel_xy(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def _local_mean(x, sigma):
    return cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigma)


def _gauss_delta(src, ref, sigma):
    sb = cv2.GaussianBlur(src.astype(np.float32), (0, 0), sigma)
    rb = cv2.GaussianBlur(ref.astype(np.float32), (0, 0), sigma)
    return rb - sb


def _channel_stats_transfer(src, ref, strength=1.0):
    out = src.copy().astype(np.float32)
    for c in range(src.shape[2]):
        s = src[..., c]
        r = ref[..., c]
        s_mean, s_std = float(np.mean(s)), _safe_std(s)
        r_mean, r_std = float(np.mean(r)), _safe_std(r)
        matched = (s - s_mean) * (r_std / s_std) + r_mean
        out[..., c] = s * (1.0 - strength) + matched * strength
    return out


def _guided_like_smooth(guide_gray, src, radius=12, eps=0.005):
    out = np.zeros_like(src)
    sigma_space = max(int(radius), 1)
    sigma_color = max(float(eps) * 255.0 * 4.0, 1.0)

    for c in range(src.shape[2]):
        ch = src[..., c].astype(np.float32)
        filtered = cv2.bilateralFilter(ch, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        out[..., c] = filtered

    edge = _normalize01(_sobel_mag(guide_gray.astype(np.float32)))
    edge = edge[..., None]
    mix = np.clip(1.0 - edge * 0.85, 0.10, 1.0)
    return src * (1.0 - mix) + out * mix


def _clamp_delta(base, modified, max_shift):
    delta = modified - base
    delta = np.clip(delta, -max_shift, max_shift)
    return base + delta


def _soft_limit_delta(delta, limit):
    if limit <= 0:
        return np.zeros_like(delta)
    return np.tanh(delta / max(limit, 1e-6)) * limit


def _wavelet_lowmid_delta(src, ref, wavelet="db2", level=2, low_strength=0.4, mid_strength=0.4):
    coeffs_s = pywt.wavedec2(src, wavelet=wavelet, level=level)
    coeffs_r = pywt.wavedec2(ref, wavelet=wavelet, level=level)

    out_coeffs = []

    cA_s = coeffs_s[0]
    cA_r = coeffs_r[0]
    cA = cA_s + (cA_r - cA_s) * low_strength
    out_coeffs.append(cA)

    n_detail = len(coeffs_s) - 1
    for i, (ds, dr) in enumerate(zip(coeffs_s[1:], coeffs_r[1:])):
        if i < n_detail - 1:
            strength = mid_strength
        else:
            strength = 0.0
        merged = tuple(a + (b - a) * strength for a, b in zip(ds, dr))
        out_coeffs.append(merged)

    rec = pywt.waverec2(out_coeffs, wavelet=wavelet)
    rec = rec[:src.shape[0], :src.shape[1]]
    return rec.astype(np.float32) - src.astype(np.float32)


def _compute_confidence_maps(target_lab, ref_lab, structure_sigma=1.0, chroma_sigma=2.0):
    tL = target_lab[..., 0]
    rL = _resize_like(ref_lab[..., 0], target_lab.shape)

    tL_low = _local_mean(tL, structure_sigma)
    rL_low = _local_mean(rL, structure_sigma)
    low_diff = np.abs(tL_low - rL_low)
    low_conf = 1.0 - _normalize01(low_diff)

    tg = _sobel_mag(tL)
    rg = _sobel_mag(rL)
    grad_diff = np.abs(tg - rg)
    grad_conf = 1.0 - _normalize01(grad_diff)

    ta = target_lab[..., 1]
    tb = target_lab[..., 2]
    ra = _resize_like(ref_lab[..., 1], target_lab.shape)
    rb = _resize_like(ref_lab[..., 2], target_lab.shape)

    ta_low = _local_mean(ta, chroma_sigma)
    tb_low = _local_mean(tb, chroma_sigma)
    ra_low = _local_mean(ra, chroma_sigma)
    rb_low = _local_mean(rb, chroma_sigma)

    chroma_dist = np.sqrt((ta_low - ra_low) ** 2 + (tb_low - rb_low) ** 2)
    chroma_compat = 1.0 - _normalize01(chroma_dist)

    local_conf = np.clip(0.45 * low_conf + 0.35 * grad_conf + 0.20 * chroma_compat, 0.0, 1.0)
    chroma_conf = np.clip(0.30 * low_conf + 0.20 * grad_conf + 0.50 * chroma_compat, 0.0, 1.0)

    local_conf = cv2.GaussianBlur(local_conf.astype(np.float32), (0, 0), 2.0)
    chroma_conf = cv2.GaussianBlur(chroma_conf.astype(np.float32), (0, 0), 2.0)
    edge = _normalize01(tg).astype(np.float32)
    return local_conf.astype(np.float32), chroma_conf.astype(np.float32), edge


def _make_edge_safety_mask(edge, local_conf, target_lab, edge_safety=0.5):
    tL = target_lab[..., 0]
    tA = target_lab[..., 1]
    tB = target_lab[..., 2]

    chroma = np.sqrt(tA * tA + tB * tB)

    edge_core = np.power(np.clip(edge, 0.0, 1.0), 0.85)
    edge_wide = cv2.GaussianBlur(edge_core.astype(np.float32), (0, 0), 1.25 + 1.25 * edge_safety)
    edge_wide = np.clip(edge_wide, 0.0, 1.0)

    luma_grad = _normalize01(_sobel_mag(tL))
    luma_grad = cv2.GaussianBlur(luma_grad.astype(np.float32), (0, 0), 0.8 + 0.8 * edge_safety)

    low_conf_inv = 1.0 - local_conf
    low_conf_inv = cv2.GaussianBlur(low_conf_inv.astype(np.float32), (0, 0), 1.0)

    low_chroma = 1.0 - np.clip(chroma / 0.22, 0.0, 1.0)
    low_chroma = cv2.GaussianBlur(low_chroma.astype(np.float32), (0, 0), 0.8)

    protect = (
        edge_wide * (0.55 + 0.45 * edge_safety) +
        luma_grad * 0.25 +
        low_conf_inv * 0.35 +
        low_chroma * 0.15
    )

    protect = np.clip(protect, 0.0, 1.0)
    protect = cv2.GaussianBlur(protect.astype(np.float32), (0, 0), 0.9 + 1.1 * edge_safety)

    chroma_gate = np.clip(1.0 - protect * (0.72 + 0.18 * edge_safety), 0.0, 1.0)
    luma_gate = np.clip(1.0 - protect * (0.92 + 0.18 * edge_safety), 0.0, 1.0)

    return chroma_gate.astype(np.float32), luma_gate.astype(np.float32), protect.astype(np.float32)


def _make_strict_luma_gate(target_luma, ref_luma, local_conf, edge_safety=0.5):
    tg = _normalize01(_sobel_mag(target_luma))
    rg = _normalize01(_sobel_mag(ref_luma))

    tgx, tgy = _sobel_xy(target_luma)
    rgx, rgy = _sobel_xy(ref_luma)

    dot = tgx * rgx + tgy * rgy
    tn = np.sqrt(tgx * tgx + tgy * tgy)
    rn = np.sqrt(rgx * rgx + rgy * rgy)
    cos = dot / np.maximum(tn * rn, 1e-6)
    orient_mismatch = 1.0 - np.clip((cos + 1.0) * 0.5, 0.0, 1.0)
    orient_mismatch = cv2.GaussianBlur(orient_mismatch.astype(np.float32), (0, 0), 1.0)

    detail_sigma = 1.5 + edge_safety * 1.25
    t_base_small = _local_mean(target_luma, detail_sigma)
    r_base_small = _local_mean(ref_luma, detail_sigma)
    t_detail = target_luma - t_base_small
    r_detail = ref_luma - r_base_small
    detail_diff = np.abs(t_detail - r_detail)
    detail_mismatch = _normalize01(detail_diff)
    detail_mismatch = cv2.GaussianBlur(detail_mismatch.astype(np.float32), (0, 0), 1.0)

    edge_mismatch = np.abs(tg - rg)
    edge_mismatch = cv2.GaussianBlur(edge_mismatch.astype(np.float32), (0, 0), 1.0)

    low_conf_inv = 1.0 - local_conf
    low_conf_inv = cv2.GaussianBlur(low_conf_inv.astype(np.float32), (0, 0), 1.0)

    structure_risk = (
        edge_mismatch * 0.35 +
        orient_mismatch * 0.30 +
        detail_mismatch * 0.35
    )
    structure_risk = cv2.GaussianBlur(structure_risk.astype(np.float32), (0, 0), 0.8 + edge_safety * 0.8)

    protect = np.clip(
        structure_risk * (0.95 + 0.45 * edge_safety) +
        tg * (0.30 + 0.25 * edge_safety) +
        low_conf_inv * 0.35,
        0.0,
        1.0,
    )

    protect = cv2.GaussianBlur(protect.astype(np.float32), (0, 0), 1.0 + 1.2 * edge_safety)
    strict_gate = np.clip(1.0 - protect, 0.0, 1.0)

    return strict_gate.astype(np.float32), protect.astype(np.float32)


def _large_field_mask(target_luma, target_lab, edge_safety=0.5):
    grad = _normalize01(_sobel_mag(target_luma))
    grad_blur = cv2.GaussianBlur(grad.astype(np.float32), (0, 0), 2.0 + 2.0 * edge_safety)

    ta = target_lab[..., 1]
    tb = target_lab[..., 2]
    chroma = np.sqrt(ta * ta + tb * tb)
    chroma_tex = _normalize01(np.abs(chroma - _local_mean(chroma, 3.0)))
    chroma_tex = cv2.GaussianBlur(chroma_tex.astype(np.float32), (0, 0), 2.0)

    base = 1.0 - np.clip(grad_blur * 0.85 + chroma_tex * 0.35, 0.0, 1.0)
    base = cv2.GaussianBlur(base.astype(np.float32), (0, 0), 2.5 + 1.5 * edge_safety)
    return np.clip(base, 0.0, 1.0).astype(np.float32)


def _preserve_saturation(
    target_lab,
    corrected_lab,
    ref_lab,
    min_chroma_preservation=1.0,
    chroma_recovery_strength=1.0,
    neutral_threshold=0.5,
    neutral_protection=1.0,
):
    out = corrected_lab.copy()

    t_ch = np.sqrt(target_lab[..., 1] ** 2 + target_lab[..., 2] ** 2)
    o_ch = np.sqrt(corrected_lab[..., 1] ** 2 + corrected_lab[..., 2] ** 2)
    r_ch = np.sqrt(ref_lab[..., 1] ** 2 + ref_lab[..., 2] ** 2)

    chroma_floor = np.maximum(t_ch * min_chroma_preservation, np.minimum(r_ch, t_ch) * 0.60)
    loss_ratio = np.clip((chroma_floor - o_ch) / np.maximum(chroma_floor, 1e-6), 0.0, 1.0)

    neutral_mask = (t_ch < neutral_threshold).astype(np.float32)
    recovery_mask = loss_ratio * (1.0 - neutral_mask * neutral_protection)
    recovery_mask = cv2.GaussianBlur(recovery_mask.astype(np.float32), (0, 0), 1.5)

    recover = np.clip(recovery_mask * chroma_recovery_strength, 0.0, 1.0)[..., None]
    out[..., 1:3] = corrected_lab[..., 1:3] * (1.0 - recover) + target_lab[..., 1:3] * recover
    return out


def _luma_base_detail_transfer(
    target_luma,
    ref_luma,
    target_lab,
    local_conf,
    luma_gate,
    edge_safety=0.5,
    luma_match=1.5,
    max_luma_shift=0.5,
):
    large_field = _large_field_mask(target_luma, target_lab, edge_safety=edge_safety)

    strict_gate, _ = _make_strict_luma_gate(
        target_luma=target_luma,
        ref_luma=ref_luma,
        local_conf=local_conf,
        edge_safety=edge_safety,
    )

    sigma_base = 5.0 + 5.0 * luma_match
    t_base = _local_mean(target_luma, sigma_base)
    r_base = _local_mean(ref_luma, sigma_base)
    t_detail = target_luma - t_base

    regional_gate = np.clip(
        local_conf * 0.35 +
        luma_gate * 0.20 +
        strict_gate * 0.45,
        0.0,
        1.0,
    )

    allow = np.clip(large_field * regional_gate, 0.0, 1.0)

    base_delta = (r_base - t_base) * allow * luma_match
    flat_limit = max_luma_shift * (0.85 + 0.55 * large_field)
    edge_limit = max_luma_shift * (0.18 + 0.22 * strict_gate)
    adaptive_limit = np.maximum(edge_limit, flat_limit * (0.35 + 0.65 * allow))
    base_delta = np.tanh(base_delta / np.maximum(adaptive_limit, 1e-6)) * adaptive_limit

    mixed_base = t_base + base_delta
    out_luma = mixed_base + t_detail

    return out_luma.astype(np.float32), allow.astype(np.float32), strict_gate.astype(np.float32)


def adaptive_chroma_fusion(
    target_rgb,
    ref_rgb,
    color_strength=1.0,
    edge_safety=0.5,
    local_detail=1.0,
    luma_match=1.5,
    saturation_guard=1.0,
):
    wavelet = "db2"
    level = 2

    global_chroma_strength = 1.0 * color_strength
    low_freq_strength = 0.4 * local_detail
    mid_freq_strength = 0.4 * local_detail
    pyramid_strength = 1.0 * color_strength

    structure_sigma = 1.0
    chroma_sigma = 2.0
    structure_sensitivity = 3.0 * max(edge_safety, 0.01)

    guided_radius = 12
    guided_eps = 0.005

    max_luma_shift = 0.5
    max_chroma_shift = 1.0

    min_chroma_preservation = 1.0 * saturation_guard
    chroma_recovery_strength = 1.0 * saturation_guard
    neutral_threshold = 0.5
    neutral_protection = 1.0

    t_lab = _rgb_to_lab01(target_rgb)
    r_lab = _rgb_to_lab01(ref_rgb)
    r_lab = _resize_like(r_lab, t_lab.shape)

    base = t_lab.copy()
    base[..., 1:3] = _channel_stats_transfer(t_lab[..., 1:3], r_lab[..., 1:3], strength=global_chroma_strength)

    local_conf, chroma_conf, edge = _compute_confidence_maps(
        t_lab,
        r_lab,
        structure_sigma=structure_sigma,
        chroma_sigma=chroma_sigma,
    )

    local_conf = np.clip(local_conf ** max(structure_sensitivity, 1e-4), 0.0, 1.0)
    chroma_gate, luma_gate, _ = _make_edge_safety_mask(
        edge=edge,
        local_conf=local_conf,
        target_lab=t_lab,
        edge_safety=edge_safety,
    )

    out = base.copy()

    for c in [1, 2]:
        src = base[..., c]
        ref = r_lab[..., c]

        dw = _wavelet_lowmid_delta(
            src,
            ref,
            wavelet=wavelet,
            level=level,
            low_strength=low_freq_strength,
            mid_strength=mid_freq_strength,
        )

        dp = (
            _gauss_delta(src, ref, 2.0) * 0.45 +
            _gauss_delta(src, ref, 4.0) * 0.35 +
            _gauss_delta(src, ref, 8.0) * 0.20
        ) * pyramid_strength

        delta = (dw * local_conf + dp * chroma_conf) * chroma_gate
        delta = _soft_limit_delta(delta, max_chroma_shift)
        out[..., c] = src + delta

    tL = t_lab[..., 0]
    rL = r_lab[..., 0]

    out_luma, luma_allow, strict_luma_gate = _luma_base_detail_transfer(
        target_luma=tL,
        ref_luma=rL,
        target_lab=t_lab,
        local_conf=local_conf,
        luma_gate=luma_gate,
        edge_safety=edge_safety,
        luma_match=luma_match,
        max_luma_shift=max_luma_shift,
    )
    out[..., 0] = out_luma

    delta_lab = out - t_lab
    delta_lab[..., 0] *= np.clip(luma_allow * strict_luma_gate, 0.0, 1.0)
    delta_lab[..., 1] *= chroma_gate
    delta_lab[..., 2] *= chroma_gate

    delta_lab = _guided_like_smooth(t_lab[..., 0], delta_lab, radius=guided_radius, eps=guided_eps)
    out = t_lab + delta_lab

    out = _preserve_saturation(
        t_lab,
        out,
        r_lab,
        min_chroma_preservation=min_chroma_preservation,
        chroma_recovery_strength=chroma_recovery_strength,
        neutral_threshold=neutral_threshold,
        neutral_protection=neutral_protection,
    )

    out[..., 0] = _clamp_delta(t_lab[..., 0], out[..., 0], max_luma_shift)
    out[..., 1] = _clamp_delta(t_lab[..., 1], out[..., 1], max_chroma_shift)
    out[..., 2] = _clamp_delta(t_lab[..., 2], out[..., 2], max_chroma_shift)

    return _lab01_to_rgb(out)


def _process_batch(image_ref, image_target, processor, **kwargs):
    ref_list = _tensor_batch_to_np_list(image_ref)
    target_list = _tensor_batch_to_np_list(image_target)
    ref_list = _match_reference_list_to_target_list(ref_list, target_list)

    out_list = []
    for ref, target in zip(ref_list, target_list):
        out_list.append(processor(target, ref, **kwargs))

    return _np_list_to_tensor(out_list)


class ColorFixAdaptiveChromaFusion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "color_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "edge_safety": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "local_detail": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "luma_match": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "saturation_guard": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/colorfix"

    def apply(
        self,
        image_ref,
        image_target,
        color_strength,
        edge_safety,
        local_detail,
        luma_match,
        saturation_guard,
    ):
        out = _process_batch(
            image_ref,
            image_target,
            adaptive_chroma_fusion,
            color_strength=color_strength,
            edge_safety=edge_safety,
            local_detail=local_detail,
            luma_match=luma_match,
            saturation_guard=saturation_guard,
        )
        return (out,)


NODE_CLASS_MAPPINGS = {
    "ColorFixAdaptiveChromaFusion": ColorFixAdaptiveChromaFusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorFixAdaptiveChromaFusion": "🎨 Color Fix Adaptive Chroma Fusion",
}