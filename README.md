# 🎨 ComfyUI-Egregora-Adaptive-Colorfix

**Color Fix Adaptive Chroma Fusion** is a custom ComfyUI node for reference-guided color correction.

It is designed for cases where simple color transfer methods often break down: tiled upscaling, restoration, enhancement, harmonization, and other workflows where the **reference image and target image do not share the exact same structure**.

This version focuses on a simpler interface, stronger default behavior, and better practical stability.

---

## ✨ What this node does

This node is especially useful when you want to:

- match the color mood of a reference image more reliably
- reduce visible tile-to-tile color variation in upscaling workflows
- improve background consistency without destroying edges
- preserve local detail while still pushing stronger global correction
- avoid the common trade-off of:
  - **Wavelet** → good color identity, but halos and edge artifacts
  - **AdaIN** → stable global tone, but washed-out or averaged-looking colors
  - **simple Lab / stats transfer** → broad correction, but poor local behavior

In short, the node tries to combine:

- the **color fidelity** often associated with wavelet methods
- the **stability** of global statistical approaches like AdaIN
- stronger **edge protection**
- smarter **luminance control**
- more robust behavior in images with **structural mismatch**

This node transfers the **color behavior** of a reference image to a target image while trying to avoid the most common failure modes of naive color matching:

- halos near edges
- color spill across contours
- washed-out global matching
- unstable local correction
- luminance contamination from structural mismatch

Instead of using only one method, the node combines **global chroma anchoring**, **multiscale chroma transfer**, **edge-aware protection**, **confidence maps**, and **base/detail luminance control**.

The goal is simple:

> **make the target inherit the color feel of the reference without forcing the reference structure onto it.**

---

## 🎯 Best use cases

This node is especially useful for:

- tiled upscaling workflows
- reference-guided color harmonization
- restoration and enhancement pipelines
- fixing tile-to-tile color drift
- improving background and large-surface consistency
- cases where Wavelet looks vivid but unstable
- cases where AdaIN looks stable but too averaged

---

## 🧠 What happens under the hood

The node works in several stages.

### 1. 🌈 RGB → Lab conversion

Both images are converted to **Lab space**.

This matters because the node can treat:

- **L** = luminance
- **a / b** = chroma

separately.

That separation is one of the main reasons it behaves better than simple all-in-one color transfer.

### 2. ⚓ Global chroma anchoring

Before any local correction, the node performs a **global chroma stats transfer** from the reference to the target.

This gives the target a stable overall color direction without immediately forcing local structure.

Think of this as the first coarse alignment step.

### 3. 🧩 Proxy confidence analysis at reduced resolution

Part of the chroma analysis is computed on an internal proxy image capped at **1024 px** on the longest side.

At this stage, the node builds confidence maps that estimate where correction is more trustworthy by comparing:

- low-frequency luminance similarity
- gradient similarity
- local chroma compatibility

This reduces cost while still preserving the broad spatial logic needed for color transfer.

### 4. 🌊 Multiscale chroma transfer

The chroma correction is not produced from a single source.

It mixes two complementary components:

- **wavelet low/mid-frequency chroma delta**
- **Gaussian multiscale chroma delta**

This is important because each component does something different:

- the **wavelet path** helps preserve color identity
- the **Gaussian path** helps stabilize the transfer spatially

Together, they allow stronger color correction without leaning entirely on a single method.

### 5. 🛡️ Edge-aware safety masks

Once the confidence maps are available, the node computes **edge safety masks** in full resolution.

These masks reduce correction strength near areas that are more likely to break visually, such as:

- strong contours
- unstable structural boundaries
- low-confidence local zones
- regions with higher risk of color bleed

This is one of the most important parts of the node.

It is what helps prevent:

- halos
- contour contamination
- false edge tinting
- unstable correction near seams and borders

### 6. 💡 Base/detail luminance transfer

Luminance is handled separately from chroma.

Instead of aggressively replacing the target luminance, the node conceptually splits it into:

- **base luminance**
- **detail luminance**

Then it pushes the **base** more strongly toward the reference in broad, safer regions while preserving local detail from the target.

This is especially useful for:

- flatter backgrounds
- large surfaces
- smoother global lighting consistency

without bringing back the classic artifacts that happen when luminance is transferred too directly.

### 7. 🧴 Saturation preservation

After the main correction, the node applies a saturation safeguard.

This helps reduce the risk of:

- gray-looking output
- washed chroma
- over-neutralized color regions

The safeguard is intentionally conservative in this version and is fixed internally.

### 8. 🧼 Final guided smoothing and safety limits

Before converting back to RGB, the node applies a light guided-style smoothing step and clamps the maximum luminance/chroma shift.

This final stage helps keep the correction controlled and reduces unstable spikes.

---

## 🎛️ Inputs

### `image_ref`
Reference image that provides the desired color behavior.

### `image_target`
Target image that will receive the correction.

### `edge_safety`
Controls how aggressively the node protects edges and structurally unstable regions.

**Lower values:**
- stronger correction
- less protection
- more aggressive behavior

**Higher values:**
- safer edges
- less spill
- more conservative transfer

**Range:** `0.0 → 3.0`

### `luma_match`
Controls how strongly the node aligns broad luminance behavior with the reference.

**Lower values:**
- more target-preserving luminance
- weaker global brightness alignment

**Higher values:**
- stronger large-field luminance matching
- more visible influence from the reference in broad areas

**Range:** `0.0 → 3.0`

---

## 🔒 Fixed internal settings in this version

To keep the node faster and easier to use, several controls are intentionally hardcoded:

- `color_strength = 1.0`
- `local_detail = 1.0`
- `saturation_guard = 1.0`
- `internal_max_res = 1024`

Other internal choices currently used:

- `wavelet = db2`
- `wavelet level = 1`
- `guided radius = 12`

This means the node now exposes only the two controls that most directly change real-world behavior during use.

---

## ✅ Practical strengths

- strong balance between global color harmonization and local protection
- useful in tiled upscale workflows
- more robust than naive mean/std or simple Lab transfer
- more controlled than broad average-style color matching
- better edge safety than many direct transfer approaches
- simplified UI with meaningful controls only

---

## ⚠️ Notes

- This node is meant for **color behavior transfer**, not geometric or structural matching.
- It works best when the reference provides a desirable color mood, palette, or lighting tendency.
- Extremely mismatched images may still require some tuning.
- Higher `edge_safety` is usually safer when the reference and target have stronger structural differences.
- Higher `luma_match` is more useful when broad surfaces or backgrounds need better tonal consistency.

---

## 🚀 Installation

Clone the repository into your `ComfyUI/custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lucasgattas/ComfyUI-Egregora-Adaptive-Colorfix.git
```

Then restart ComfyUI.

If needed, install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Dependencies

This node uses:

- `numpy`
- `opencv-python`
- `PyWavelets`
- `torch`

---

## 🧱 Project structure

```text
ComfyUI-Egregora-Adaptive-Colorfix/
├── __init__.py
├── egregora_adaptive_colorfix_node.py
├── README.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## 🏷️ Node name in ComfyUI

The node appears in ComfyUI as:

**🎨 Color Fix Adaptive Chroma Fusion**

Category:

**`image/colorfix`**

---

## 📝 Changelog

### Current simplified version

- ✨ simplified the public UI to **two exposed controls only**
- 🛡️ kept **`edge_safety`** as the main protection control
- 💡 kept **`luma_match`** as the main luminance control
- 🔒 hardcoded `color_strength = 1.0`
- 🔒 hardcoded `local_detail = 1.0`
- 🔒 hardcoded `saturation_guard = 1.0`
- 🔒 hardcoded `internal_max_res = 1024`
- 📈 increased caps for:
  - `edge_safety` → `3.0`
  - `luma_match` → `3.0`
- 🧼 reduced UI clutter and removed redundant tuning for typical workflows
- 🎯 kept the version that was visually more reliable in testing than the more aggressive hybrid luma experiments

### Earlier direction

- exposed more internal controls
- allowed broader manual tuning
- had a more parameter-heavy workflow
- was more flexible, but also easier to overtune and harder to keep consistent

---

## ❤️ Credits

Developed for the ComfyUI workflow ecosystem.

If this node helps your workflow, consider starring the repository ⭐
