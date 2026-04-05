# 🎨 ComfyUI-Egregora-Adaptive-Colorfix

**Color Fix Adaptive Chroma Fusion** is a custom node for **ComfyUI** designed to improve color consistency between a **reference image** and a **target image**, especially in workflows where traditional color-matching methods tend to break down.

It was developed with a practical goal in mind: achieve a result that is often more stable and visually pleasing than **wavelet-only**, **AdaIN**, **Lab-based global matching**, and similar approaches when the image structure is **not perfectly aligned**, which is very common in AI image generation, tiled upscaling, restoration, and enhancement pipelines.

---

## ✨ What this node is for

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

---

## 🧠 Core idea behind the node

The node uses an **adaptive chroma fusion** strategy.

Instead of relying on a single color-transfer method, it combines several ideas:

### 1. Global chroma correction
A global color adjustment brings the target closer to the reference in a stable way.

### 2. Multiscale chroma transfer
It mixes:
- a **wavelet low/mid-frequency chroma delta**
- a **Gaussian multiscale chroma delta**

This helps preserve the reference color identity without forcing high-frequency structure from the reference onto the target.

### 3. Strong edge-aware gating
The node creates masks that reduce correction strength near:
- strong edges
- structural disagreement
- local detail mismatch
- unstable regions

This is one of its main strengths. It helps reduce:
- halos
- contour contamination
- false illumination shifts
- color spill near edges

### 4. Base/detail luminance transfer
Luminance is handled more carefully than in simpler methods.

Instead of just pushing the full luminance toward the reference, the node separates it conceptually into:
- **base luminance**
- **detail luminance**

Then it allows stronger luminance matching in large, smooth fields while preserving local target detail. This is important in tiled upscaling workflows where you want the background and large surfaces to become more uniform, without bringing back classic wavelet-style edge artifacts.

### 5. Saturation protection
To reduce gray or washed-out regions, the node includes a saturation safeguard that helps preserve chroma where needed.

---

## ✅ Why it can work better than Wavelet or AdaIN

### Compared to Wavelet
**Wavelet** often preserves color character very well, but it can become unstable when the reference and target do not share the same structure.

Typical problems:
- haloing
- brightness contamination along edges
- local tonal spill
- detail mismatch artifacts

**Color Fix Adaptive Chroma Fusion** reduces those issues by:
- restricting correction near unstable edges
- focusing wavelet influence on low/mid frequency chroma
- handling luminance separately
- combining wavelet with other multiscale cues instead of relying on it alone

### Compared to AdaIN
**AdaIN** is useful for broad color adaptation, but it often behaves like a statistical average.

Typical problems:
- flatter color character
- weaker local color fidelity
- washed-out or over-averaged appearance

**Color Fix Adaptive Chroma Fusion** improves on that by:
- keeping stronger local color identity
- using multiscale deltas instead of only global moments
- protecting saturation
- preserving more of the target structure

### Compared to simple Lab / histogram / mean-std approaches
These methods can be useful for rough matching, but they usually lack:
- structural awareness
- edge protection
- adaptive local confidence
- distinction between global and local correction

This node was designed specifically to address those limitations.

---

## 🚀 Main strengths

- 🎯 Better balance between **global consistency** and **local fidelity**
- 🧩 Useful for **tiled upscale** pipelines where background color drift can happen
- 🛡️ Stronger **edge protection** than naive color transfer methods
- 🌈 Better preservation of chroma than broad average-based methods
- 🌓 More controlled luminance behavior
- 🧪 Designed through iterative testing against practical failure cases

---

## 🖼️ Best use cases

This node is particularly useful for:

- tiled upscaling
- image enhancement pipelines
- restoration workflows
- reference-guided color harmonization
- cases where the target image is **similar in color intent** but **different in structure**
- situations where Wavelet gives artifacts and AdaIN gives bland results

---

## ⚙️ Inputs

### `image_ref`
Reference image that provides the desired color behavior.

### `image_target`
Target image that will receive the color correction.

### `color_strength`
Controls the overall color transfer strength.

### `edge_safety`
Controls how aggressively the node protects edges and structurally unstable areas.

### `local_detail`
Controls how much the local multiscale detail transfer contributes.

### `luma_match`
Controls how strongly the luminance base is aligned with the reference.

### `saturation_guard`
Controls how strongly saturation is preserved and recovered.

---

## 🧪 Recommended starting values

A strong starting preset found through practical testing is:

- `color_strength = 1.00`
- `edge_safety = 0.50`
- `local_detail = 1.00`
- `luma_match = 1.50`
- `saturation_guard = 1.00`

These values were selected as a good compromise between:
- reduced haloing
- stronger background consistency
- stable overall color transfer
- controlled luminance matching

---

## 📦 Installation

Clone or copy the repository into your `ComfyUI/custom_nodes` folder:

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

## 📚 Dependencies

This node uses:

- `numpy`
- `opencv-python`
- `PyWavelets`
- `torch`

---

## 🛠️ Project structure

Typical minimal structure:

```text
ComfyUI-Egregora-Adaptive-Colorfix/
├── __init__.py
├── egregora_adaptive_colorfix_node.py
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 📌 Notes

- The node name shown inside ComfyUI is:

**🎨 Color Fix Adaptive Chroma Fusion**

- The Python filename can be different from the display name.
- You can rename the module file as long as `__init__.py` imports it correctly.

---

## 🤝 Motivation

This node was built from repeated real-world testing in ComfyUI pipelines where classic color transfer methods were not enough.

The main objective was not just “match colors”, but to do so in a way that is more useful for AI image workflows where:
- structure changes
- tiles differ
- backgrounds shift
- edges break easily
- wavelet and statistical methods each solve only part of the problem

---

## 📄 License

See the `LICENSE` file in this repository.

---

## 🙌 Credits

Developed by **Egrégora Labs** for the ComfyUI workflows ecosystem.

If this node helps your workflow, consider starring the repository ⭐
