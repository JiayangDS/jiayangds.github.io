---
layout: page
title: DINO-3DRA
description: Leveraging 2D Foundation Model Semantics for 3D Cerebral Aneurysm Segmentation
img: assets/img/dino3dra_preview.png
importance: 1
category: work
related_publications: false
---

<!-- ─── HERO ─────────────────────────────────────────────────── -->
<div class="text-center mt-2 mb-2">
  <span class="badge" style="background:#1565C0;color:white;font-size:0.85rem;padding:5px 14px;border-radius:4px;letter-spacing:.04em;">MICCAI 2026</span>
</div>

<div class="text-center mb-3">
  <p class="text-muted mb-1">Author A &nbsp;·&nbsp; Author B &nbsp;·&nbsp; Author C</p>
  <p class="text-muted mb-3"><small>Anonymized Affiliations</small></p>
  <a href="#" class="btn btn-sm btn-primary mr-1">📄 Paper</a>
  <a href="#" class="btn btn-sm btn-dark mr-1">💻 GitHub</a>
  <a href="#" class="btn btn-sm btn-secondary">📦 arXiv</a>
</div>

<div class="row justify-content-center mb-2">
  <div class="col-md-9">
    <blockquote class="blockquote text-center" style="font-size:0.97rem;border-left:none;border-top:2px solid #1565C0;border-bottom:2px solid #1565C0;padding:12px 0;">
      We transfer <strong>frozen 2D VFM semantics</strong> into a 3D U-Net via slice-level embeddings, depth-coherence restoration, and calibrated residual fusion — achieving <strong>+13% Dice</strong> over nnU-Net with only <strong>116K extra parameters</strong> and <strong>zero catastrophic failures</strong> on cross-dataset evaluation.
    </blockquote>
  </div>
</div>

<!-- ─── KEY NUMBERS ──────────────────────────────────────────── -->
<div class="row text-center my-3" style="border-top:1px solid #eee;border-bottom:1px solid #eee;padding:16px 0;">
  <div class="col-3">
    <h3 class="font-weight-bold mb-0" style="color:#1565C0;">0.758</h3>
    <small class="text-muted">Aneurysm Dice</small>
  </div>
  <div class="col-3">
    <h3 class="font-weight-bold mb-0" style="color:#1565C0;">+13%</h3>
    <small class="text-muted">over nnU-Net</small>
  </div>
  <div class="col-3">
    <h3 class="font-weight-bold mb-0" style="color:#1565C0;">116K</h3>
    <small class="text-muted">Extra params</small>
  </div>
  <div class="col-3">
    <h3 class="font-weight-bold mb-0" style="color:#1565C0;">0%</h3>
    <small class="text-muted">Failure rate</small>
  </div>
</div>

---

<!-- ─── ARCHITECTURE ─────────────────────────────────────────── -->
<div class="row justify-content-center">
  <div class="col-sm-11 mt-2">
    {% include figure.liquid loading="eager" path="assets/img/dino3dra_fig1.png" title="DINO-3DRA Architecture" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  <strong>Fig. 1.</strong> Overview of DINO-3DRA. (a) Frozen DINOv3-Small extracts per-slice semantic embeddings from 2D axial slices. (b) Embeddings are broadcast to multi-scale spatial supports, refined by Room-Lite Mixers for depth coherence, and injected into U-Net skip connections via Calibrated Fusion.
</div>

---

## Abstract

Accurate aneurysm segmentation in 3D rotational angiography (3DRA) is hindered by extreme class imbalance, morphological similarity to vessels, and absent large-scale 3D pretraining. 2D vision foundation models encode dense structural priors from ~1.7 billion images, yet naïve slice-wise transfer fragments anatomical continuity and destabilises optimisation.

We propose DINO-3DRA, a dual-path framework achieving effective cross-dimensional semantic transfer by injecting frozen DINOv3 features into a 3D U-Net via Room-Lite spatial mixing and calibrated residual fusion. On multi-centre 3DRA data, DINO-3DRA achieves state-of-the-art aneurysm segmentation (Dice: 0.758; HD95: 2.75 mm; +13% over nnU-Net) with only 5.72M trainable parameters. Ablation studies confirm that gains arise from structured cross-dimensional transfer rather than loss design alone. Cross-dataset evaluation eliminates all catastrophic failure cases, demonstrating robust generalisation across heterogeneous imaging protocols.

---

## The Challenge

**Ambiguity.** 3DRA produces near-binary contrast — bright vessel lumen on dark background. On any single 2D slice, both a vessel cross-section and an aneurysm appear identical: a white circle on black. Distinguishing them requires observing how the cross-sectional shape changes across consecutive depth slices — a vessel maintains a consistent tube, while an aneurysm bulges and contracts.

**The 2D–3D gap.** The best 2D models (DINOv3) are trained on 1.7B images with rich structural priors. No equivalent large-scale 3D pretraining exists. But naïvely injecting 2D features into 3D models fails — it makes performance *worse* than using no 2D features at all.

<div class="row justify-content-center">
  <div class="col-sm-10 mt-2">
    {% include figure.liquid loading="eager" path="assets/img/dino3dra_sample.gif" title="3D Rotational Angiography" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  <strong>Fig. 2.</strong> 3D rotational angiography sample. Left: ground truth label. Right: raw image.
</div>

---

## Method

DINO-3DRA pairs a frozen 2D VFM with a trainable 3D U-Net. The frozen branch provides *what kind of anatomy* is in each slice. The trainable branch provides *where* exactly structures are. Two lightweight modules bridge the gap.

**1 — Slice-Level Semantic Extraction.** All 64 axial slices processed independently by frozen DINOv3-Small. 196 patch tokens globally averaged to a single 384-dim semantic embedding per slice. Projected via 1×1 convolutions and stacked into 3D volumes at three U-Net skip scales.

```
(B×64, 196, 384) → mean-pool → (B×64, 384) → (B, {24/48/96}, {64³/32³/16³})
```

**2 — Room-Lite Depth Coherence.** Learnable depth-positional bias `B(z)` + depthwise-separable 3D convolutions restore inter-slice continuity. Without this, Dice collapses −49%. **21K params total.**

```
Y = X + GELU(GN(Conv₁ₓ₁(Conv₃ₓ₃(X + B(z)))))
```

**3 — Calibrated Residual Fusion.** GN-normalised DINO features injected as a learnable residual correction into U-Net skips. The residual bypass guarantees DINO can only help, never hurt.

```
F_fused = F_U + GELU(GN(W[F_U; GN(F_D)]))
```

---

## Results

### Ablation Study

Four of five DINO variants fall *below* the plain U-Net baseline. Only the full pipeline — Room-Lite + Calibrated Fusion — surpasses it.

| Configuration | Aneurysm Dice | HD95 (mm) | Δ vs Baseline |
|:---|:---:|:---:|:---:|
| **Ours (Full)** | **0.758 ± 0.234** | **2.75** | **+0.047** |
| Baseline U-Net | 0.711 ± 0.240 | 4.20 | — |
| Random Features | 0.628 ± 0.279 | 6.15 | −0.083 |
| DINOv2-small | 0.590 ± 0.252 | 6.50 | −0.121 |
| w/o Calibration | 0.492 ± 0.301 | 10.12 | −0.219 |
| w/o Room-Lite | 0.385 ± 0.310 | 22.08 | −0.326 |

<div class="row justify-content-center my-3">
  <div class="col-12">
    <iframe
      src="{{ '/assets/html/dino3dra_fig3.html' | relative_url }}"
      width="100%"
      height="600"
      style="border:none;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.12);"
      loading="lazy"
      title="Ablation Study Interactive Visualization">
    </iframe>
  </div>
</div>
<div class="caption">
  <strong>Fig. 4.</strong> Interactive vessel confidence maps across ablation variants. Ground truth (semi-transparent blue), predictions (red; darker = higher confidence). Full DINO-3DRA produces homogeneous confidence across the aneurysm dome, recognising aneurysms as pathological vessel dilations rather than isolated anomalies.
</div>

---

## Cross-Dataset Generalisation

Evaluated on two external datasets without fine-tuning. DINO-3DRA eliminates all catastrophic failures (Dice < 0.5).

<div class="row text-center my-3" style="border-top:1px solid #eee;border-bottom:1px solid #eee;padding:14px 0;">
  <div class="col-4">
    <h5 class="font-weight-bold mb-0" style="color:#1565C0;">11.1% → 0%</h5>
    <small class="text-muted">Failures on CADA</small>
  </div>
  <div class="col-4">
    <h5 class="font-weight-bold mb-0" style="color:#1565C0;">3.3% → 0%</h5>
    <small class="text-muted">Failures on SHINY-ICARUS</small>
  </div>
  <div class="col-4">
    <h5 class="font-weight-bold mb-0" style="color:#1565C0;">75.8%</h5>
    <small class="text-muted">HD95 reduction (SI)</small>
  </div>
</div>

| Dataset | Model | Dice | Jaccard | Vol. Sim. | HD95 | Failures |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| CADA | Baseline | 0.711 ± 0.186 | 0.577 ± 0.186 | 0.813 ± 0.128 | 5.16 | 11.1% |
| CADA | **DINO-3DRA** | **0.739 ± 0.093** | **0.594 ± 0.118** | 0.807 ± 0.109 | **3.58** | **0%** |
| SHINY-ICARUS | Baseline | 0.718 ± 0.147 | 0.574 ± 0.129 | 0.808 ± 0.089 | 15.52 | 3.3% |
| SHINY-ICARUS | **DINO-3DRA** | **0.793 ± 0.049** | **0.660 ± 0.065** | 0.812 ± 0.056 | **3.76** | **0%** |

<div class="row justify-content-center">
  <div class="col-12 mt-2">
    {% include figure.liquid loading="eager" path="assets/img/dino3dra_fig4.png" title="Cross-dataset visual comparison" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  <strong>Fig. 5.</strong> Cross-dataset validation on CADA (left) and SHINY-ICARUS (right). Blue: vessel. Red: aneurysm. Columns: ground truth, baseline, DINO-3DRA.
</div>

---

## BibTeX

```bibtex
@inproceedings{dino3dra2026,
  title     = {DINO-3DRA: Leveraging 2D Foundation Model
               Semantics for 3D Cerebral Aneurysm Segmentation},
  author    = {Anonymized Authors},
  booktitle = {International Conference on Medical Image Computing
               and Computer-Assisted Intervention (MICCAI)},
  year      = {2026}
}
```