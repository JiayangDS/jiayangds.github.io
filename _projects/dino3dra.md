---
layout: page
title: DINO-3DRA
description: Leveraging 2D Foundation Model Semantics for 3D Cerebral Aneurysm Segmentation
img: assets/img/dino3dra_preview.png
importance: 1
category: research
related_publications: false
---

<div class="row justify-content-center text-center mt-2 mb-3">
  <div class="col-12">
    <span class="badge" style="background-color:#1565C0; color:white; font-size:0.85rem; padding:5px 14px; border-radius:4px;">MICCAI 2026</span>
  </div>
</div>

<div class="row justify-content-center text-center mb-3">
  <div class="col-md-8">
    <p class="text-muted mb-1">Author A &nbsp;·&nbsp; Author B &nbsp;·&nbsp; Author C</p>
    <p class="text-muted"><small>Anonymized Affiliations</small></p>
    <a href="#" class="btn btn-sm btn-primary mr-1">📄 Paper</a>
    <a href="#" class="btn btn-sm btn-dark mr-1">💻 GitHub</a>
    <a href="#" class="btn btn-sm btn-secondary">📦 arXiv</a>
  </div>
</div>

---

> We transfer **frozen 2D VFM semantics** into a 3D U-Net via slice-level embeddings, depth-coherence restoration, and calibrated residual fusion — achieving **+13% Dice** over nnU-Net with only **116K extra parameters** and **zero catastrophic failures** on cross-dataset evaluation.

---

<div class="row justify-content-center">
  <div class="col-sm-10 mt-2">
    {% include figure.liquid loading="eager" path="assets/img/dino3dra_fig1.png" title="DINO-3DRA Architecture" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  <strong>Fig. 1.</strong> (a) Frozen DINOv3-Small extracts per-slice semantic embeddings. (b) Multi-scale features are refined by Room-Lite Mixers and fused into U-Net skip connections via Calibrated Fusion.
</div>

Automated detection in 3D rotational angiography is extremely challenging. 2D VFMs know what anatomical structures look like but can't work in 3D; we build two cheap modules that safely bridge that knowledge into a 3D U-Net — the first framework where frozen 2D features genuinely improve volumetric medical segmentation instead of degrading it.

---

## The Challenge

**Ambiguity.** On a single 2D slice, vessels and aneurysms look identical — both are white parts on black. Only tracking shape across depth separates them.

**The 2D–3D gap.** 2D VFMs encode rich priors from 1.7B images. No 3D equivalent exists. But naïvely injecting 2D features into 3D models makes performance *worse* than no 2D features at all.

**Prior approaches fail.** Inflating 2D kernels destroys inter-slice continuity. Late concatenation ignores the distribution gap. Full spatial token propagation is memory prohibitive. No existing method addresses all three.

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

A frozen 2D VFM provides *what* each slice contains. A trainable 3D U-Net provides *where* structures are. Two lightweight modules bridge the gap.

**1 — Slice-Level Semantic Extraction.** Each axial slice processed by frozen DINOv3-Small. 196 patch tokens mean-pooled to one 384-dim embedding per slice. Projected via 1×1 convolutions and reshaped into 3D volumes at three U-Net skip scales.

**2 — Room-Lite Depth Coherence.** Learnable depth bias `B(z)` + depthwise-separable 3D conv restore inter-slice continuity. **21K params total.**

```
Y = X + GELU(GN(Conv₁ₓ₁(Conv₃ₓ₃(X + B(z)))))
```

**3 — Calibrated Residual Fusion.** GroupNorm aligns DINO to U-Net statistics. Residual bypass: if DINO is unhelpful, W → 0 and output is pure U-Net.

```
F_fused = F_U + GELU(GN(W[F_U; GN(F_D)]))
```

---

## Key Results

Four of five DINO variants fall *below* the plain U-Net baseline. Only the full pipeline surpasses it.

<div class="row justify-content-center my-3">
  <div class="col-12">
    <iframe
      src="{{ '/assets/html/dino3dra_fig3.html' | relative_url }}"
      width="100%"
      height="600"
      style="border: none; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);"
      loading="lazy"
      title="Ablation Study Interactive Visualization">
    </iframe>
  </div>
</div>
<div class="caption">
  <strong>Fig. 3.</strong> Interactive ablation visualization. Select a model variant to compare confidence maps. Ground truth (blue), predictions (red; darker = higher confidence).
</div>

---

## Cross-Dataset Generalisation

Evaluated on two external datasets without fine-tuning. DINO-3DRA eliminates all catastrophic failures (Dice < 0.5).

<div class="row text-center my-3">
  <div class="col-4">
    <h5 class="font-weight-bold mb-0">11.1% → 0%</h5>
    <small class="text-muted">Failures on CADA</small>
  </div>
  <div class="col-4">
    <h5 class="font-weight-bold mb-0">3.3% → 0%</h5>
    <small class="text-muted">Failures on SHINY-ICARUS</small>
  </div>
  <div class="col-4">
    <h5 class="font-weight-bold mb-0">75.8%</h5>
    <small class="text-muted">HD95 reduction (SI)</small>
  </div>
</div>

| Dataset | Model | Dice | Jaccard | Vol. Sim. | HD95 | Failures |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| CADA | Baseline | 0.711 ± 0.186 | 0.577 ± 0.186 | 0.813 ± 0.128 | 5.16 | 11.1% |
| CADA | **DINO-3DRA** | **0.739 ± 0.093** | **0.594 ± 0.118** | 0.807 ± 0.109 | **3.58** | **0%** |
| SI | Baseline | 0.718 ± 0.147 | 0.574 ± 0.129 | 0.808 ± 0.089 | 15.52 | 3.3% |
| SI | **DINO-3DRA** | **0.793 ± 0.049** | **0.660 ± 0.065** | 0.812 ± 0.056 | **3.76** | **0%** |

<div class="row justify-content-center">
  <div class="col-12 mt-2">
    {% include figure.liquid loading="eager" path="assets/img/dino3dra_fig4.png" title="Cross-dataset visual comparison" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  <strong>Fig. 4.</strong> Cross-dataset validation on CADA (left) and SHINY-ICARUS (right). Blue: vessel. Red: aneurysm. Columns: ground truth, baseline, DINO-3DRA.
</div>

---

## Limitations & Future Work

<div class="alert alert-warning" role="alert" style="border-left: 4px solid #e6a817;">
  <strong>⚠️ Known Limitations</strong><br>
  DINO-3DRA under-segments large or atypical aneurysms due to (1) training data bias toward saccular morphologies, and (2) patch boundary effects for aneurysms exceeding the 64³ input size. Future work: knowledge distillation, diverse morphology training, and multi-centre validation.
</div>

See the full paper for complete ablation analysis, per-case breakdowns, training protocol, and implementation details.

---

## BibTeX

```bibtex
@inproceedings{dino3dra2026,
  title     = {DINO-3DRA: Leveraging 2D Foundation Model
               Semantics for 3D Cerebral Aneurysm Segmentation},
  author    = {Anonymized Authors},
  booktitle = {MICCAI},
  year      = {2026}
}
```