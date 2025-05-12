### 🧬 Vestibular Schwannoma Segmentation

```markdown
# Vestibular Schwannoma Segmentation

An ML pipeline for automatic segmentation of vestibular schwannomas (VS) from 3D MRI scans using size-aware data augmentation and transformer-enhanced nnUNet architectures.

## 🎯 Highlights
- 2.5D UNet with Transformer bottleneck + gated residuals
- Conditional Diffusion (Med-DDPM) for generating rare tumor samples
- Cross-modality fusion decoder (T1 + T2)
- Dice score: 0.926 on T1, 0.906 on T2

## 🧪 Data
- Public VS-SEG dataset (242 patient scans)
- T1-weighted and T2-weighted MRIs
- Tumor contours in JSON, converted to NIfTI using 3DSlicer
