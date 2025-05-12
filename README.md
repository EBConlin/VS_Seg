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

## 🔧 Setup
```bash
pip install niftynet tensorflow
```

## 🧠 Architecture
- nnUNet backbone
- Optional Transformer block between encoder/decoder
- Fusion Decoder uses pretrained Swin Transformer

## 🖼 Sample Outputs
- Synthetic tumors in rare size bins
- Segmentation visualizations coming soon

## 📝 Citation
Work based on: Poe, Wu, Conlin, Longhitano et al. (2024), BU NLP
