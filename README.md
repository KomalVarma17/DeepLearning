# Diffusion-Based Image Compression Using ResNet Encoder and U-Net Decoder

## 1. Introduction

This project implements an image compression model using a conditional diffusion process. The aim is to reduce the storage size of images while preserving perceptual quality. The model combines a ResNet-based encoder, a forward diffusion process that adds Gaussian noise, and a U-Net-based decoder that learns to denoise and reconstruct the image.

---

## 2. Architecture Overview

The model consists of three major components:

- **Encoder**: A convolutional ResNet architecture that compresses the image into a compact latent representation.
- **Diffusion Process**: Applies noise to the latent representation during training, simulating degradation.
- **Decoder**: A U-Net-style network that reconstructs the image from the noisy latent using learned denoising and skip connections.

---

## 3. Encoder Details

The encoder downsamples the image while extracting features:

- **Input**: 256x256x3 image
- **Initial Conv Layer**: 7x7 kernel, stride 2 → Output: 128x128x64
- **ResNet Block 1**: Two 3x3 convs → Output: 128x128x64
- **Downsample**: 3x3, stride 2 → Output: 64x64x128
- **ResNet Block 2**: Two 3x3 convs → Output: 64x64x128
- **Downsample**: 3x3, stride 2 → Output: 32x32x128
- **ResNet Block 3**: Two 3x3 convs → Output: 32x32x128

Skip connections are preserved at each ResNet block for the decoder.

---

## 4. Diffusion Process

During training, controlled Gaussian noise is added to the latent representation:

- **Beta Schedule**: Linearly increases from 1e-4 to 0.02 over 1000 steps
- **Alphas**: Computed as 1 - beta
- **Alpha Cumprod**: Used for noise scaling
- **q_sample Function**: x_n = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
- This simulates a forward diffusion process. The decoder learns to reverse this process to reconstruct the original image.

---

## 5. Decoder Details

The decoder upsamples the noisy latent back to image size:

- **Input**: Noisy latent (32x32x128)
- **Up Block 1**: 128→128, concat with skip3
- **Up Block 2**: 256→128, concat with skip2
- **Up Block 3**: 192→64, concat with skip1
- **Final Conv**: 128→3 → Output: 256x256x3

Transposed convolutions, BatchNorm, and ReLU are used in each block. Skip connections are upsampled to match resolution.

---

## 6. Dataset

- **Input**: 256x256 JPEG images from various categories
- **Training Set**: Used to learn encoder and decoder
- **Validation Set**: Used to track performance
- **Test Set**: 5 manually selected images to evaluate quality

---

## 7. Training Details

- **Epochs**: 100
- **Batch Size**: 32
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Validation Checkpoint**: Every 50 epochs
- **Noise Sampling**: Random timestep `t` selected per batch
- **Inference**: No noise added at test time

---

## 8. Experimental Results

### Compression Performance

- **Memory-Based Compression Ratio**:
  - Input: 4.3200 MB
  - Latent: 1.0486 MB
  - Ratio: 4.12x
- **Pixel-Based Compression Ratio**:
  - Input: 270,000 values
  - Latent: 131,072 values
  - Ratio: 2.06x

### Quality Metrics

- **Validation Loss**: 0.001180
- **Average PSNR**: 29.45 dB

### SSIM and MS-SSIM (Sample Images)

| Image                | SSIM   | MS-SSIM |
|---------------------|--------|---------|
| DlsOa5moK4w.jpg      | 0.9303 | 0.9832  |
| PXdbkNF8rlk.jpg      | 0.9118 | 0.9892  |
| XBGacbT3vXI.jpg      | 0.9289 | 0.9883  |

---

## 9. Conclusion

This project demonstrates that conditional diffusion models can be used effectively for semantic image compression. The ResNet encoder captures essential features, the diffusion process introduces robustness, and the U-Net decoder successfully reconstructs high-quality images. The model achieves strong perceptual quality (SSIM > 0.91, MS-SSIM > 0.98) and significant compression.
