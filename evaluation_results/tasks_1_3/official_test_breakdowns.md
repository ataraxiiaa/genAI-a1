# Official Oxford-IIIT Pet Test Breakdowns

Each row contains all 3,669 images from the untouched official test split. Corruptions come from the deterministic manifests generated with seed 42.

## Task 1: Universal Restoration by Severity

| Condition | Severity | L1 | SSIM | PSNR (dB) |
|---|---:|---:|---:|---:|
| clean | none | 0.0396 | 0.7817 | 25.47 |
| salt_and_pepper | low | 0.0405 | 0.7662 | 25.28 |
| salt_and_pepper | medium | 0.0425 | 0.7407 | 24.87 |
| salt_and_pepper | high | 0.0460 | 0.7035 | 24.17 |
| gaussian_blur | low | 0.0391 | 0.7829 | 25.60 |
| gaussian_blur | medium | 0.0399 | 0.7679 | 25.42 |
| gaussian_blur | high | 0.0442 | 0.7166 | 24.42 |
| rectangular_occlusion | low | 0.0476 | 0.7357 | 23.00 |
| rectangular_occlusion | medium | 0.0551 | 0.6959 | 21.47 |
| rectangular_occlusion | high | 0.0663 | 0.6445 | 19.78 |

## Task 3: Soft-MoE Routing by Severity

| Condition | Severity | Clean | Noise | Blur | Occlusion | Dominant branch | SSIM | PSNR (dB) |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| clean | none | 0.9301 | 0.0009 | 0.0564 | 0.0126 | clean_identity | 0.9960 | 60.24 |
| salt_and_pepper | low | 0.0006 | 0.9988 | 0.0001 | 0.0004 | noise_expert | 0.8148 | 27.14 |
| salt_and_pepper | medium | 0.0002 | 0.9995 | 0.0001 | 0.0002 | noise_expert | 0.8016 | 26.82 |
| salt_and_pepper | high | 0.0001 | 0.9997 | 0.0000 | 0.0002 | noise_expert | 0.7742 | 26.10 |
| gaussian_blur | low | 0.1753 | 0.0003 | 0.8221 | 0.0022 | blur_expert | 0.8644 | 28.78 |
| gaussian_blur | medium | 0.0157 | 0.0001 | 0.9837 | 0.0005 | blur_expert | 0.8289 | 27.85 |
| gaussian_blur | high | 0.0174 | 0.0001 | 0.9819 | 0.0006 | blur_expert | 0.7633 | 26.16 |
| rectangular_occlusion | low | 0.1544 | 0.0022 | 0.0075 | 0.8359 | occlusion_expert | 0.8031 | 24.06 |
| rectangular_occlusion | medium | 0.0223 | 0.0006 | 0.0015 | 0.9756 | occlusion_expert | 0.7358 | 22.45 |
| rectangular_occlusion | high | 0.0044 | 0.0002 | 0.0004 | 0.9950 | occlusion_expert | 0.6805 | 20.53 |
