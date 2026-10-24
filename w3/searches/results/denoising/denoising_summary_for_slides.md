# Denoising experiment summary

## Experiment

- Matched QSD1-W3 images: **30**.
- Flagged noisy IDs: **[5, 6, 16, 18, 20, 23, 29]** (7 images).
- Filters were evaluated only on flagged images.
- Quality was measured against each image's matched clean reference.
- Metrics: PSNR (higher is better) and SSIM (higher is better).

## Best results

- Highest mean PSNR: **Median 3x3** (30.78 dB).
- Highest mean SSIM: **Median 3x3** (0.792).

## Comparison table

| Type | Filter | Mean PSNR (dB) | Δ PSNR | Mean SSIM | Δ SSIM | PSNR wins |
|---|---|---:|---:|---:|---:|---:|
| Baseline | No filter | 23.73 | +0.00 | 0.484 | +0.000 | 0/7 |
| Linear | Mean 3x3 | 29.05 | +5.32 | 0.731 | +0.246 | 7/7 |
| Linear | Mean 5x5 | 28.47 | +4.75 | 0.724 | +0.240 | 6/7 |
| Linear | Gaussian 3x3 | 29.12 | +5.39 | 0.730 | +0.245 | 7/7 |
| Linear | Gaussian 5x5 | 29.44 | +5.71 | 0.760 | +0.275 | 7/7 |
| Nonlinear | Median 3x3 | 30.78 | +7.05 | 0.792 | +0.307 | 7/7 |
| Nonlinear | Median 5x5 | 30.27 | +6.54 | 0.770 | +0.285 | 6/7 |

## Interpretation and limitation

The paired clean references make PSNR and SSIM comparisons possible.
The residual-sigma noise check also uses those references; it is a
development-set analysis step, not a detector that can be applied
to future images when their clean references are unavailable.
