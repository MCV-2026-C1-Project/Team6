# Week 3 denoising: decision for slides

## Decision

**Selected filter: Mean 5x5.**

On this QSD1 run, it achieved mAP@1 = **0.7333** and mAP@5 = **0.7833**.

The unfiltered baseline was mAP@1 = 0.6667 and mAP@5 = 0.7444. The selected setup changed the scores by +0.0667 mAP@1 and +0.0389 mAP@5.

Several filters tied on both retrieval scores: Mean 5x5, Non-local means h=10, Non-local means h=15. The simpler filter, Mean 5x5, was selected as the tie-break.

## Test method

- Dataset: 30 QSD1 development queries with correspondence ground truth.
- Noise detector: flag a query when more than 1% of grayscale pixels differ by more than 30 intensity levels from all eight neighbors.
- Flagged query IDs in this run: [5, 6, 16, 18, 20, 23, 29].
- Each candidate filter was applied only to flagged queries. Unflagged queries and BBDD images were left unchanged.
- The Week 2 color-light descriptor and histogram-intersection distance were kept fixed. The comparison metric was retrieval mAP.

## Limitation

This experiment measures retrieval performance; it does not calculate PSNR or SSIM. Those pixelwise measures require paired clean reference images. The comparison uses the Week 2 descriptor, so repeat the evaluation with the final Week 3 descriptor before reporting the team's final result.

## Short presentation wording

“We compared the unfiltered baseline with eight filters on the 7 detected noisy development queries. Mean 5x5 gave the best retrieval result (mAP@5 0.7833 versus 0.7444 without filtering). We selected it because it improved retrieval and, where scores tied, it was the simpler option.”
